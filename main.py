import numpy as np
import cv2 as cv
import time
from ultralytics import YOLO
from collections import defaultdict
import logging
import yaml

class Config:    
    def __init__(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
            
        presets = config["presets"]
        variance = config["variance"]
        preprocess = config["preprocess"]
        motionManager = config["motionManager"]
        tracker = config["tracker"]
        timers = config["timers"]
            
        active_preset = config["active_preset"]
        self.source = presets[active_preset]["source"]
        self.close_view = presets[active_preset]["close_view"]
        
        if self.close_view:
            self.min_variance = variance["close"]["min"]
            self.max_variance = variance["close"]["max"]
        else:
            self.min_variance = variance["far"]["min"]
            self.max_variance = variance["far"]["max"]
        # preprocess
        self.dilatation_size = preprocess["dilatation_size"]
        self.dilatation_shape = getattr(cv, preprocess["dilatation_shape"], None)
        if self.dilatation_shape is None:
            raise AttributeError(f'Configuration Error:\n Unknown Dilatation Shape: "{preprocess["dilatation_shape"]}".')
        # motionManager
        self.motion_threshold = motionManager["motion_threshold"]
        self.small_cam_mov_thresh = motionManager["drop_cam_thresh"]  
        self.big_cam_mov_thresh = motionManager["shift_cam_thresh"]
        # tracker
        self.model_path = tracker["model_path"]
        self.track_classes = tracker["track_classes"]
        # timers
        self.cam_reset_check_time = timers["cam_reset_check_time"]
        self.motion_check_time = timers["motion_check_time"]
        self.drop_check_time = timers["drop_check_time"]
        self.gone_check_time = timers["gone_check_time"]        
        # logging
        self.debug = config["log"]["debug"]
        self.log_level = getattr(logging, config["log"]["log_level"], None)
        if self.log_level is None:
            raise AttributeError(f'Configuration Error:\n Unknown Log Level: "{config["log"]["log_level"]}".')

class Camera:
    # camera handling (open / restart / read / validate)
    def __init__(self, config):
        self.config = config
        self.source = config.source
        self.cap = None
        self.last_reset_time = time.time()
        self.last_gone_time = 0
        self.gone_check_time = config.gone_check_time
        
    def open(self):
        self.cap = cv.VideoCapture(self.source)
        if self.cap.isOpened():
            logging.info('Camera Started')
        else:
            logging.error('Camera Not Starting')
    
    def restart(self):
        logging.info('Attempting Camera Restart')
        self.cap.release()
        self.open()
    
    def read(self):
        ret, frame = self.cap.read()

        while not ret:
            if time.time() - self.last_reset_time > self.config.cam_reset_check_time:
                logging.warning('No Frames Received')
                self.last_reset_time = time.time()
                self.restart()
                ret, frame = self.cap.read()
        return frame
    
    def validate(self, frame):
        variance = np.var(frame)
        logging.debug(f'Variance Level {variance}')
        if variance == 0:
            logging.error('No Camera Signal')
            self.restart()
        elif variance < self.config.min_variance or variance > self.config.max_variance:  
            logging.warning('View Blocked')
            
class ROIManager:
    def __init__(self, config, p1=(0,0), p2=(0,0), state=0):
        self.config = config
        self.p1 = p1
        self.p2 = p2
        self.state = state
                
    def reset(self):
        self.p1 = (0,0)
        self.p2 = (0,0)
        self.state = 0
        logging.info('ROI Deleted')
        
    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv.EVENT_LBUTTONUP:
            if self.state == 0:
                self.p1 = (x, y)
                self.state += 1
            elif self.state == 1:
                self.p2 = (x, y)
                self.state += 1
                if self.p1 == self.p2: # if user clicks same place
                    self.reset()
                else:
                    logging.info('ROI Set Correctly')
            else: 
                self.reset()
        elif event == cv.EVENT_RBUTTONUP or event == cv.EVENT_LBUTTONDBLCLK:
            self.reset()
            
    def get_roi(self, ready_frame, real_frame):
        # normalization - if user first clicks bottom-right instead of top-left
        x1 = min(self.p1[0], self.p2[0])
        x2 = max(self.p1[0], self.p2[0])
        y1 = min(self.p1[1], self.p2[1])
        y2 = max(self.p1[1], self.p2[1])
        
        roi = ready_frame[y1:y2, x1:x2] # [row:row, col:col]
        real_roi = real_frame[y1:y2, x1:x2]
        return roi, real_roi
    
    def has_roi(self):
        if self.state == 2:
            return True
        return False

class MotionDetector:
    # motion + optical flow
    def __init__(self, config):
        self.config = config
        self.motion_threshold = self.config.motion_threshold
        self.small_cam_mov_thresh = self.config.small_cam_mov_thresh
        self.big_cam_mov_thresh = self.config.big_cam_mov_thresh
        
        self.dilatation_shape = self.config.dilatation_shape
        self.dilatation_size = self.config.dilatation_size
        
        self.prev_points = None
        self.defined_points = None
        self.defined_frame = None
        
        self.motion_check_time = self.config.motion_check_time
        self.drop_check_time = self.config.drop_check_time
        
        self.last_motion_time = time.time()
        self.last_drop_time = time.time()
        
    def initialize(self, cur_frame):
        self.prev_points = cv.goodFeaturesToTrack(cur_frame, 50, 0.01, 7) # points for camera shift detection
        self.defined_points = cv.goodFeaturesToTrack(cur_frame, 100, 0.01, 7) # points for camera drop detection
        self.defined_frame = cur_frame
    
    def preprocess(self, cur_frame, prev_frame):
        dest_frame = cv.absdiff(cur_frame, prev_frame)
        threshold = cv.threshold(dest_frame, 63, 255, cv.THRESH_BINARY)

        element = cv.getStructuringElement(self.dilatation_shape, 
                                           (2 * self.dilatation_size + 1, 2 * self.dilatation_size + 1), 
                                           (self.dilatation_size, self.dilatation_size))
        dilatation_dst = cv.dilate(threshold[1], element)

        contours, _ = cv.findContours(dilatation_dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(dilatation_dst, contours, -1, (0, 255, 0), 3)
        
        return dilatation_dst
    
    def detect_motion(self, det_roi):
        motion_level = np.count_nonzero(det_roi) 
        if motion_level > self.motion_threshold:
            self.last_motion_time = time.time()
            logging.debug(f'Motion Level {motion_level}')
            logging.warning('Motion Detected in ROI')
            return True
        return False
    
    def detect_camera_shift(self, prev_gray, cur_gray):
        threshold = self.big_cam_mov_thresh
        
        cur_points, status, err = cv.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, self.prev_points, None, 
            winSize=(15, 15), maxLevel=2, 
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        
        good_new = cur_points[status==1] # tracking succesful
        good_old = self.prev_points[status==1]
        
        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]
        
        mean_dx = np.mean(dx)
        mean_dy = np.mean(dy)

        if len(self.prev_points) < 20:
            self.prev_points = cv.goodFeaturesToTrack(prev_gray, 50, 0.01, 7)
        else:
            self.prev_points = good_new.reshape(-1, 1, 2)
            
        if abs(mean_dx) > threshold or abs(mean_dy) > threshold:
            logging.debug(f"mean_dx: {abs(mean_dx)} | mean_dy: {abs(mean_dy)}")
            logging.warning('Camera Shifted')
            return True
        
        return False
            
    def detect_camera_drop(self, cur_frame):
        threshold = self.small_cam_mov_thresh
        cur_points, status, err = cv.calcOpticalFlowPyrLK(
            self.defined_frame, cur_frame, self.defined_points, None, 
            winSize=(15, 15), maxLevel=2, 
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        
        good_new = cur_points[status==1] # tracking succesful
        good_old = self.defined_points[status==1]
        
        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]
        
        mean_dx = np.mean(dx)
        mean_dy = np.mean(dy)

        if abs(mean_dx) > threshold or abs(mean_dy) > threshold:
            self.last_drop_time = time.time()
            logging.debug(f"mean_dx: {abs(mean_dx)} | mean_dy: {abs(mean_dy)}")
            if mean_dx > threshold:
                direction = "RIGHT"
            elif mean_dx < -threshold:
                direction = "LEFT"
            elif mean_dy > threshold:
                direction = "UP"
            elif mean_dy < -threshold:
                direction = "DOWN"
            else:
                direction = "UNKNOWN"
            logging.warning(f'Camera Dropped {direction}')

class Tracker:
    # YOLO tracking
    def __init__(self, config):
        self.config = config
        self.track_history = defaultdict(lambda: [])
        self.model = YOLO(self.config.model_path)
        self.track_classes = self.config.track_classes

    def tracking(self, frame): # better to track on the whole frame, not just ROI
        track_result = self.model.track(frame, persist=True, classes=self.track_classes)[0]

        if track_result.boxes and track_result.boxes.is_track:
            logging.info('Tracking')
            track_boxes = track_result.boxes.xywh.cpu()
            track_ids = track_result.boxes.id.int().cpu().tolist()

            frame = track_result.plot()

            # plot the tracks
            for box, track_id in zip(track_boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))  # (x, y) of center point
                
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        return frame
    
class App:
    # orchestration of main loop
    def __init__(self, config):
        self.config = config
        self.camera = Camera(config)
        self.roi_manager = ROIManager(config)
        self.motion_detector = MotionDetector(config)
        self.tracker = Tracker(config)
        
        self.start = time.time()
        self.fps = None
        
    def calc_fps(self):
        self.stop = time.time()
        self.fps = 1 / (self.stop - self.start)
        self.start = self.stop
    
    def run(self):
        cv.namedWindow('Normal View')
        cv.namedWindow('Detection View')
        cv.setMouseCallback('Normal View', self.roi_manager.on_mouse)
        
        self.camera.open()        
        frame = self.camera.read()
        blur = cv.GaussianBlur(frame, (3, 3), 0)
        cur_frame = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

        self.motion_detector.initialize(cur_frame)
        
        while True:
            prev_frame = cur_frame
            frame = self.camera.read()
            
            self.calc_fps()

            blur = cv.GaussianBlur(frame, (3, 3), 0)
            cur_frame = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

            if time.time() - self.camera.last_gone_time > self.camera.gone_check_time:
                self.camera.last_gone_time = time.time()
                self.camera.validate(frame)

            detection_frame = self.motion_detector.preprocess(cur_frame, prev_frame)
            
            if not self.roi_manager.has_roi():
                det_roi, real_roi = detection_frame, frame
            else:
                cv.rectangle(frame, self.roi_manager.p1, self.roi_manager.p2, (255, 0, 0), 1)
                det_roi, real_roi = self.roi_manager.get_roi(detection_frame, frame)

            if time.time() - self.motion_detector.last_motion_time > self.motion_detector.motion_check_time:
                if self.motion_detector.detect_motion(det_roi):
                    if not self.motion_detector.detect_camera_shift(prev_frame, cur_frame):
                        frame = self.tracker.tracking(frame)
            if time.time() - self.motion_detector.last_drop_time > self.motion_detector.drop_check_time:
                self.motion_detector.detect_camera_drop(cur_frame)

            cv.imshow('Detection View', detection_frame)
            cv.putText(frame, f"{frame.shape[1]}x{frame.shape[0]} FPS:{int(self.fps)}", (100, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv.imshow('Normal View', frame)

            if cv.waitKey(30) == ord('q'): # in ms - 1/1000s 
                break
    
def main():
    config = Config()
    logging.basicConfig(level=config.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    app = App(config)
    app.run()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()