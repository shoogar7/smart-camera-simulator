import cv2 as cv
import time
from camera import Camera
from roi_manager import ROIManager
from motion_detector import MotionDetector
from tracker import Tracker
from config import Config

# orchestration of main loop
class App:    
    WAITKEY_MS = 30
    MAIN_WINDOW_NAME = 'Normal View'
    DETECTION_WINDOW_NAME = 'Detection View'
    FPS_TXT_POSITION = (100, 90)
    FPS_TXT_SCALE = 0.5
    FPS_TXT_COLOR = (0, 0, 0)
    FPS_TXT_THICKNESS = 2
    ROI_RECT_COLOR = (255, 0, 0)
    ROI_RECT_THICK = 1
    
    GAUSSIAN_KERNEL = (3, 3)
    GAUSSIAN_SIGMA = 0    
    
    def __init__(self, config: Config):
        self.config = config
        self.camera = Camera(config)
        self.roi_manager = ROIManager(config)
        self.motion_detector = MotionDetector(config)
        self.tracker = Tracker(config)
        
        self.start = time.time()
        self.fps = None
        
    def calc_fps(self) -> None:
        self.stop = time.time()
        self.fps = 1 / (self.stop - self.start)
        self.start = self.stop
    
    def run(self) -> None:
        cv.namedWindow(App.MAIN_WINDOW_NAME)
        cv.namedWindow(App.DETECTION_WINDOW_NAME)
        cv.setMouseCallback(App.MAIN_WINDOW_NAME, self.roi_manager.on_mouse)
        
        self.camera.open()        
        frame = self.camera.read()
        blur = cv.GaussianBlur(frame, App.GAUSSIAN_KERNEL, App.GAUSSIAN_SIGMA)
        cur_frame = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

        self.motion_detector.initialize(cur_frame)
        
        while True: 
            prev_frame = cur_frame
            frame = self.camera.read()
            
            self.calc_fps()

            blur = cv.GaussianBlur(frame, App.GAUSSIAN_KERNEL, App.GAUSSIAN_SIGMA)
            cur_frame = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

            if time.time() - self.camera.last_gone_time > self.camera.gone_check_time:
                self.camera.last_gone_time = time.time()
                self.camera.validate(frame)

            detection_frame = self.motion_detector.preprocess(cur_frame, prev_frame)
            
            if not self.roi_manager.has_roi():
                det_roi, real_roi = detection_frame, frame
            else:
                cv.rectangle(frame, self.roi_manager.p1, self.roi_manager.p2, App.ROI_RECT_COLOR, App.ROI_RECT_THICK)
                det_roi, real_roi = self.roi_manager.get_roi(detection_frame, frame)

            if time.time() - self.motion_detector.last_motion_time > self.motion_detector.motion_check_time:
                if self.motion_detector.detect_motion(det_roi):
                    if not self.motion_detector.detect_camera_shift(prev_frame, cur_frame):
                        frame = self.tracker.tracking(frame)
            if time.time() - self.motion_detector.last_drop_time > self.motion_detector.drop_check_time:
                self.motion_detector.detect_camera_drop(cur_frame)

            cv.imshow(App.DETECTION_WINDOW_NAME, detection_frame)
            cv.putText(frame, f"{frame.shape[1]}x{frame.shape[0]} FPS:{int(self.fps)}", 
                       App.FPS_TXT_POSITION, cv.FONT_HERSHEY_SIMPLEX, App.FPS_TXT_SCALE, 
                       App.FPS_TXT_COLOR, App.FPS_TXT_THICKNESS) 
            cv.imshow(App.MAIN_WINDOW_NAME, frame)

            if cv.waitKey(App.WAITKEY_MS) == ord('q'): # in ms - 1/1000s 
                self.camera.cap.release()
                break