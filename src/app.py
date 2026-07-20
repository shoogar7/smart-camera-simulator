import cv2 as cv
import time
from camera import Camera
from roi_manager import ROIManager
from motion_detector import MotionDetector
from tracker import Tracker
from config import Config

class App:
    # orchestration of main loop
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
                self.camera.cap.release()
                break