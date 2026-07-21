import cv2 as cv
import numpy as np
import logging
import time
from config import Config

# motion + optical flow
class MotionDetector:
    LK_PARAMS = dict(winSize = (15,15), maxLevel = 2,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    PREV_POINT_LIMIT = 20
    CONTOUR_ID = -1
    CONTOUR_COLOR = (0, 255, 0)
    CONTOUR_THICKNESS = 3
    
    PREV_POINT_MAX_CORNERS = 50
    PREV_POINT_QUALITY_LEVEL = 0.01
    PREV_POINT_MIN_DISTANCE = 7
    
    DEFINED_POINT_MAX_CORNERS = 100
    DEFINED_POINT_QUALITY_LEVEL = 0.01
    DEFINED_POINT_MIN_DISTANCE = 7
    
    THRESHOLD_VALUE = 63
    THRESHOLD_MAX = 255

    def __init__(self, config: Config):
        self.config = config
        self.motion_threshold = config.motion_threshold
        self.small_cam_mov_thresh = config.small_cam_mov_thresh
        self.big_cam_mov_thresh = config.big_cam_mov_thresh
        
        self.dilatation_shape = config.dilatation_shape
        self.dilatation_size = config.dilatation_size
        
        self.prev_points = None
        self.defined_points = None
        self.defined_frame = None
        
        self.motion_check_time = config.motion_check_time
        self.drop_check_time = config.drop_check_time
        
        self.last_motion_time = time.time()
        self.last_drop_time = time.time()
        
    def initialize(self, cur_frame: np.ndarray) -> None:
        self.prev_points = cv.goodFeaturesToTrack(cur_frame, MotionDetector.PREV_POINT_MAX_CORNERS,
            MotionDetector.PREV_POINT_QUALITY_LEVEL, MotionDetector.PREV_POINT_MIN_DISTANCE) # points for camera shift detection
        self.defined_points = cv.goodFeaturesToTrack(cur_frame, MotionDetector.DEFINED_POINT_MAX_CORNERS,
            MotionDetector.DEFINED_POINT_QUALITY_LEVEL, MotionDetector.DEFINED_POINT_MIN_DISTANCE) # points for camera drop detection
        self.defined_frame = cur_frame
    
    def preprocess(self, cur_frame: np.ndarray, prev_frame: np.ndarray) -> np.ndarray:
        dest_frame = cv.absdiff(cur_frame, prev_frame)
        threshold = cv.threshold(dest_frame, MotionDetector.THRESHOLD_VALUE, 
                                 MotionDetector.THRESHOLD_MAX, cv.THRESH_BINARY)

        element = cv.getStructuringElement(self.dilatation_shape, 
                                           (2 * self.dilatation_size + 1, 2 * self.dilatation_size + 1), 
                                           (self.dilatation_size, self.dilatation_size))
        dilatation_dst = cv.dilate(threshold[1], element)

        contours, _ = cv.findContours(dilatation_dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(dilatation_dst, contours, MotionDetector.CONTOUR_ID, 
                        MotionDetector.CONTOUR_COLOR, MotionDetector.CONTOUR_THICKNESS)
        
        return dilatation_dst
    
    def detect_motion(self, det_roi: np.ndarray) -> bool:
        motion_level = np.count_nonzero(det_roi) 
        if motion_level > self.motion_threshold:
            self.last_motion_time = time.time()
            logging.debug(f'Motion Level {motion_level}')
            logging.warning('Motion Detected in ROI')
            return True
        return False
    
    def detect_camera_shift(self, prev_gray: np.ndarray, cur_gray: np.ndarray) -> bool:
        threshold = self.big_cam_mov_thresh
        
        cur_points, status, err = cv.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, self.prev_points, None, **MotionDetector.LK_PARAMS)
        
        good_new = cur_points[status==1] # tracking succesful
        good_old = self.prev_points[status==1]
        
        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]
        
        mean_dx = np.mean(dx)
        mean_dy = np.mean(dy)

        if len(self.prev_points) < MotionDetector.PREV_POINT_LIMIT:
            self.prev_points = cv.goodFeaturesToTrack(prev_gray, MotionDetector.PREV_POINT_MAX_CORNERS,
                MotionDetector.PREV_POINT_QUALITY_LEVEL, MotionDetector.PREV_POINT_MIN_DISTANCE) 
        else:
            self.prev_points = good_new.reshape(-1, 1, 2)
            
        if abs(mean_dx) > threshold or abs(mean_dy) > threshold:
            logging.debug(f"mean_dx: {abs(mean_dx)} | mean_dy: {abs(mean_dy)}")
            logging.warning('Camera Shifted')
            return True
        
        return False
            
    def detect_camera_drop(self, cur_frame: np.ndarray) -> None:
        threshold = self.small_cam_mov_thresh
        cur_points, status, err = cv.calcOpticalFlowPyrLK(
            self.defined_frame, cur_frame, self.defined_points, None, **MotionDetector.LK_PARAMS)
        
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