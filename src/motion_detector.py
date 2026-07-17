import cv2 as cv
import numpy as np
import logging
import time

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