import cv2 as cv
import numpy as np
import logging
import time
from config import Config

class Camera:
    # camera handling (open / restart / read / validate)
    def __init__(self, config: Config):
        self.config = config
        self.source = config.source
        self.cap = None
        self.last_reset_time = time.time()
        self.last_gone_time = 0
        self.gone_check_time = config.gone_check_time
        
    def open(self) -> None:
        self.cap = cv.VideoCapture(self.source)
        if self.cap.isOpened():
            logging.info('Camera Started')
        else:
            logging.error('Camera Not Starting')
    
    def restart(self) -> None:
        logging.info('Attempting Camera Restart')
        self.cap.release()
        self.open()
    
    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()

        while not ret:
            if time.time() - self.last_reset_time > self.config.cam_reset_check_time:
                logging.warning('No Frames Received')
                self.last_reset_time = time.time()
                self.restart()
                ret, frame = self.cap.read()
        return frame
    
    def validate(self, frame: np.ndarray) -> None:
        variance = np.var(frame)
        logging.debug(f'Variance Level {variance}')
        if variance == 0:
            logging.error('No Camera Signal')
            self.restart()
        elif variance < self.config.min_variance or variance > self.config.max_variance:  
            logging.warning('View Blocked')