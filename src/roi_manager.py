import cv2 as cv
import logging

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