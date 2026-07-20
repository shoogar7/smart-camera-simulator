import cv2 as cv
import logging
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

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