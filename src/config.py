import yaml
import cv2 as cv
import logging

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