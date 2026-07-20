import cv2 as cv
import logging
from config import Config
from app import App

def main():
    cfg = Config()
    logging.basicConfig(level=cfg.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    App(cfg).run()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()