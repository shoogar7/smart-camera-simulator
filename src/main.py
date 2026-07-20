import cv2 as cv
import logging
import config
import app

def main():
    cfg = config.Config()
    logging.basicConfig(level=cfg.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    app.App(cfg).run()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()