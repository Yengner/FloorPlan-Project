# model_setup.py
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_predictor(config_path, weights_path, threshold=0.3):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.MODEL.DEVICE = "cpu"  # or "cuda" for GPU
    predictor = DefaultPredictor(cfg)
    return predictor, cfg  # Return both the predictor and the config
