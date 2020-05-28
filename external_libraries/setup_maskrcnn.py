from maskrcnn_benchmark.config import cfg
from networks.maskrcnn import COCODemo

config_file = "networks/weights/maskrcnn_weights.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
