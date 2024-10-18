from ultralytics.nn import DetectionModel
# 模型网络结构配置文件路径
yaml_path = 'C:/Users/LX/Desktop/mango/ultralytics-8.0.227/ultralytics/cfg/models/v8/yolov8-FasterNet.yaml'
# 改进的模型结构路径
# yaml_path = 'ultralytics/cfg/models/v8/yolov8n-CBAM.yaml'
# 传入模型网络结构配置文件cfg, nc为模型检测类别数
DetectionModel(cfg=yaml_path, nc=80)
