from ultralytics import YOLO

# 加载模型
model = YOLO('C:/Users/LX/Desktop/ultralytics-8.1.0/ultralytics-8.1.0/ultralytics/cfg/models/v8/yolov8n_mango.yaml')
results = model.train(data='C:/Users/LX/Desktop/ultralytics-8.1.0/ultralytics-8.1.0/ultralytics/data/Silkworm.yaml',
                      epochs=150, batch=4)  # 训练模型
