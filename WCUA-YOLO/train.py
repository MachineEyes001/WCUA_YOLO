from  ultralytics import  YOLO
# model=YOLO("D:/ultralytics-main/ultralytics-main/ultralytics/yolov8n.pt")
if __name__ == '__main__':
    model=YOLO("ultralytics/cfg/models/v8/yolov8n.yaml")
    results=model.train(data=r"ultralytics/cfg/dataset/train.yaml",save=True,imgsz=1280,patience = 100,device = 0)

