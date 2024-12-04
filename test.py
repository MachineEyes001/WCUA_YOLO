from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs/train/exp/weights/best.pt')
    # results = model.predict(source='D:/XCData/train/images',save=True)
    results = model.predict(source=r'dataset/data.yaml',
                            line_width=12,
                            save=True)