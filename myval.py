from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r"D:\deeplearning\lung_nodules_od\results\yolo11n4\weights\best.pt")

    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(metrics.box.map)  # mAP50-95
    print(metrics.box.map50)  # mAP50
    print(metrics.box.map75)  # mAP75
    print(metrics.box.maps)  # list of mAP50-95 for each category