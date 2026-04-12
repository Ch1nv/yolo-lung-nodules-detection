from ultralytics import YOLO

models = ["yolo11l", "yolo11m", "yolo11s", "yolo11n"]

if __name__ == "__main__":

    for model_name in models:

        print(f"\n🚀 Training {model_name} ...\n")

        model = YOLO(fr"D:\deeplearning\lung_nodules_od\{model_name}.pt")

        model.train(
            data=r"D:\deeplearning\lung_nodules_od\yaml_datasets\lung_nodules_detection.yaml",

            epochs=150,
            imgsz=1280,

            lr0=0.003,
            lrf=0.01,
            optimizer="AdamW",
            weight_decay=0.0005,

            box=7.0,
            cls=0.5,
            dfl=1.8,

            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.3,
            scale=0.5,

            cache="ram",
            batch=-1,
            workers=2,

            project="results",
            name=model_name,
        )