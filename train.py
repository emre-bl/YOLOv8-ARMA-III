from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
import torch
import yaml

SMALL_OBJ_THRESHOLD = 48
SMALL_OBJ_SCALE = 4.0

class SmallObjectLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        # Get parameters from model args
        self.small_obj_threshold = model.args.get('small_obj_threshold', SMALL_OBJ_THRESHOLD)
        self.small_obj_scale = model.args.get('small_obj_scale', SMALL_OBJ_SCALE)

    def __call__(self, preds, batch):
        loss, loss_items = super().__call__(preds, batch)
        
        targets = batch["bbox"]
        if targets.shape[0] == 0:
            return loss, loss_items

        imgsz = self.model.args.get('imgsz', 640)
        widths = targets[:, 4] * imgsz
        heights = targets[:, 5] * imgsz
        areas = widths * heights

        small_objs = areas < (self.small_obj_threshold ** 2)
        if not small_objs.any():
            return loss, loss_items

        scaled_box_loss = loss_items[1] * self.small_obj_scale
        total_loss = loss + (scaled_box_loss - loss_items[1])

        return total_loss, (total_loss, scaled_box_loss, loss_items[2], loss_items[3])

class CustomModel(DetectionModel):
    def __init__(self, cfg, ch=3, nc=None, args=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.args = args or {}
        self.loss = SmallObjectLoss(self)

if __name__ == "__main__":
    data_path = "data/data.yaml"
    
    with open(data_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg['nc']

    # Initialize model with custom parameters
    model = YOLO("yolov8m.pt")
    model.model = CustomModel(
        cfg=model.model.yaml,
        ch=3,
        nc=nc,
        args={
            'small_obj_threshold': SMALL_OBJ_THRESHOLD,  # Custom parameters here
            'small_obj_scale': SMALL_OBJ_SCALE
        }
    )

    # Standard training parameters only
    results = model.train(
        data=data_path,
        epochs=600,
        pretrained=True,
        imgsz=640,
        batch=24,
        val=True,
        verbose=True,
        cos_lr=True,
        optimizer="AdamW", # Better for small datasets
        plots=True,
        patience=150,
        lr0=0.001,
        lrf=0.01,


        # Regularization Dont use weight decay and dropout together
        weight_decay=0.005, 
        dropout=0.0, 
        label_smoothing=0.1,

        # Augmentation
        degrees=40,
        mixup=0.25,
        copy_paste=0.2,
        scale=0.25,
        flipud=0.5,
        fliplr=0.5,
        shear=0.1,
        translate=0.1,
        mosaic=0.5,
        close_mosaic=10,
        # Loss coefficients (corrected names)
        cls = 3,
        box = 7,
        dfl = 1.5,
    )