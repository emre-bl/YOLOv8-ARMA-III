import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict

class ObjectDetector:
    def __init__(self, model_path, yaml_path):
        self.model = YOLO(model_path)
        self.yaml_data = self.model.cfg
        self.colors = self._generate_colors(len(self.model.names))

    def _generate_colors(self, num_classes):
        np.random.seed(42)
        return np.random.randint(0, 255, size=(num_classes, 3)).tolist()

    def process_image(self, image_path):
        results = self.model(image_path, conf=0.3, iou=0.5, verbose=False)
        image = cv2.imread(image_path)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                color = self.colors[cls]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image, results

    def evaluate_model(self, folder_path, output_path):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        iou_threshold = 0.5
        detection_results = []

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, file)
                image = cv2.imread(image_path)
                img_height, img_width = image.shape[:2]
                
                # Load ground truth
                label_path = os.path.join(folder_path.replace('images', 'labels'), 
                                        os.path.splitext(file)[0] + '.txt')
                gt_boxes = []
                gt_classes = []
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) < 5: continue
                            cls = int(parts[0])
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            x1 = max(0, int(x_center - width/2))
                            y1 = max(0, int(y_center - height/2))
                            x2 = min(img_width, int(x_center + width/2))
                            y2 = min(img_height, int(y_center + height/2))
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_classes.append(cls)
                
                # Run detection
                output_image, results = self.process_image(image_path)
                cv2.imwrite(os.path.join(output_path, file), output_image)
                
                # Process predictions
                pred_boxes = []
                pred_classes = []
                pred_confs = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        pred_boxes.append([x1, y1, x2, y2])
                        pred_classes.append(cls)
                        pred_confs.append(conf)
                
                # Match predictions with ground truths
                matched_gt = set()
                sorted_indices = sorted(range(len(pred_confs)), key=lambda i: -pred_confs[i])

                for p_idx in sorted_indices:
                    pred_box = pred_boxes[p_idx]
                    pred_cls = pred_classes[p_idx]
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                        if gt_idx in matched_gt:
                            continue
                        iou = self.compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_threshold:
                        matched_gt.add(best_gt_idx)
                        if pred_cls == gt_classes[best_gt_idx]:
                            tp[pred_cls] += 1
                        else:
                            fp[pred_cls] += 1
                            fn[gt_classes[best_gt_idx]] += 1
                    else:
                        fp[pred_cls] += 1

                # Handle unmatched ground truths
                for gt_idx in range(len(gt_boxes)):
                    if gt_idx not in matched_gt:
                        fn[gt_classes[gt_idx]] += 1

                detection_results.append({'image': file, 'detections': len(pred_boxes)})

        performance_report = {
            'detection_summary': detection_results,
            'total_images': len(os.listdir(folder_path)),
            'total_detections': sum(res['detections'] for res in detection_results),
            'per_class_metrics': self._calculate_per_class_metrics(tp, fp, fn)
        }
        return performance_report

    @staticmethod
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union != 0 else 0

    def _calculate_per_class_metrics(self, tp, fp, fn):
        metrics = {}
        for cls in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
            tp_cls = tp.get(cls, 0)
            fp_cls = fp.get(cls, 0)
            fn_cls = fn.get(cls, 0)

            precision = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0
            recall = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[self.model.names[cls]] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'tp': tp_cls,
                'fp': fp_cls,
                'fn': fn_cls
            }
        return metrics

    def plot_detection_summary(self, performance_report):
        plt.figure(figsize=(10, 6))
        plt.bar([det['image'] for det in performance_report['detection_summary']], 
                [det['detections'] for det in performance_report['detection_summary']])
        plt.title('Detections per Image')
        plt.xlabel('Image')
        plt.ylabel('Number of Detections')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('detection_summary.png')
        plt.close()

def main():
    model_path = "YOLOv8_Model.pt"
    yaml_path = "data/data.yaml"
    folder_path = "data/val/images"
    output_path = "data/val/output"

    detector = ObjectDetector(model_path, yaml_path)
    performance_report = detector.evaluate_model(folder_path, output_path)
    
    detector.plot_detection_summary(performance_report)

    print("Performance Report:")
    print(f"Total Images Processed: {performance_report['total_images']}")
    print(f"Total Detections: {performance_report['total_detections']}")
    print("\nPer-Class Metrics:")
    for cls, metrics in performance_report['per_class_metrics'].items():
        print(f"\nClass: {cls}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1-Score:  {metrics['f1']}")
        print(f"  TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")

if __name__ == "__main__":
    main()