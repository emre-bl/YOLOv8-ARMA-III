import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging

class ObjectDetectionTester:
    def __init__(self, model_path):
        """
        Initialize the object detection tester with a trained YOLO model
        
        Args:
            model_path (str): Path to the trained YOLO model
        """
        self.model = YOLO(model_path)
        self.colors = self._generate_colors()  # Generate unique colors for each vehicle class

    def _generate_colors(self):
        """
        Generate unique colors for different object classes
        
        Returns:
            dict: A dictionary mapping class indices to unique BGR colors
        """
        np.random.seed(42)
        return {i: tuple(np.random.randint(0, 255, 3).tolist()) 
                for i in range(len(self.model.names))}

    def _create_log_file(self, input_path, detections):
        """
        Create a log file with detection details
        
        Args:
            input_path (str): Path to input image/video
            detections (list): List of detected objects
        """
        base_name = os.path.splitext(os.path.basename(input_path))[0] 
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        log_name = f"{base_name}{'_Video' if is_video else '_Gorsel'}.log"
        
        with open(log_name, 'w') as log_file:
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                log_entry = f"{det['class']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {conf:.2f}"
                if is_video:
                    log_entry += f" {det['frame']}"
                log_file.write(log_entry + "\n")

        print(f"Log file saved to {log_name}")

    def detect_and_visualize(self, input_path):
        """
        Detect objects in an image or video and visualize results
        
        Args:
            input_path (str): Path to input image or video file
        """
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self._process_image(input_path)
        elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self._process_video(input_path)
        else:
            raise ValueError("Unsupported file type")

    def _process_image(self, image_path):
        """Process a single image for object detection"""
        results = self.model(image_path, conf=0.4, iou=0.5, verbose=False)

        # Get inference time from results
        inference_time = results[0].speed['inference']  # Inference time in ms
        print(f"\nImage processed in {inference_time:.1f}ms")

        image = cv2.imread(image_path)
        log_detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                center_x = (x1 + x2) / 2 / image.shape[1]
                center_y = (y1 + y2) / 2 / image.shape[0]
                width = (x2 - x1) / image.shape[1]
                height = (y2 - y1) / image.shape[0]

                color = self.colors[cls]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

                log_detections.append({
                    'class': cls,
                    'bbox': [center_x, center_y, width, height],
                    'confidence': conf
                })

        output_path = f"{os.path.splitext(image_path)[0]}_detected.png"
        cv2.imwrite(output_path, image)
        self._create_log_file(image_path, log_detections)
        print(f"Results saved to {output_path}")

    def _process_video(self, video_path):
        """Process a video for object detection"""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_path = f"{os.path.splitext(video_path)[0]}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_log_detections = []

        total_inference = 0.0
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            

            results = self.model(frame, conf=0.30, iou=0.5, verbose=False)

            # Accumulate inference time
            frame_inference = results[0].speed['inference']
            total_inference += frame_inference

            frame_number += 1
            frame_detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    center_x = (x1 + x2) / 2 / frame.shape[1]
                    center_y = (y1 + y2) / 2 / frame.shape[0]
                    width_dim = (x2 - x1) / frame.shape[1]
                    height_dim = (y2 - y1) / frame.shape[0]

                    color = self.colors[cls]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

                    frame_detections.append({
                        'class': cls,
                        'bbox': [center_x, center_y, width_dim, height_dim],
                        'confidence': conf
                    })

            out.write(frame)
            all_log_detections.extend([{**det, 'frame': frame_number} 
                                      for det in frame_detections])
            
        # Calculate and display average inference time
        if frame_number > 0:
            avg_inference = total_inference / frame_number
            print(f"\nVideo processing complete")
            print(f"Frames processed: {frame_number}")
            print(f"Average inference time per frame: {avg_inference:.1f}ms")

        cap.release()
        out.release()
        self._create_log_file(video_path, all_log_detections)
        print(f"Results saved to {output_path}")

def main():
    model_path = "YOLOv8_Model.pt"
    detector = ObjectDetectionTester(model_path)

    while True:
        print("\nObject Detection CLI")
        print("1. Process Image")
        print("2. Process Video")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '3':
            print("Exiting program...")
            break

        if choice not in ('1', '2'):
            print("Invalid choice. Please try again.")
            continue
        
        file_path = input("Enter file path: ").strip()
        if not os.path.exists(file_path):
            print("Error: File not found")
            continue

        try:
            if choice == '1':
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print("Error: Invalid image format")
                    continue
            else:
                if not file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    print("Error: Invalid video format")
                    continue
            
            print("Processing file...")
            detector.detect_and_visualize(file_path)
        except Exception as e:
            print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()