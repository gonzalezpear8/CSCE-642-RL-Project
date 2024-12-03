import argparse
from ultralytics import YOLO
import cv2
import torch

def predict_image(model_path, image_path):
    """
    Perform YOLO inference on a single image.
    """
    detections = []
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    
    # Perform prediction
    results = model(img)
    xywh = results[0].boxes.xywh.cpu()  # Bounding boxes
    cls = results[0].boxes.cls.cpu()   # Class labels

    # Combine bounding boxes and class labels
    combined = torch.cat((xywh, cls.unsqueeze(1)), dim=1)
    for i in range(combined.size(0)):
        grid_x = int((combined[i, 0] / 848) * 16)  # Normalize to 16x16 grid
        grid_y = int((combined[i, 1] / 480) * 16)
        classes = int(combined[i, -1])
        detections.append({"grid_x": grid_x, "grid_y": grid_y, "class": classes})
    
    print("Detections:", detections)
    return detections

def predict_video(model_path, video_path):
    """
    Perform YOLO inference on a video.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        xywh = results[0].boxes.xywh.cpu()  # Bounding boxes
        cls = results[0].boxes.cls.cpu()   # Class labels

        combined = torch.cat((xywh, cls.unsqueeze(1)), dim=1)
        detections = []
        for i in range(combined.size(0)):
            grid_x = int((combined[i, 0] / 848) * 16)  # Normalize to 16x16 grid
            grid_y = int((combined[i, 1] / 480) * 16)
            classes = int(combined[i, -1])
            detections.append({"grid_x": grid_x, "grid_y": grid_y, "class": classes})
        
        print("Detections:", detections)

def main():
    parser = argparse.ArgumentParser(description="YOLO Inference for Images or Videos")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--video", type=str, help="Path to the video file")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model file (e.g., best.pt)")

    args = parser.parse_args()

    if args.image:
        print(f"Running YOLO on image: {args.image}")
        predict_image(args.model, args.image)
    elif args.video:
        print(f"Running YOLO on video: {args.video}")
        predict_video(args.model, args.video)
    else:
        print("Error: You must provide either an --image or --video argument.")
        parser.print_help()

if __name__ == "__main__":
    main()
