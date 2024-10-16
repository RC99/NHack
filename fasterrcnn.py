import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Check if a GPU is available and use it if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model and move it to the appropriate device
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

video_path = 'trimmed.mov'
cap = cv2.VideoCapture(video_path)

def transform_image(image):
    image = F.to_tensor(image).to(device)  # Move tensor to the same device as the model
    return image

car_class_index = 3  
frame_counter = 0  # Frame counter
skip_frames = 2    # Process every nth frame

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    iou = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    return iou

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every nth frame
    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))  # Adjust size as needed
    image_tensor = transform_image(frame)

    with torch.no_grad():
        predictions = model([image_tensor])

    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    car_boxes = []
    for box, label, score in zip(boxes, labels, scores):
        if label == car_class_index and score > 0.85:  
            x1, y1, x2, y2 = [int(coord) for coord in box]
            car_boxes.append((x1, y1, x2, y2))

    for i, box1 in enumerate(car_boxes):
        for j, box2 in enumerate(car_boxes):
            if i != j:  
                iou = compute_iou(box1, box2)
                if iou > 0.30:  
                    print(f"Overlap detected between boxes: {box1} and {box2} with IoU: {iou:.2f}")

    for box, label, score in zip(boxes, labels, scores):
        if label == car_class_index and score > 0.85:  
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Car {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Car Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
