import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open the video file
video_path = r'backend\WhatsApp Video 2024-12-28 at 18.22.34_54abe45e.mp4'
cap = cv2.VideoCapture(video_path)

# Video writer to save the output video
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Initialize car counter
    car_count = 0
    
    # Draw bounding boxes and labels
    for result in results.xyxy[0].numpy():
        x1, y1, x2, y2, conf, cls = result
        if model.names[int(cls)] == 'car':  # Only count cars
            car_count += 1
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display car count at the bottom center of the frame
    text = f'Cars detected: {car_count}'
    font_scale = 2
    thickness = 3
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 20
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    
    # Save the output frame
    out.write(frame)

    # Display the frame with detections
    cv2.imshow('YOLOv5 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass