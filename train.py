from ultralytics import YOLO

# Load a pretrained YOLOv8 model (Nano version)
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for small, 'yolov8m.pt' for medium

# Train the model
model.train(
    data="data.yaml",  # Path to data.yaml
    epochs=50,        # Number of training epochs
    imgsz=640,        # Image size
    batch=16,         # Batch size
    workers=4,        # Number of dataloader workers
    device='cuda'     # Use GPU if available, otherwise 'cpu'
)

# Validate the trained model
metrics = model.val()
print("Validation Metrics:", metrics)

# Export model to PyTorch format (optional)
model.export()