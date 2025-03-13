import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput

class YOLOv8Model(MLModel):
    async def load(self) -> bool:
        """Load the YOLOv8 model"""
        self.model = YOLO("/app/model/best.pt")
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """Perform inference using the YOLOv8 model and return base64 encoded output"""

        # ✅ Correct extraction of base64 image from the request
        input_data = payload.inputs[0].data[0]  # Extracts the actual base64 string

        # ✅ Decode base64 string to an image
        image_data = base64.b64decode(input_data.encode("utf-8"))
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)

        # Convert RGB to BGR (YOLO expects BGR)
        if len(image_np.shape) == 3 and image_np.shape[-1] == 3:
            image_np = image_np[:, :, ::-1]

        # Run inference
        results = self.model(image_np)

        # Encode output image with bounding boxes as base64
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        encoded_output_image = base64.b64encode(buffer).decode("utf-8")

        return InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[
                ResponseOutput(name="annotated_image", shape=[1], datatype="BYTES", data=[encoded_output_image])
            ]
        )
