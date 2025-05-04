from ultralytics import YOLO
from config import YOLO_MODEL_PATH, SKILL_THRESHOLD

class YoloModel:
    def __init__(self, model_path: str = YOLO_MODEL_PATH, conf_thresh: float = SKILL_THRESHOLD):
        # Load YOLO model once
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh / 100.0

    def predict(self, image):
        """
        Run YOLO inference on a PIL.Image or path.
        Returns a list of dicts: class_id, name, confidence, bbox
        """
        results = self.model(image)
        detections = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf)  # 0–1
                cls = int(box.cls)
                name = self.model.names.get(cls, str(cls))
                # unpack xyxy
                x0, y0, x1, y1 = map(float, box.xyxy[0])

                detections.append({
                    'class_id': cls,
                    'name': name,
                    'confidence': conf,
                    'bbox': [x0, y0, x1, y1]
                })

        return detections