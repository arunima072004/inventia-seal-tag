from ultralytics import YOLO
import numpy as np


class SealDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image, confidence=0.3, iou=0.45):

        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            verbose=False
        )

        return results[0].boxes

    def apply_nms(self, boxes, iou_threshold=0.80):

        if len(boxes) == 0:
            return [], None, None

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        indices = np.argsort(confs)[::-1]

        keep = []

        for i in indices:

            current_box = xyxy[i]

            keep_box = True

            x1, y1, x2, y2 = current_box

            current_area = (
                (x2 - x1) *
                (y2 - y1)
            )

            for kept_idx in keep:

                kept_box = xyxy[kept_idx]

                xx1 = max(x1, kept_box[0])
                yy1 = max(y1, kept_box[1])
                xx2 = min(x2, kept_box[2])
                yy2 = min(y2, kept_box[3])

                inter_w = max(0, xx2 - xx1)
                inter_h = max(0, yy2 - yy1)

                intersection = inter_w * inter_h

                kept_area = (
                    (kept_box[2] - kept_box[0]) *
                    (kept_box[3] - kept_box[1])
                )

                union = (
                    current_area +
                    kept_area -
                    intersection
                )

                iou = (
                    intersection / union
                    if union > 0 else 0
                )

                if iou > iou_threshold:
                    keep_box = False
                    break

            if keep_box:
                keep.append(i)

        return keep, xyxy, confs

    def extract_seal_centers(
        self,
        filtered_boxes,
        xyxy,
        confs
    ):

        seal_centers = []

        for idx in filtered_boxes:

            box = xyxy[idx]

            x1, y1, x2, y2 = map(int, box)

            conf = float(confs[idx])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            seal_centers.append({
                "center": (cx, cy),
                "box": (x1, y1, x2, y2),
                "conf": conf
            })

        return seal_centers