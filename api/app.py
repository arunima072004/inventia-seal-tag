from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import cv2
import io
import base64

from detector import SealDetector
from classifier import SealClassifier
from renderer import OverlayRenderer

app = FastAPI(
    title="Power Meter Seal Detection API"
)

# Load once
detector = SealDetector("best.pt")
renderer = OverlayRenderer()


@app.get("/")
def home():
    return {
        "message": "Power Meter Seal Detection API Running"
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):
    confidence = 0.25
    max_angle = 25
    min_bottom_distance = 120
    show_debug = True


    contents = await file.read()


    print("=" * 50)
    print("Filename:", file.filename)
    print("Content-Type:", file.content_type)
    print("Bytes:", len(contents))
    print("=" * 50)

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")


    image_np = np.array(image)

    height, width = image_np.shape[:2]

    classifier = SealClassifier(
        max_angle=max_angle,
        min_bottom_distance=min_bottom_distance
    )

    # -------------------------
    # Detection
    # -------------------------

    boxes = detector.predict(
        image_np,
        confidence
    )

    filtered_boxes, xyxy, confs = (
        detector.apply_nms(boxes)
    )

    seal_centers = (
        detector.extract_seal_centers(
            filtered_boxes,
            xyxy,
            confs
        )
    )

    # -------------------------
    # Classification
    # -------------------------

    result = classifier.classify_meter(
        seal_centers,
        width,
        height
    )

    # -------------------------
    # Rendering
    # -------------------------

    annotated_image = (
        renderer.draw_detection_boxes(
            image_np.copy(),
            filtered_boxes,
            xyxy,
            confs,
            show_debug
        )
    )

    annotated_image = (
        renderer.draw_anchor(
            annotated_image,
            result["anchor"]
        )
    )

    annotated_image = (
        renderer.draw_cone(
            annotated_image,
            result["anchor"],
            result["direction"],
            max_angle
        )
    )

    annotated_image = (
        renderer.draw_match(
            annotated_image,
            result["anchor"],
            result["matched_candidate"]
        )
    )

    annotated_image = (
        renderer.draw_angle_distance(
            annotated_image,
            result["anchor"],
            result["angle"],
            result["distance"]
        )
    )

    annotated_image = (
        renderer.draw_dashboard(
            annotated_image,
            result
        )
    )

    # -------------------------
    # Convert image to base64
    # -------------------------

    success, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(
            annotated_image,
            cv2.COLOR_RGB2BGR
        )
    )

    image_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return {
        "status": result["status"],
        "reason": result["reason"],
        "suspicious": result["suspicious"],
        "main_seals_detected":
            result["main_seals_detected"],
        "extra_seals":
            result["extra_seals"],
        "total_seals":
            len(filtered_boxes),
        "angle":
            round(result["angle"], 2),
        "distance":
            round(result["distance"], 2)
    }