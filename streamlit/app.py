import cv2
import streamlit as st
from PIL import Image
import numpy as np
import os

from detector import SealDetector

from classifier import SealClassifier
from renderer import OverlayRenderer
from report_generator import ReportGenerator

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Power Meter Seal Detection",
    page_icon="🔍",
    layout="wide"
)

st.title(
    "Power Meter Seal Geometry Validation"
)

st.markdown(
    "Upload power meter images to detect and validate seal geometry"
)

# =====================================================
# OUTPUT DIRECTORY
# =====================================================

OUTPUT_DIR = "predictions"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "Detection Settings"
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.25,
    0.05
)

max_angle = st.sidebar.slider(
    "Maximum Allowed Angle",
    0,
    45,
    25,
    1
)

min_bottom_distance = st.sidebar.slider(
    "Minimum Bottom Seal Distance",
    0,
    600,
    120,
    10
)

show_debug = st.sidebar.checkbox(
    "Show Debug Visuals",
    value=True
)

# =====================================================
# INITIALIZE CLASSES
# =====================================================

detector = SealDetector(
    "best.pt"
)

classifier = SealClassifier(
    max_angle=max_angle,
    min_bottom_distance=min_bottom_distance
)

renderer = OverlayRenderer()

reporter = ReportGenerator()

# =====================================================
# FILE UPLOADER
# =====================================================


uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =====================================================
# PROCESS IMAGES
# =====================================================

if uploaded_files is not None and len(uploaded_files) > 0:

    for file in os.listdir(
            OUTPUT_DIR
    ):

        try:
            os.remove(
                os.path.join(
                    OUTPUT_DIR,
                    file
                )
            )
        except:
            pass

    for uploaded_file in uploaded_files:
        st.markdown("---")

        st.subheader(
            f"📁 {uploaded_file.name}"
        )

        image = Image.open(
            uploaded_file
        )

        image_np = np.array(
            image
        )

        # =====================================
        # AUTO ROTATION
        # =====================================

        height, width = (
            image_np.shape[:2]
        )

        # =====================================
        # DETECTION
        # =====================================

        boxes = detector.predict(
            image_np,
            confidence
        )

        (
            filtered_boxes,
            xyxy,
            confs
        ) = detector.apply_nms(
            boxes
        )

        total_seals = len(filtered_boxes)

        seal_centers = (
            detector.extract_seal_centers(
                filtered_boxes,
                xyxy,
                confs
            )
        )

        # =====================================
        # CLASSIFICATION
        # =====================================

        result = (
            classifier.classify_meter(
                seal_centers,
                width,
                height
            )
        )

        # =====================================
        # DRAWING
        # =====================================

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
                result[
                    "anchor"
                ],
                result[
                    "matched_candidate"
                ]
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

        # =====================================
        # REPORT
        # =====================================

        reporter.add_record(
            uploaded_file.name,
            result,
            len(filtered_boxes)
        )

        # =====================================
        # SAVE IMAGE
        # =====================================

        output_path = os.path.join(
            OUTPUT_DIR,
            f"pred_{uploaded_file.name}"
        )

        Image.fromarray(
            annotated_image
        ).save(
            output_path
        )

        # =====================================
        # DISPLAY
        # =====================================

        st.image(
            annotated_image,
            width=700
        )

        st.write(
            f"Classification : {result['status']}"
        )

        st.write(
            f"Reason : {result['reason']}"
        )

        st.write(
            f"Main Seals : {result['main_seals_detected']}/3"
        )

        st.write(
            f"Extra Seals : {result['extra_seals']}"
        )

        st.write(
            f"Total Seals Detected : {len(filtered_boxes)}"
        )

    # ==========================================
    # EXCEL
    # ==========================================

    excel_path = (
        reporter.export_excel(
            OUTPUT_DIR
        )
    )

    zip_path = (
        reporter.create_zip(
            OUTPUT_DIR
        )
    )

    with open(
            zip_path,
            "rb"
    ) as f:

        st.download_button(
            "📥 Download Predictions",
            f,
            file_name="predictions.zip"
        )

    with open(
            excel_path,
            "rb"
    ) as f:

        st.download_button(
            "📊 Download Excel Report",
            f,
            file_name="seal_analysis.xlsx"
        )

st.markdown("---")

st.markdown(
    "YOLOv8 Power Meter Seal Geometry Validation System"
)