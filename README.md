# Inventia Seal Tag

An AI-powered power meter seal validation system that detects security seals from meter images and classifies the meter as **VALID** or **TAMPER SUSPECTED** using a custom-trained YOLOv8 model.

## Overview

This project automates the inspection of electricity meter seals by combining object detection with geometric validation rules. It provides both:

- A Streamlit web application for interactive analysis
- A FastAPI backend for API integration

## Features

- Detects meter seals using a custom YOLOv8 model
- Identifies the three primary seals (top, bottom-left, bottom-right)
- Detects additional or unexpected seals
- Calculates seal angle and bottom seal distance
- Classifies meters as **VALID** or **TAMPER SUSPECTED**
- Adjustable confidence and validation thresholds
- Bulk image processing
- Excel report generation
- REST API support via FastAPI

## Project Structure

```
inventia-seal-tag/
│
├── api/                 # FastAPI backend
│
├── streamlit/           # Streamlit application
│   ├── app.py
│   ├── detector.py
│   ├── classifier.py
│   ├── renderer.py
│   ├── report_generator.py
│   └── best.pt
│
└── README.md
```

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- FastAPI
- Pandas
- NumPy
- OpenPyXL

## Installation

Clone the repository:

```bash
git clone https://github.com/arunima072004/inventia-seal-tag.git
cd inventia-seal-tag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

```bash
cd streamlit
streamlit run app.py
```

## Running the FastAPI Server

```bash
cd api
uvicorn main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

## Workflow

1. Upload a meter image.
2. YOLOv8 detects all seals.
3. The system identifies the three primary seals.
4. Geometric validation is performed.
5. The meter is classified as VALID or TAMPER SUSPECTED.
6. Results are displayed and can be exported as an Excel report.

## Future Improvements

- Automatic meter orientation correction
- OCR integration
- Improved tamper detection logic
- Deployment using Docker
- Cloud-based API service

## Author

**Arunima Sakharkar**

GitHub: https://github.com/arunima072004
