# PCB Defect Detection using YOLOv8

This project is an AI-based PCB defect detection system developed using YOLOv8 and Streamlit for real-time defect identification.

## Project Overview

Printed Circuit Boards (PCBs) are essential in electronic devices. During manufacturing, defects may occur that affect circuit performance. This project automatically detects PCB defects using Deep Learning.

## Defects Detected

* Open
* Short
* Mousebite
* Spur
* Copper
* Pin-hole

## Technologies Used

* Python
* YOLOv8
* Streamlit
* DeepPCB Dataset
* Kaggle GPU

## Model Performance

* Precision: 97.1%
* Recall: 96.3%
* mAP@50: 98.8%

## Files

* `app.py` → Streamlit application
* `best.pt` → Trained YOLOv8 model weights
* `requirements.txt` → Required Python packages

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run app:

streamlit run app.py
