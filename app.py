import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import io
import numpy as np
from collections import Counter
import pandas as pd
import time

# Page settings
st.set_page_config(page_title="PCB Defect Detection", layout="wide")

# Custom CSS Styling
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

div[data-testid="metric-container"] {
    background-color: #1E2530;
    border: 1px solid #2E3B4E;
    padding: 15px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
# 🔍 PCB Defect Detection Dashboard
### AI-powered defect inspection using YOLOv8
Upload a PCB image to detect manufacturing defects in real-time.
""")

# Sidebar
st.sidebar.success("Model Loaded Successfully")
st.sidebar.write("Supported Defects:")
st.sidebar.write("• open")
st.sidebar.write("• short")
st.sidebar.write("• mousebite")
st.sidebar.write("• spur")
st.sidebar.write("• copper")
st.sidebar.write("• pin-hole")

# Load model
model = YOLO("best.pt")

# Upload image
uploaded_file = st.file_uploader(
    "Upload PCB Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Loading spinner + processing time
    with st.spinner("Analyzing PCB defects..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)

            start = time.time()
            results = model.predict(tmp.name, conf=0.25)
            end = time.time()

    processing_time = round(end - start, 3)

    result = results[0]
    predicted = result.plot()

    # Convert predicted image for download
    predicted_pil = Image.fromarray(np.uint8(predicted))
    buf = io.BytesIO()
    predicted_pil.save(buf, format="PNG")

    # Side-by-side image display
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original PCB", width=500)

    with col2:
        st.image(predicted, caption="Detected PCB", width=500)

    total_defects = len(result.boxes)

    st.write("## Detection Results")

    # Metric Cards (4 cards now)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Defects", total_defects)
    m2.metric("Model", "YOLOv8")
    m3.metric("mAP@50", "98.8%")
    m4.metric("Inference Time", f"{processing_time}s")

    # Severity Classification
    if total_defects == 0:
        st.success("PCB is Healthy ✅")
    elif total_defects <= 3:
        st.warning("Minor Defects Detected ⚠️")
    elif total_defects <= 6:
        st.warning("Moderate Damage 🟠")
    else:
        st.error("Critical PCB Damage 🚨")

    if total_defects > 0:
        # Count defect types
        class_names = result.names
        detected_classes = [class_names[int(box.cls)] for box in result.boxes]
        defect_counts = Counter(detected_classes)

        st.write("### Defect Summary")
        for defect, count in defect_counts.items():
            st.write(f"• {defect}: {count}")

        # Bar Chart
        chart_data = pd.DataFrame({
            "Defect": list(defect_counts.keys()),
            "Count": list(defect_counts.values())
        })

        st.write("### Defect Distribution")
        st.bar_chart(chart_data.set_index("Defect"))

    # Download button
    st.download_button(
        label="📥 Download Detection Result",
        data=buf.getvalue(),
        file_name="pcb_detection_result.png",
        mime="image/png"
    )