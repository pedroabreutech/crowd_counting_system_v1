import streamlit as st
import numpy as np  # Import NumPy first to ensure it's available
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import gdown
import cv2
from ultralytics import YOLO

# Verify NumPy is available
if not hasattr(np, 'array'):
    raise RuntimeError("NumPy is not properly installed or available")

from csrnet_model import CSRNet

# ---------- Constants ----------
CSRNET_MODEL_PATH = "csrnet_model.pth"
CSRNET_MODEL_URL = "https://drive.google.com/uc?id=1-698yvi-ZwsPrnRlm6EKE2TXZC_f7QZ7"

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# ---------- CSS ----------
st.set_page_config(page_title="Crowd Counting System", layout="wide", initial_sidebar_state="collapsed")

# ---------- MODERN CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    background-attachment: fixed;
}

.main .block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #0066ff 0%, #00d4ff 50%, #0066ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
    animation: gradient-shift 3s ease infinite;
    background-size: 200% auto;
}

@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

h2, h3, h4 {
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

.description-box {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 102, 255, 0.2);
    border-left: 4px solid;
    border-image: linear-gradient(135deg, #0066ff, #00d4ff) 1;
    padding: 24px;
    margin-bottom: 24px;
    border-radius: 16px;
    color: #e8eaf6;
    box-shadow: 0 8px 32px rgba(0, 102, 255, 0.1);
    transition: all 0.3s ease;
}

.description-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 102, 255, 0.2);
    border-color: rgba(0, 102, 255, 0.4);
}

.description-box h4 {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #0066ff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-box {
    background: linear-gradient(135deg, rgba(0, 102, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
    backdrop-filter: blur(10px);
    padding: 32px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 102, 255, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(0, 102, 255, 0.3);
    text-align: center;
    margin-top: 24px;
    transition: all 0.3s ease;
}

.result-box:hover {
    transform: scale(1.02);
    box-shadow: 0 12px 48px rgba(0, 102, 255, 0.3);
}

.result-box h3 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0066ff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.accuracy-box {
    background: rgba(26, 46, 58, 0.6);
    backdrop-filter: blur(10px);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(0, 102, 255, 0.3);
    margin-top: 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 102, 255, 0.15);
}

.accuracy-box h4 {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #0066ff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.accuracy-metric {
    display: inline-block;
    margin: 8px 12px;
    padding: 12px 20px;
    background: rgba(15, 26, 32, 0.6);
    backdrop-filter: blur(5px);
    border-radius: 12px;
    border: 1px solid rgba(0, 102, 255, 0.3);
    transition: all 0.3s ease;
    font-weight: 500;
}

.accuracy-metric:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0, 102, 255, 0.3);
    border-color: rgba(0, 102, 255, 0.6);
}

.accuracy-metric strong {
    color: #00d4ff;
    font-weight: 600;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important;
    background: rgba(28, 30, 36, 0.5) !important;
    backdrop-filter: blur(10px);
    border-radius: 12px 12px 0 0;
    padding: 14px 24px !important;
    color: #8ba3d1 !important;
    border: 1px solid rgba(0, 102, 255, 0.2);
    transition: all 0.3s ease;
    font-size: 1rem;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0, 102, 255, 0.1) !important;
    color: #00d4ff !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0066ff, #00a8ff) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(0, 102, 255, 0.4);
    border-color: rgba(0, 102, 255, 0.5);
}

.stFileUploader {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border: 2px dashed rgba(0, 102, 255, 0.3);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: rgba(0, 102, 255, 0.6);
    background: rgba(0, 102, 255, 0.05);
}

.stButton>button {
    background: linear-gradient(135deg, #0066ff, #00a8ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(0, 102, 255, 0.3) !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0, 102, 255, 0.4) !important;
}

.stCheckbox label {
    color: #e8eaf6 !important;
    font-weight: 500;
}

.stNumberInput>div>div>input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(0, 102, 255, 0.3) !important;
    border-radius: 10px !important;
    color: #e8eaf6 !important;
}

.stNumberInput>div>div>input:focus {
    border-color: #0066ff !important;
    box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.2) !important;
}

.stMarkdown p {
    color: #b8c5d6;
    line-height: 1.7;
}

.stSpinner>div {
    border-color: #0066ff transparent transparent transparent !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 20, 31, 0.5);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #0066ff, #00d4ff);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #00a8ff, #00d4ff);
}

/* Image containers */
.stImage {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* Success/Info/Warning/Error messages */
.stSuccess {
    background: rgba(0, 212, 255, 0.1) !important;
    border-left: 4px solid #00d4ff !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

.stInfo {
    background: rgba(0, 102, 255, 0.1) !important;
    border-left: 4px solid #0066ff !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

.stWarning {
    background: rgba(255, 193, 7, 0.1) !important;
    border-left: 4px solid #ffc107 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

.stError {
    background: rgba(244, 67, 54, 0.1) !important;
    border-left: 4px solid #f44336 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Image Transform ----------
import torchvision.transforms as transforms
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
])

# ---------- Downloader ----------
def download_model(model_name):
    if model_name == "CSRNet" and not os.path.exists(CSRNET_MODEL_PATH):
        with st.spinner("🔽 Downloading CSRNet model..."):
            gdown.download(CSRNET_MODEL_URL, CSRNET_MODEL_PATH, quiet=False)




@st.cache_resource
def load_csrnet_model():
    download_model("CSRNet")
    model = CSRNet()
    checkpoint = torch.load(CSRNET_MODEL_PATH, map_location=torch.device("cpu"))

    # ✅ Extract only the state_dict from the checkpoint
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8l.pt')

# ---------- Predict & Visualize ----------
def predict_and_visualize(image, model):
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transform
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)

    count = float(output.sum().item())
    density_map = output.squeeze().cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(density_map, cmap='jet')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return count, fig, buf

def predict_with_yolo(image):
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    model = load_yolo_model()
    results = model(image_bgr)
    annotated = results[0].plot()
    count = len(results[0].boxes)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return count, annotated_rgb

# ---------- Accuracy Calculation ----------
def calculate_accuracy(predicted, actual):
    """
    Calculate accuracy metrics between predicted and actual counts.
    
    Returns:
        dict with 'absolute_error', 'relative_error_percent', 'accuracy_percent', 'mae'
    """
    if actual == 0:
        return {
            'absolute_error': abs(predicted - actual),
            'relative_error_percent': None,
            'accuracy_percent': None,
            'mae': abs(predicted - actual)
        }
    
    absolute_error = abs(predicted - actual)
    relative_error_percent = (absolute_error / actual) * 100
    accuracy_percent = max(0, 100 - relative_error_percent)
    
    return {
        'absolute_error': absolute_error,
        'relative_error_percent': relative_error_percent,
        'accuracy_percent': accuracy_percent,
        'mae': absolute_error
    }

# ---------- Main App ----------
def main():
    # ---------- TITLE + INTRO ----------
    st.title("Crowd Counting System")

    st.markdown("""
    <div style="text-align:center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #b8c5d6; margin-bottom: 0.5rem; font-weight: 400;">
            🤖 Intelligent AI-powered crowd counting system
        </p>
        <p style="font-size: 1rem; color: #8ba3d1; margin: 0;">
            Powered by <strong style="color: #00d4ff;">CSRNet</strong> and <strong style="color: #00d4ff;">YOLO</strong> deep learning models
        </p>
    </div>
    """, unsafe_allow_html=True)


    tab1, tab2 = st.tabs(["Outdoor Crowds", "Indoor Spaces"])

    with tab1:
        st.markdown("""
        <div class="description-box">
            <h4>🏙️ Outdoor Crowd Estimation</h4>
            <p style="margin: 0; line-height: 1.8;">
                Perfect for analyzing large-scale gatherings in open environments such as streets, festivals, parks, concerts, and public events.<br><br>
                <strong style="color: #00d4ff;">Powered by CSRNet</strong> — An advanced deep learning architecture that generates density heatmaps to accurately estimate crowd size, 
                particularly effective in high-density scenarios where individual detection becomes challenging.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "📤 Upload Image for Outdoor Analysis (CSRNet)", 
            type=["png", "jpg", "jpeg"],
            key="CSRNet",
            help="Upload an image of an outdoor crowd scene for analysis"
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Processing with CSRNet..."):
                model = load_csrnet_model()
                count, fig, buf = predict_and_visualize(image, model)
                st.markdown(f"""
                <div class='result-box'>
                    <h3>🧮 Estimated Count</h3>
                    <div style="font-size: 3rem; font-weight: 800; margin: 1rem 0; background: linear-gradient(135deg, #0066ff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                        {count:.2f}
                    </div>
                    <p style="color: #8ba3d1; margin: 0; font-size: 0.9rem;">people detected</p>
                </div>
                """, unsafe_allow_html=True)
                st.pyplot(fig)
                st.download_button(
                    "📥 Download Density Map", 
                    data=buf, 
                    file_name="density_map.png", 
                    mime="image/png",
                    help="Download the generated density heatmap as a PNG image"
                )
                
                # Accuracy calculation section
                st.markdown("---")
                st.markdown("""
                <div style="margin: 2rem 0 1rem 0;">
                    <h3 style="background: linear-gradient(135deg, #0066ff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">
                        📊 Accuracy Evaluation
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Checkbox to enable/disable accuracy evaluation
                enable_accuracy = st.checkbox(
                    "🔍 Enable accuracy evaluation",
                    value=False,
                    key="enable_accuracy_csrnet",
                    help="Check this box to enter the actual count and calculate prediction accuracy"
                )
                
                if enable_accuracy:
                    st.markdown("""
                    <p style="color: #b8c5d6; margin-bottom: 1rem;">
                        Enter the actual count to calculate prediction accuracy:
                    </p>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        actual_count = st.number_input(
                            "Actual Count",
                            min_value=0.0,
                            value=None,
                            step=1.0,
                            key="actual_csrnet",
                            help="Enter the real number of people in the image"
                        )
                    
                    if actual_count is not None and actual_count >= 0:
                        accuracy_metrics = calculate_accuracy(count, actual_count)
                        
                        accuracy_html = f"""
                        <div class='accuracy-box'>
                            <h4>📈 Accuracy Metrics</h4>
                            <div class='accuracy-metric'>
                                <strong>Absolute Error:</strong> {accuracy_metrics['absolute_error']:.2f}
                            </div>
                            <div class='accuracy-metric'>
                                <strong>Relative Error:</strong> {accuracy_metrics['relative_error_percent']:.2f}%
                            </div>
                            <div class='accuracy-metric'>
                                <strong>Accuracy:</strong> {accuracy_metrics['accuracy_percent']:.2f}%
                            </div>
                        </div>
                        """
                        st.markdown(accuracy_html, unsafe_allow_html=True)
                        
                        # Visual indicator
                        accuracy_value = accuracy_metrics['accuracy_percent']
                        if accuracy_value >= 90:
                            st.success(f"✅ Excellent accuracy: {accuracy_value:.2f}%")
                        elif accuracy_value >= 75:
                            st.info(f"ℹ️ Good accuracy: {accuracy_value:.2f}%")
                        elif accuracy_value >= 50:
                            st.warning(f"⚠️ Moderate accuracy: {accuracy_value:.2f}%")
                        else:
                            st.error(f"❌ Low accuracy: {accuracy_value:.2f}%")

    with tab2:
        st.markdown("""
        <div class="description-box">
            <h4>🏢 Indoor Crowd Detection</h4>
            <p style="margin: 0; line-height: 1.8;">
                Optimized for controlled indoor environments including classrooms, conference rooms, hallways, lobbies, and office spaces.<br><br>
                <strong style="color: #00d4ff;">Powered by YOLO</strong> — A state-of-the-art real-time object detection model that identifies and tracks individuals 
                with precise bounding boxes, providing accurate person-by-person counting in sparse crowd scenarios.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "📤 Upload Image for Indoor Detection (YOLO)", 
            type=["png", "jpg", "jpeg"],
            key="YOLO",
            help="Upload an image of an indoor space for person detection"
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Processing with YOLOv8..."):
                count, annotated_image = predict_with_yolo(image)
                st.image(annotated_image, caption="YOLO Prediction", use_container_width=True)
                st.markdown(f"""
                <div class='result-box'>
                    <h3>🧮 Detected Count</h3>
                    <div style="font-size: 3rem; font-weight: 800; margin: 1rem 0; background: linear-gradient(135deg, #0066ff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                        {count}
                    </div>
                    <p style="color: #8ba3d1; margin: 0; font-size: 0.9rem;">people detected</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Accuracy calculation section
                st.markdown("---")
                st.markdown("""
                <div style="margin: 2rem 0 1rem 0;">
                    <h3 style="background: linear-gradient(135deg, #0066ff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">
                        📊 Accuracy Evaluation
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Checkbox to enable/disable accuracy evaluation
                enable_accuracy = st.checkbox(
                    "🔍 Enable accuracy evaluation",
                    value=False,
                    key="enable_accuracy_yolo",
                    help="Check this box to enter the actual count and calculate prediction accuracy"
                )
                
                if enable_accuracy:
                    st.markdown("""
                    <p style="color: #b8c5d6; margin-bottom: 1rem;">
                        Enter the actual count to calculate prediction accuracy:
                    </p>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        actual_count = st.number_input(
                            "Actual Count",
                            min_value=0.0,
                            value=None,
                            step=1.0,
                            key="actual_yolo",
                            help="Enter the real number of people in the image"
                        )
                    
                    if actual_count is not None and actual_count >= 0:
                        accuracy_metrics = calculate_accuracy(float(count), actual_count)
                        
                        accuracy_html = f"""
                        <div class='accuracy-box'>
                            <h4>📈 Accuracy Metrics</h4>
                            <div class='accuracy-metric'>
                                <strong>Absolute Error:</strong> {accuracy_metrics['absolute_error']:.2f}
                            </div>
                            <div class='accuracy-metric'>
                                <strong>Relative Error:</strong> {accuracy_metrics['relative_error_percent']:.2f}%
                            </div>
                            <div class='accuracy-metric'>
                                <strong>Accuracy:</strong> {accuracy_metrics['accuracy_percent']:.2f}%
                            </div>
                        </div>
                        """
                        st.markdown(accuracy_html, unsafe_allow_html=True)
                        
                        # Visual indicator
                        accuracy_value = accuracy_metrics['accuracy_percent']
                        if accuracy_value >= 90:
                            st.success(f"✅ Excellent accuracy: {accuracy_value:.2f}%")
                        elif accuracy_value >= 75:
                            st.info(f"ℹ️ Good accuracy: {accuracy_value:.2f}%")
                        elif accuracy_value >= 50:
                            st.warning(f"⚠️ Moderate accuracy: {accuracy_value:.2f}%")
                        else:
                            st.error(f"❌ Low accuracy: {accuracy_value:.2f}%")


if __name__ == "__main__":
    main()
