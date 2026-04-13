"""
app.py - Streamlit Web Application for Plant Disease Detection

Usage:
    streamlit run app.py
"""

import os
import sys
import numpy as np

# Suppress TensorFlow info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# Import utility functions
from utils import (
    IMG_SIZE,
    MODEL_PATH,
    DATASET_DIR,
    load_class_indices,
    get_display_name,
    get_disease_info,
    preprocess_uploaded_image,
)
from predict import predict_from_array


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown(
    """
<style>

.main-title {
    text-align: center;
    color: #4CAF50;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #bdbdbd;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Prediction result card */
.result-box {
    background: linear-gradient(135deg, #1f3d2b, #2e7d32);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #66bb6a;
    margin: 1rem 0;
    color: white;
}

/* Disease detected card */
.disease-box {
    background: linear-gradient(135deg, #40231b, #8e2a0f);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #ff7043;
    margin: 1rem 0;
    color: white;
}

/* Healthy card */
.healthy-box {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #81c784;
    margin: 1rem 0;
    color: white;
}

/* Disease info cards (FIXED brightness) */
.info-box {
    background: #1f2933;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #42a5f5;
    margin: 0.5rem 0;
    color: #e0e0e0;
}

/* Sidebar text improvement */
[data-testid="stSidebar"] {
    background-color: #0f1720;
}

/* Footer */
.footer {
    text-align: center;
    color: #9e9e9e;
    margin-top: 3rem;
    font-size: 0.85rem;
}

/* Fix Streamlit success box brightness */
.stSuccess {
    background-color: #1b5e20 !important;
    color: white !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING (CACHED)
# ============================================================


@st.cache_resource
def load_model_cached():
    """
    Load the trained model with Streamlit caching.
    Loaded only once and reused across sessions.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None

        model = load_model(MODEL_PATH)
        class_indices = load_class_indices()
        return model, class_indices
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# ============================================================
# SIDEBAR
# ============================================================


def render_sidebar():
    """Render the sidebar with project information."""
    with st.sidebar:
        st.title("🌿 About")

        st.markdown(
            """
        This AI-powered application uses **Transfer Learning** with
        **MobileNet** to detect diseases in plant leaves.

        ### How It Works
        1. Upload a plant leaf image
        2. The AI analyzes the image
        3. Get disease prediction & remedy

        ### Supported Plants
        - 🫑 Pepper Bell
        - 🥔 Potato
        - 🍅 Tomato

        ### Detectable Diseases
        - Bacterial Spot
        - Early Blight
        - Late Blight
        - Leaf Mold
        - Healthy (no disease)
        """
        )

        st.markdown("---")

        st.markdown(
            """
        ### Model Info
        - **Architecture:** MobileNet
        - **Input Size:** 224 x 224
        - **Training:** Transfer Learning
        - **Framework:** TensorFlow/Keras
        """
        )

        st.markdown("---")

        st.markdown(
            """
        ### Disclaimer
        This is a university project for educational purposes.
        Always consult agricultural experts for professional advice.
        """
        )


# ============================================================
# MAIN APPLICATION
# ============================================================


def main():
    """Main application function."""

    # Render sidebar
    render_sidebar()

    # ---------------------------------------------------------
    # Header Section
    # ---------------------------------------------------------
    st.markdown(
        '<h1 class="main-title">🌿 Plant Disease Detection</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Upload a plant leaf image to detect diseases '
        "using AI-powered analysis</p>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Load Model
    # ---------------------------------------------------------
    model, class_indices = load_model_cached()

    if model is None or class_indices is None:
        st.error("**Model not found!** Please train the model first.")

        st.markdown(
            """
        ### How to Train the Model

        Open your terminal and run:

        ```bash
        # Option 1: Quick test with synthetic data
        python train_model.py --create-sample

        # Option 2: Train with real dataset (download PlantVillage first)
        python train_model.py
        ```

        After training, restart this app with:
        ```bash
        streamlit run app.py
        ```
        """
        )
        return

    # Show model loaded status
    num_classes = len(class_indices)
    st.success(f"Model loaded successfully! ({num_classes} disease classes)")

    # ---------------------------------------------------------
    # Image Upload Section
    # ---------------------------------------------------------
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Plant Leaf Image")
        st.markdown("Supported formats: **JPG, JPEG, PNG**")

        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf for disease detection",
        )

        # Add sample image option
        use_sample = st.checkbox("Use a sample image from dataset (for testing)")

    # Variables for the image to predict
    image_to_predict = None
    display_img = None
    sample_class_name = None

    # Handle sample image
    if use_sample and not uploaded_file:
        if os.path.exists(DATASET_DIR):
            for class_dir in sorted(os.listdir(DATASET_DIR)):
                class_path = os.path.join(DATASET_DIR, class_dir)
                if os.path.isdir(class_path):
                    images = [
                        f
                        for f in os.listdir(class_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                    if images:
                        sample_path = os.path.join(class_path, images[0])
                        with col1:
                            st.info(f"Using sample: `{class_dir}/{images[0]}`")

                        # Load and process
                        image_cv = cv2.imread(sample_path)
                        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                        image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
                        display_img = image_resized.copy()
                        image_normalized = image_resized.astype(np.float32) / 255.0
                        image_to_predict = np.expand_dims(image_normalized, axis=0)
                        sample_class_name = class_dir

                        with col2:
                            st.markdown("### 🖼️ Sample Image")
                            sample_pil = Image.open(sample_path)
                            st.image(
                                sample_pil,
                                caption=f"Sample: {class_dir}",
                                use_container_width=True,
                            )
                        break
        else:
            with col1:
                st.warning("No dataset directory found.")

    # Handle uploaded file
    if uploaded_file is not None:
        with col2:
            st.markdown("### 🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

            st.markdown(
                f"""
            **Image Details:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Mode: {image.mode}
            - File: {uploaded_file.name}
            """
            )

        # Reset file pointer and preprocess
        uploaded_file.seek(0)
        try:
            image_to_predict, display_img = preprocess_uploaded_image(uploaded_file)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return

    # ---------------------------------------------------------
    # Prediction Section
    # ---------------------------------------------------------
    if image_to_predict is not None:
        st.markdown("---")

        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            predict_button = st.button(
                "🔍 Analyze Leaf & Detect Disease",
                type="primary",
                use_container_width=True,
            )

        if predict_button:
            # Show progress
            with st.spinner("Analyzing the leaf image..."):
                result = predict_from_array(model, image_to_predict, class_indices)

            # ---------------------------------------------------------
            # Display Results
            # ---------------------------------------------------------
            st.markdown("## 📊 Prediction Results")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                is_healthy = "healthy" in result["class_name"].lower()

                if is_healthy:
                    st.markdown(
                        f"""
                    <div class="healthy-box">
                        <h3>✅ {result['display_name']}</h3>
                        <p><strong>Confidence:</strong> {result['confidence'] * 100:.2f}%</p>
                        <p>The plant appears to be <strong>healthy</strong>!
                        No disease detected.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="disease-box">
                        <h3>⚠️ {result['display_name']}</h3>
                        <p><strong>Confidence:</strong> {result['confidence'] * 100:.2f}%</p>
                        <p>A potential disease has been detected.
                        Please see the details below.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Confidence meter
                st.markdown("### 📈 Confidence Score")
                confidence_pct = result["confidence"] * 100

                if confidence_pct >= 80:
                    conf_text = "High Confidence"
                elif confidence_pct >= 50:
                    conf_text = "Medium Confidence"
                else:
                    conf_text = "Low Confidence"

                st.progress(result["confidence"])
                st.markdown(f"**{confidence_pct:.2f}%** — {conf_text}")

            with res_col2:
                st.markdown("### 📋 Disease Information")

                info = result["disease_info"]

                st.markdown(
                    f"""
                <div class="info-box">
                    <p><strong>📝 Description:</strong></p>
                    <p>{info['description']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                <div class="info-box">
                    <p><strong>💊 Recommended Remedy:</strong></p>
                    <p>{info['remedy']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # ---------------------------------------------------------
            # Confidence Bar Chart
            # ---------------------------------------------------------
            st.markdown("---")
            st.markdown("### 📊 Prediction Confidence for All Classes")

            top_predictions = result["all_predictions"][:10]
            names = [p["display_name"] for p in top_predictions]
            scores = [p["confidence"] * 100 for p in top_predictions]

            fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))

            colors = []
            for i, pred in enumerate(top_predictions):
                if i == 0:
                    colors.append("#2e7d32")
                elif "healthy" in pred["class_name"].lower():
                    colors.append("#66bb6a")
                else:
                    colors.append("#42a5f5")

            bars = ax.barh(range(len(names)), scores, color=colors, height=0.6)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel("Confidence (%)", fontsize=12)
            ax.set_title(
                "Disease Prediction Confidence", fontsize=14, fontweight="bold"
            )
            if scores:
                ax.set_xlim([0, max(scores) * 1.15])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)

            for bar, score in zip(bars, scores):
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Expandable detailed predictions
            with st.expander("Show Detailed Predictions"):
                st.markdown("**All class predictions sorted by confidence:**")
                for i, pred in enumerate(result["all_predictions"]):
                    confidence = pred["confidence"] * 100
                    if confidence > 50:
                        icon = "🟢"
                    elif confidence > 10:
                        icon = "🟡"
                    else:
                        icon = "⚪"
                    st.write(
                        f"{icon} **{i + 1}.** {pred['display_name']} — "
                        f"`{confidence:.4f}%`"
                    )

    elif uploaded_file is None and not use_sample:
        # Show instructions when no image is uploaded
        st.markdown("---")
        st.markdown(
            """
        <div class="info-box">
            <h4>👆 Upload an image to get started!</h4>
            <p>Upload a clear photo of a plant leaf using the upload button above.
            The AI will analyze the image and predict if the plant has any disease.</p>
            <p><strong>Tips for best results:</strong></p>
            <ul>
                <li>Use a clear, well-lit image of the leaf</li>
                <li>Capture the leaf against a plain background if possible</li>
                <li>Include the full leaf in the image</li>
                <li>Supported plants: Tomato, Potato, Pepper Bell</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # Footer
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<p class="footer">🌿 Plant Disease Detection | '
        "Built with TensorFlow & Streamlit | "
        "University AI Project</p>",
        unsafe_allow_html=True,
    )


# ============================================================
# RUN THE APP
# ============================================================

if __name__ == "__main__":
    main()