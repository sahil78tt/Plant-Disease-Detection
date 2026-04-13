"""
predict.py - Prediction Script for Plant Disease Detection

This script loads a trained model and predicts the disease class
for a given plant leaf image.

Usage:
    python predict.py path/to/leaf_image.jpg
    python predict.py  (auto-finds a sample image)
"""

import os
import sys
import numpy as np

# Suppress TensorFlow info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# Import utility functions
from utils import (
    IMG_SIZE,
    MODEL_PATH,
    DATASET_DIR,
    preprocess_image,
    load_class_indices,
    get_class_name,
    get_display_name,
    get_disease_info,
)


def load_trained_model():
    """
    Load the trained plant disease detection model.

    Returns:
        keras.Model: The loaded and compiled model
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Trained model not found at: {MODEL_PATH}\n"
            f"Please train the model first by running:\n"
            f"  python train_model.py\n"
            f"\nOr for a quick test with synthetic data:\n"
            f"  python train_model.py --create-sample"
        )

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully!")

    return model


def predict_disease(model, image_path, class_indices):
    """
    Predict the plant disease from a leaf image.

    Args:
        model: Trained Keras model
        image_path (str): Path to the leaf image
        class_indices (dict): Dictionary mapping class names to indices

    Returns:
        dict: Prediction results
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make prediction - outputs probabilities for each class
    predictions = model.predict(processed_image, verbose=0)

    # Get the predicted class index (highest probability)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    # Get the class name
    class_name = get_class_name(predicted_index, class_indices)
    display_name = get_display_name(class_name)
    disease_info = get_disease_info(class_name)

    # Create sorted list of all predictions
    index_to_class = {v: k for k, v in class_indices.items()}
    all_predictions = []
    for idx in range(len(predictions[0])):
        cls_name = index_to_class.get(idx, f"Unknown_{idx}")
        cls_display = get_display_name(cls_name)
        cls_confidence = float(predictions[0][idx])
        all_predictions.append(
            {
                "class_name": cls_name,
                "display_name": cls_display,
                "confidence": cls_confidence,
            }
        )

    # Sort by confidence (highest first)
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "class_name": class_name,
        "display_name": display_name,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "disease_info": disease_info,
    }


def predict_from_array(model, image_array, class_indices):
    """
    Predict disease from a preprocessed image array.
    Used by the Streamlit app.

    Args:
        model: Trained Keras model
        image_array: Preprocessed image array of shape (1, 224, 224, 3)
        class_indices (dict): Dictionary mapping class names to indices

    Returns:
        dict: Prediction results
    """
    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    # Get the predicted class
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    # Get class details
    class_name = get_class_name(predicted_index, class_indices)
    display_name = get_display_name(class_name)
    disease_info = get_disease_info(class_name)

    # All predictions sorted by confidence
    index_to_class = {v: k for k, v in class_indices.items()}
    all_predictions = []
    for idx in range(len(predictions[0])):
        cls_name = index_to_class.get(idx, f"Unknown_{idx}")
        cls_display = get_display_name(cls_name)
        cls_confidence = float(predictions[0][idx])
        all_predictions.append(
            {
                "class_name": cls_name,
                "display_name": cls_display,
                "confidence": cls_confidence,
            }
        )
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "class_name": class_name,
        "display_name": display_name,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "disease_info": disease_info,
    }


def display_prediction(image_path, result):
    """
    Display the prediction results in the terminal and save a visualization.

    Args:
        image_path (str): Path to the original image
        result (dict): Prediction results
    """
    print("\n" + "=" * 60)
    print("   PLANT DISEASE PREDICTION RESULTS")
    print("=" * 60)
    print(f"   Image:       {os.path.basename(image_path)}")
    print(f"   Prediction:  {result['display_name']}")
    print(f"   Confidence:  {result['confidence'] * 100:.2f}%")
    print("-" * 60)
    print(f"   Description: {result['disease_info']['description']}")
    print(f"   Remedy:      {result['disease_info']['remedy']}")
    print("-" * 60)

    # Show top 5 predictions
    print("\n   Top Predictions:")
    for i, pred in enumerate(result["all_predictions"][:5]):
        bar_length = int(pred["confidence"] * 30)
        bar = "#" * bar_length + "." * (30 - bar_length)
        print(
            f"   {i + 1}. {pred['display_name']:<35} "
            f"[{bar}] {pred['confidence'] * 100:.2f}%"
        )

    print("=" * 60)

    # Create and save visualization
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot the image
        axes[0].imshow(image_rgb)
        axes[0].set_title(f"Input: {os.path.basename(image_path)}", fontsize=12)
        axes[0].axis("off")

        # Plot prediction bar chart (top 5)
        top_5 = result["all_predictions"][:5]
        names = [p["display_name"] for p in top_5]
        scores = [p["confidence"] * 100 for p in top_5]
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(names))]

        bars = axes[1].barh(range(len(names)), scores, color=colors)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=10)
        axes[1].set_xlabel("Confidence (%)", fontsize=12)
        axes[1].set_title("Prediction Confidence", fontsize=12, fontweight="bold")
        axes[1].set_xlim([0, 105])
        axes[1].invert_yaxis()

        for bar, score in zip(bars, scores):
            axes[1].text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()

        output_dir = os.path.dirname(image_path) or "."
        output_path = os.path.join(
            output_dir,
            f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.png",
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n[INFO] Prediction visualization saved to: {output_path}")
    except Exception as e:
        print(f"\n[WARNING] Could not save visualization: {e}")


def find_sample_image():
    """Try to find a sample image from the dataset directory."""
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
                    return os.path.join(class_path, images[0])
    return None


def main():
    """Main function for command-line prediction."""

    # Determine image path
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        print("\nUsage: python predict.py <path_to_leaf_image>")
        print("\n[INFO] No image path provided. Looking for a sample image...")

        image_path = find_sample_image()

        if image_path:
            print(f"[INFO] Found sample image: {image_path}")
        else:
            print("[ERROR] No sample images found. Please provide an image path.")
            sys.exit(1)

    # Validate image path
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # Load model and class indices
    try:
        model = load_trained_model()
        class_indices = load_class_indices()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    # Make prediction
    print(f"\n[INFO] Predicting disease for: {image_path}")
    result = predict_disease(model, image_path, class_indices)

    # Display results
    display_prediction(image_path, result)


if __name__ == "__main__":
    main()