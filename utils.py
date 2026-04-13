"""
utils.py - Utility functions for Plant Disease Detection Project

This module contains helper functions used across the project for:
- Image preprocessing
- Dataset downloading and preparation
- Label encoding/decoding
- Common constants and configurations
"""

import os
import json
import numpy as np
import cv2

# ============================================================
# PROJECT CONSTANTS
# ============================================================

# Image dimensions expected by MobileNet
IMG_SIZE = 224

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.h5")

# We will use a curated subset of PlantVillage dataset classes
# to keep training lightweight and CPU-friendly
SELECTED_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_healthy",
]

# Human-readable names for display in the app
CLASS_DISPLAY_NAMES = {
    "Pepper__bell___Bacterial_spot": "Pepper Bell - Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Bell - Healthy",
    "Potato___Early_blight": "Potato - Early Blight",
    "Potato___Late_blight": "Potato - Late Blight",
    "Potato___healthy": "Potato - Healthy",
    "Tomato_Bacterial_spot": "Tomato - Bacterial Spot",
    "Tomato_Early_blight": "Tomato - Early Blight",
    "Tomato_Late_blight": "Tomato - Late Blight",
    "Tomato_Leaf_Mold": "Tomato - Leaf Mold",
    "Tomato_healthy": "Tomato - Healthy",
}

# Disease information for display
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot causes leaf and fruit lesions on pepper plants.",
        "remedy": "Use copper-based bactericides and practice crop rotation.",
    },
    "Pepper__bell___healthy": {
        "description": "The plant appears healthy with no signs of disease.",
        "remedy": "Continue regular care and monitoring.",
    },
    "Potato___Early_blight": {
        "description": "Early blight causes dark concentric spots on older leaves.",
        "remedy": "Apply fungicides and remove infected plant debris.",
    },
    "Potato___Late_blight": {
        "description": "Late blight causes water-soaked lesions that turn brown.",
        "remedy": "Use resistant varieties and apply fungicides preventively.",
    },
    "Potato___healthy": {
        "description": "The plant appears healthy with no signs of disease.",
        "remedy": "Continue regular care and monitoring.",
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot causes small, dark, raised spots on leaves.",
        "remedy": "Use disease-free seeds and copper-based sprays.",
    },
    "Tomato_Early_blight": {
        "description": "Early blight shows as dark spots with concentric rings.",
        "remedy": "Prune lower branches and apply appropriate fungicides.",
    },
    "Tomato_Late_blight": {
        "description": "Late blight causes large, dark, water-soaked lesions.",
        "remedy": "Remove infected plants immediately and use fungicides.",
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold appears as yellow spots on upper leaf surface.",
        "remedy": "Improve air circulation and reduce humidity.",
    },
    "Tomato_healthy": {
        "description": "The plant appears healthy with no signs of disease.",
        "remedy": "Continue regular care and monitoring.",
    },
}


# ============================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================


def preprocess_image(image_path):
    """
    Load and preprocess a single image for prediction.

    Steps:
    1. Read the image using OpenCV
    2. Convert BGR to RGB (OpenCV loads as BGR)
    3. Resize to 224x224 (MobileNet input size)
    4. Normalize pixel values to [0, 1]
    5. Add batch dimension

    Args:
        image_path (str): Path to the image file

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
                       Shape: (1, 224, 224, 3)
    """
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image at: {image_path}")

    # Convert from BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to the required input size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)

    return image


def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess an image uploaded through Streamlit.

    This function handles the uploaded file object from Streamlit's
    file_uploader widget and converts it to a format suitable for
    model prediction.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        tuple: (preprocessed_image, display_image)
            - preprocessed_image: numpy array ready for prediction (1, 224, 224, 3)
            - display_image: numpy array for display (224, 224, 3) in RGB
    """
    # Read the uploaded file bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the uploaded image.")

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize for the model
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))

    # Create display version (before normalization)
    display_image = image_resized.copy()

    # Normalize for model input
    image_normalized = image_resized.astype(np.float32) / 255.0

    # Add batch dimension
    preprocessed = np.expand_dims(image_normalized, axis=0)

    return preprocessed, display_image


# ============================================================
# CLASS LABEL FUNCTIONS
# ============================================================


def save_class_indices(class_indices):
    """
    Save the class indices mapping to a JSON file.

    Args:
        class_indices (dict): Dictionary mapping class names to indices
    """
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(class_indices, f, indent=4)

    print(f"[INFO] Class indices saved to: {CLASS_INDICES_PATH}")


def load_class_indices():
    """
    Load the class indices mapping from JSON file.

    Returns:
        dict: Dictionary mapping class names to indices

    Raises:
        FileNotFoundError: If the class indices file doesn't exist
    """
    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(
            f"Class indices file not found at: {CLASS_INDICES_PATH}\n"
            "Please train the model first by running: python train_model.py"
        )

    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)

    return class_indices


def get_class_name(predicted_index, class_indices):
    """
    Convert a predicted class index back to the class name.

    Args:
        predicted_index (int): The predicted class index from the model
        class_indices (dict): Dictionary mapping class names to indices

    Returns:
        str: The class name corresponding to the predicted index
    """
    # Reverse the dictionary: {index: class_name}
    index_to_class = {v: k for k, v in class_indices.items()}

    return index_to_class.get(predicted_index, "Unknown")


def get_display_name(class_name):
    """
    Get the human-readable display name for a class.

    Args:
        class_name (str): Raw class name from the dataset

    Returns:
        str: Human-readable display name
    """
    return CLASS_DISPLAY_NAMES.get(class_name, class_name.replace("_", " "))


def get_disease_info(class_name):
    """
    Get disease description and remedy information.

    Args:
        class_name (str): Raw class name from the dataset

    Returns:
        dict: Dictionary with 'description' and 'remedy' keys
    """
    return DISEASE_INFO.get(
        class_name,
        {
            "description": "No information available for this class.",
            "remedy": "Please consult an agricultural expert.",
        },
    )


# ============================================================
# DATASET HELPER FUNCTIONS
# ============================================================


def verify_dataset():
    """
    Verify that the dataset directory exists and contains the expected
    class subdirectories with images.

    Returns:
        tuple: (is_valid, message, class_count, total_images)
    """
    if not os.path.exists(DATASET_DIR):
        return (
            False,
            f"Dataset directory not found at: {DATASET_DIR}\n"
            "Please follow the README instructions to download the dataset.",
            0,
            0,
        )

    # Check for subdirectories (each class should be a subdirectory)
    classes = [
        d
        for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    if len(classes) == 0:
        return (
            False,
            "No class subdirectories found in the dataset directory.\n"
            "The dataset should have subdirectories for each plant disease class.",
            0,
            0,
        )

    # Count total images
    total_images = 0
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        images = [
            f
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        total_images += len(images)

    if total_images == 0:
        return (
            False,
            "No images found in the dataset class directories.",
            len(classes),
            0,
        )

    return (
        True,
        f"Dataset verified successfully!\n"
        f"Found {len(classes)} classes with {total_images} total images.",
        len(classes),
        total_images,
    )


def create_sample_dataset():
    """
    Create a small sample dataset with synthetic images for testing purposes.

    This function generates colored noise images organized into class
    subdirectories. This is ONLY for testing the pipeline - for real
    results, use the actual PlantVillage dataset.

    Creates 30 sample images per class for quick testing.
    """
    print("[INFO] Creating sample dataset for testing...")
    print("[WARNING] This creates SYNTHETIC images for pipeline testing only!")
    print("[WARNING] For real predictions, use the actual PlantVillage dataset.\n")

    os.makedirs(DATASET_DIR, exist_ok=True)

    # Color hints for different classes (just for visual variety in BGR)
    class_colors = {
        "Pepper__bell___Bacterial_spot": (139, 69, 19),
        "Pepper__bell___healthy": (34, 139, 34),
        "Potato___Early_blight": (160, 82, 45),
        "Potato___Late_blight": (105, 105, 105),
        "Potato___healthy": (0, 128, 0),
        "Tomato_Bacterial_spot": (178, 34, 34),
        "Tomato_Early_blight": (210, 105, 30),
        "Tomato_Late_blight": (128, 0, 0),
        "Tomato_Leaf_Mold": (107, 142, 35),
        "Tomato_healthy": (50, 205, 50),
    }

    samples_per_class = 30  # Small number for quick training

    for class_name, base_color in class_colors.items():
        class_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            # Create a synthetic image with noise and the class-specific color
            image = np.random.randint(0, 50, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            # Add base color tint
            for c in range(3):
                image[:, :, c] = np.clip(
                    image[:, :, c].astype(np.int16) + base_color[c], 0, 255
                ).astype(np.uint8)

            # Add some random shapes to make images different
            for _ in range(np.random.randint(3, 8)):
                center = (
                    np.random.randint(0, IMG_SIZE),
                    np.random.randint(0, IMG_SIZE),
                )
                radius = np.random.randint(5, 30)
                color = tuple(int(c) for c in np.random.randint(0, 255, 3))
                cv2.circle(image, center, radius, color, -1)

            # Save the image
            filename = f"sample_{i:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            cv2.imwrite(filepath, image)

        print(f"  Created {samples_per_class} samples for: {class_name}")

    print(f"\n[INFO] Sample dataset created at: {DATASET_DIR}")
    print(
        f"[INFO] Total: {len(class_colors)} classes x {samples_per_class} images "
        f"= {len(class_colors) * samples_per_class} images"
    )


# ============================================================
# MODEL HELPER FUNCTIONS
# ============================================================


def print_model_summary_info(model):
    """
    Print a clean summary of the model architecture.

    Args:
        model: Keras model instance
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    model.summary()
    print("=" * 60)

    # Count trainable and non-trainable parameters
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(
        np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    )

    print(f"\nTrainable parameters:     {trainable:>12,}")
    print(f"Non-trainable parameters: {non_trainable:>12,}")
    print(f"Total parameters:         {trainable + non_trainable:>12,}")
    print("=" * 60 + "\n")


# ============================================================
# MAIN - Run utility checks
# ============================================================

if __name__ == "__main__":
    print("Plant Disease Detection - Utility Module")
    print("=" * 50)

    # Verify dataset
    is_valid, message, class_count, total_images = verify_dataset()
    print(f"\nDataset Status: {'VALID' if is_valid else 'NOT FOUND'}")
    print(message)

    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"\nModel found at: {MODEL_PATH}")
    else:
        print(f"\nModel not found. Train it with: python train_model.py")

    # Check class indices
    if os.path.exists(CLASS_INDICES_PATH):
        class_indices = load_class_indices()
        print(f"Class indices found: {len(class_indices)} classes")
    else:
        print("Class indices not found. Train the model first.")