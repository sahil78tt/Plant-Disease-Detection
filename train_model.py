"""
train_model.py - Training Script for Plant Disease Detection

This script:
1. Loads and preprocesses the PlantVillage dataset
2. Creates a Transfer Learning model using MobileNet
3. Freezes the base layers and adds a custom classifier
4. Trains the model on the plant disease dataset
5. Saves the trained model and class indices
6. Generates training history plots

Usage:
    python train_model.py
    python train_model.py --create-sample
"""

import os
import sys
import numpy as np

# Suppress TensorFlow info messages for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Import our utility functions
from utils import (
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DATASET_DIR,
    MODELS_DIR,
    MODEL_PATH,
    SELECTED_CLASSES,
    save_class_indices,
    verify_dataset,
    create_sample_dataset,
    print_model_summary_info,
)


def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("[INFO] Directories verified/created.")


def create_data_generators():
    """
    Create training and validation data generators.

    Uses Keras ImageDataGenerator for:
    - Data augmentation on training set (rotation, flip, zoom, shift)
    - Simple rescaling on validation set
    - Automatic 80/20 train/validation split

    Returns:
        tuple: (train_generator, validation_generator)
    """
    print("\n[INFO] Setting up data generators...")

    # ---------------------------------------------------------
    # Training Data Generator with Data Augmentation
    # ---------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,        # Normalize pixel values to [0, 1]
        rotation_range=20,           # Random rotation up to 20 degrees
        width_shift_range=0.2,       # Random horizontal shift
        height_shift_range=0.2,      # Random vertical shift
        shear_range=0.15,            # Random shearing
        zoom_range=0.15,             # Random zoom
        horizontal_flip=True,        # Random horizontal flip
        fill_mode="nearest",         # Fill strategy for new pixels
        validation_split=0.2,        # Reserve 20% for validation
    )

    # ---------------------------------------------------------
    # Validation Data Generator (no augmentation, only rescaling)
    # ---------------------------------------------------------
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )

    # Check which classes are available in the dataset
    available_classes = []
    if os.path.exists(DATASET_DIR):
        all_dirs = [
            d
            for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ]
        # Use selected classes if available, otherwise use all
        available_classes = [c for c in SELECTED_CLASSES if c in all_dirs]
        if not available_classes:
            available_classes = sorted(all_dirs)

    print(f"[INFO] Using {len(available_classes)} classes:")
    for cls in available_classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        num_images = len(
            [
                f
                for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        print(f"  - {cls}: {num_images} images")

    # ---------------------------------------------------------
    # Create Training Generator
    # ---------------------------------------------------------
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=available_classes if available_classes else None,
        subset="training",
        shuffle=True,
        seed=42,
    )

    # ---------------------------------------------------------
    # Create Validation Generator
    # ---------------------------------------------------------
    validation_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=available_classes if available_classes else None,
        subset="validation",
        shuffle=False,
        seed=42,
    )

    # Print dataset information
    print(f"\n[INFO] Training samples:   {train_generator.samples}")
    print(f"[INFO] Validation samples: {validation_generator.samples}")
    print(f"[INFO] Number of classes:  {train_generator.num_classes}")
    print(f"[INFO] Class mapping:      {train_generator.class_indices}")

    return train_generator, validation_generator


def build_model(num_classes):
    """
    Build the Transfer Learning model using MobileNet as the base.

    Architecture:
    1. MobileNet base (pretrained on ImageNet) - FROZEN
    2. Global Average Pooling layer
    3. Dense layer (256 units, ReLU)
    4. Dropout (0.5) for regularization
    5. Dense layer (128 units, ReLU)
    6. Dropout (0.3) for regularization
    7. Output Dense layer (num_classes, softmax)

    Args:
        num_classes (int): Number of disease classes to predict

    Returns:
        keras.Model: Compiled model ready for training
    """
    print(f"\n[INFO] Building model with {num_classes} output classes...")

    # ---------------------------------------------------------
    # Step 1: Load MobileNet pretrained on ImageNet
    # ---------------------------------------------------------
    # include_top=False removes the original classification layers
    # weights='imagenet' uses weights pretrained on ImageNet
    base_model = MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # ---------------------------------------------------------
    # Step 2: Freeze the base model layers
    # ---------------------------------------------------------
    # We don't want to update the pretrained weights during training
    base_model.trainable = False

    print(
        f"[INFO] MobileNet base loaded with {len(base_model.layers)} layers (all frozen)"
    )

    # ---------------------------------------------------------
    # Step 3: Add custom classification layers on top
    # ---------------------------------------------------------
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Pass input through the base model
    x = base_model(inputs, training=False)

    # Global Average Pooling reduces each feature map to a single number
    x = GlobalAveragePooling2D()(x)

    # First fully connected layer with 256 neurons
    x = Dense(256, activation="relu", name="dense_1")(x)

    # Dropout for regularization (randomly drops 50% of connections)
    x = Dropout(0.5, name="dropout_1")(x)

    # Second fully connected layer with 128 neurons
    x = Dense(128, activation="relu", name="dense_2")(x)

    # Another dropout layer (drops 30% of connections)
    x = Dropout(0.3, name="dropout_2")(x)

    # Output layer: softmax converts outputs to probabilities
    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    # ---------------------------------------------------------
    # Step 4: Create and compile the final model
    # ---------------------------------------------------------
    model = Model(inputs=inputs, outputs=outputs, name="PlantDiseaseDetector")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print model summary
    print_model_summary_info(model)

    return model


def setup_callbacks():
    """
    Set up training callbacks for better training control.

    Returns:
        list: List of Keras callbacks
    """
    callbacks = [
        # Stop training if validation loss doesn't improve for 3 epochs
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=1,
            restore_best_weights=True,
        ),
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        # Reduce learning rate if validation loss plateaus for 2 epochs
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-7,
        ),
    ]

    return callbacks


def plot_training_history(history):
    """
    Generate and save training history plots.

    Args:
        history: Keras training history object
    """
    print("\n[INFO] Generating training plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy
    axes[0].plot(
        history.history["accuracy"],
        label="Training Accuracy",
        color="blue",
        linewidth=2,
    )
    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        color="orange",
        linewidth=2,
    )
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Plot 2: Loss
    axes[1].plot(
        history.history["loss"],
        label="Training Loss",
        color="blue",
        linewidth=2,
    )
    axes[1].plot(
        history.history["val_loss"],
        label="Validation Loss",
        color="orange",
        linewidth=2,
    )
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(MODELS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Training plots saved to: {plot_path}")


def train():
    """
    Main training function that orchestrates the entire training pipeline.
    """
    print("=" * 60)
    print("   PLANT DISEASE DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   Python version: {sys.version.split()[0]}")
    print("=" * 60)

    # Step 1: Setup
    setup_directories()

    # Step 2: Verify dataset
    is_valid, message, class_count, total_images = verify_dataset()
    print(f"\n{message}")

    if not is_valid:
        print("\n[ERROR] Dataset not found or empty!")
        print("Options:")
        print("  1. Download the PlantVillage dataset (see README.md)")
        print("  2. Run with --create-sample flag to generate test data:")
        print("     python train_model.py --create-sample")
        sys.exit(1)

    # Step 3: Create data generators
    train_gen, val_gen = create_data_generators()

    # Get number of classes
    num_classes = train_gen.num_classes

    if num_classes < 2:
        print("[ERROR] Need at least 2 classes for classification!")
        sys.exit(1)

    # Step 4: Build the model
    model = build_model(num_classes)

    # Step 5: Setup callbacks
    callbacks = setup_callbacks()

    # Step 6: Train the model
    print("\n" + "=" * 60)
    print("   STARTING TRAINING")
    print("=" * 60)
    print(f"   Epochs:     {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   LR:         {LEARNING_RATE}")
    print(f"   Classes:    {num_classes}")
    print("=" * 60 + "\n")

    # Calculate steps per epoch
    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    validation_steps = max(1, val_gen.samples // BATCH_SIZE)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Step 7: Save class indices
    save_class_indices(train_gen.class_indices)

    # Step 8: Save the final model
    model.save(MODEL_PATH)
    print(f"\n[INFO] Model saved to: {MODEL_PATH}")

    # Step 9: Generate training plots
    plot_training_history(history)

    # ---------------------------------------------------------
    # Print Final Results
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE!")
    print("=" * 60)

    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    best_val_acc = history.history["val_accuracy"][best_epoch]
    best_val_loss = history.history["val_loss"][best_epoch]

    print(f"   Best Epoch:            {best_epoch + 1}")
    print(f"   Best Val Accuracy:     {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"   Best Val Loss:         {best_val_loss:.4f}")
    print(f"   Model saved at:        {MODEL_PATH}")
    print("=" * 60)
    print("\n   Next steps:")
    print("   1. Test prediction:  python predict.py")
    print("   2. Run the app:      streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Handle --create-sample flag
    if "--create-sample" in sys.argv:
        setup_directories()
        create_sample_dataset()
        print("\n[INFO] Sample dataset created. Starting training...\n")

    # Run training
    train()