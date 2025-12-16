"""
Improved AlexNet Implementation with:
- Batch Normalization for faster convergence
- Modern activation functions (ReLU)
- Better regularization techniques
- CIFAR-10 dataset integration with automatic download
- Training and evaluation pipeline
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import gc
import os

# Memory optimization: Use CPU if GPU memory is limited
# Comment out the next line if you have sufficient GPU memory
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure TensorFlow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), memory growth enabled")
    except RuntimeError as e:
        print(e)

class ImprovedAlexNet(Sequential):
    """
    Enhanced AlexNet with Batch Normalization and modern best practices
    Memory-efficient version for systems with limited RAM/GPU
    """
    def __init__(self, input_shape, num_classes, lite_mode=False):
        super().__init__()
        
        # Adjust filters based on mode
        filters = [48, 128, 192, 192, 128] if lite_mode else [64, 128, 256, 256, 128]

        # First Convolutional Block with BatchNorm
        self.add(Conv2D(filters[0], kernel_size=(5, 5), strides=2, padding='same', 
                       input_shape=input_shape, kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        # Second Convolutional Block with BatchNorm
        self.add(Conv2D(filters[1], kernel_size=(3, 3), strides=1, padding='same',
                       kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        # Third Convolutional Block
        self.add(Conv2D(filters[2], kernel_size=(3, 3), strides=1, padding='same',
                       kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        
        # Fourth Convolutional Block
        self.add(Conv2D(filters[3], kernel_size=(3, 3), strides=1, padding='same',
                       kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        
        # Fifth Convolutional Block
        self.add(Conv2D(filters[4], kernel_size=(3, 3), strides=1, padding='same',
                       kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        # Flatten Layer
        self.add(Flatten())

        # Fully Connected Layers with BatchNorm (memory-efficient)
        dense_size = 256 if lite_mode else 512
        self.add(Dense(dense_size, kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        self.add(Dropout(0.5))
        
        self.add(Dense(dense_size // 2, kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        self.add(Dropout(0.5))
        
        self.add(Dense(num_classes, activation='softmax'))

def load_and_preprocess_data(use_subset=False, target_size=64):
    """
    Load CIFAR-10 dataset and preprocess with memory-efficient approach
    
    Args:
        use_subset: If True, use only 20% of data for faster training on limited hardware
        target_size: Target image size (default 64 for memory efficiency)
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Use subset for memory-constrained systems
    if use_subset:
        print("Using subset of data (20%) for memory efficiency...")
        train_samples = len(x_train) // 5
        test_samples = len(x_test) // 5
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]
    
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Resize images in small batches to avoid OOM errors
    print(f"Resizing images to {target_size}x{target_size} (this may take a moment)...")
    batch_size = 500  # Small batch size for memory safety
    
    # Resize training images in batches
    x_train_resized = []
    for i in range(0, len(x_train), batch_size):
        batch = x_train[i:i+batch_size]
        resized_batch = tf.image.resize(batch, [target_size, target_size]).numpy()
        x_train_resized.append(resized_batch)
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(x_train))}/{len(x_train)} training images...")
        gc.collect()  # Force garbage collection
    x_train_resized = np.concatenate(x_train_resized, axis=0)
    
    # Clear original training data from memory
    del x_train
    gc.collect()
    
    # Resize test images in batches
    x_test_resized = []
    for i in range(0, len(x_test), batch_size):
        batch = x_test[i:i+batch_size]
        resized_batch = tf.image.resize(batch, [target_size, target_size]).numpy()
        x_test_resized.append(resized_batch)
        gc.collect()
    x_test_resized = np.concatenate(x_test_resized, axis=0)
    
    # Clear original test data from memory
    del x_test
    gc.collect()
    
    print("Resizing complete!")
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train_resized, y_train, x_test_resized, y_test

def plot_training_history(history):
    """
    Plot training metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('alexnet_training_history.png')
    print("Training history saved to 'alexnet_training_history.png'")
    plt.show()

if __name__ == "__main__":
    # ===== CONFIGURATION FOR YOUR HARDWARE =====
    # Adjust these settings based on your available RAM/GPU
    USE_SUBSET = False  # Set to True if you have <8GB RAM
    IMAGE_SIZE = 64     # Use 64 for low RAM, 96 for medium, 128 for high RAM
    BATCH_SIZE = 32     # Use 16 for low RAM, 32 for medium, 64 for high RAM
    LITE_MODE = False   # Set to True for systems with <4GB RAM
    # ===========================================
    
    print("="*60)
    print("MEMORY-OPTIMIZED ALEXNET TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Lite mode: {LITE_MODE}")
    print(f"  - Use subset: {USE_SUBSET}")
    print("="*60)
    
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data(
        use_subset=USE_SUBSET,
        target_size=IMAGE_SIZE
    )
    
    # Model configuration
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    num_classes = 10  # CIFAR-10 has 10 classes
    
    # Create improved model
    print("\nBuilding Improved AlexNet model...")
    model = ImprovedAlexNet(input_shape, num_classes, lite_mode=LITE_MODE)
    
    # Compile with modern optimizer and learning rate schedule
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train the model
    print("\nTraining model...")
    print(f"Using batch size: {BATCH_SIZE}")
    
    try:
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=50,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        # Evaluate the model
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Plot training history
        plot_training_history(history)
        
        # Save the model
        model.save('improved_alexnet_cifar10.h5')
        print("\nModel saved to 'improved_alexnet_cifar10.h5'")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Out of memory or other issue encountered!")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nTry adjusting the configuration at the top of the script:")
        print("  1. Set USE_SUBSET = True")
        print("  2. Reduce IMAGE_SIZE to 64")
        print("  3. Reduce BATCH_SIZE to 16")
        print("  4. Set LITE_MODE = True")
        print("  5. Or uncomment the CPU-only line at the top")
        print("="*60)
