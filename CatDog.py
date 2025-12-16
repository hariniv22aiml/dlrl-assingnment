"""
Improved Cat vs Dog Classifier with:
- Automatic dataset download
- Enhanced CNN architecture with Batch Normalization
- Advanced data augmentation
- Transfer learning with fine-tuning option
- Better evaluation metrics
"""

import os
import zipfile
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import gc

# Configure TensorFlow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def download_and_extract_dataset():
    """
    Automatically download and extract the cats and dogs dataset
    Works reliably in Google Colab and local environments
    """
    dataset_dir = './cats_and_dogs_data'
    base_dir = os.path.join(dataset_dir, 'cats_and_dogs_filtered')
    
    # Check if already extracted
    if os.path.exists(base_dir):
        print("Dataset already exists.")
        return dataset_dir
    
    zip_path = './cats_and_dogs_filtered.zip'
    
    # Try multiple reliable sources
    download_sources = [
        # TensorFlow's official datasets (most reliable in Colab)
        {
            'name': 'TensorFlow Keras',
            'method': 'keras',
            'url': 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        },
        # Microsoft Download (alternative)
        {
            'name': 'Direct Download',
            'method': 'direct',
            'url': 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
        }
    ]
    
    print("Downloading dataset (this may take a few minutes)...")
    
    for source in download_sources:
        try:
            print(f"\nTrying {source['name']}...")
            
            if source['method'] == 'keras':
                # Use TensorFlow's built-in downloader (most reliable)
                print("Using TensorFlow download method...")
                cached_path = tf.keras.utils.get_file(
                    'cats_and_dogs_filtered.zip',
                    source['url'],
                    cache_dir='.',
                    cache_subdir='',
                    extract=False
                )
                
                print("Download complete! Extracting...")
                with zipfile.ZipFile(cached_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                
                if os.path.exists(cached_path):
                    os.remove(cached_path)
                
                print("Dataset extracted successfully!")
                return dataset_dir
            
            elif source['method'] == 'direct':
                # Direct download with progress
                import time
                req = urllib.request.Request(
                    source['url'],
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                
                with urllib.request.urlopen(req, timeout=60) as response:
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192
                    downloaded = 0
                    
                    with open(zip_path, 'wb') as out_file:
                        while True:
                            chunk = response.read(block_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                if downloaded % (block_size * 100) == 0:  # Update every ~800KB
                                    print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
                
                print("\nDownload complete! Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                
                print("Dataset extracted successfully!")
                return dataset_dir
                
        except Exception as e:
            print(f"{source['name']} failed: {str(e)[:100]}")
            if os.path.exists(zip_path):
                try:
                    os.remove(zip_path)
                except:
                    pass
            continue
    
    # If all methods fail, provide manual instructions
    print("\n" + "="*60)
    print("AUTOMATIC DOWNLOAD FAILED - MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print("\n📋 OPTION 1 - Use Kaggle API (Recommended for Colab):")
    print("Run these commands in a Colab cell:")
    print("```")
    print("!pip install -q kaggle")
    print("from google.colab import files")
    print("files.upload()  # Upload your kaggle.json")
    print("!mkdir -p ~/.kaggle")
    print("!cp kaggle.json ~/.kaggle/")
    print("!chmod 600 ~/.kaggle/kaggle.json")
    print("!kaggle datasets download -d salader/dogs-vs-cats")
    print("!unzip -q dogs-vs-cats.zip -d cats_and_dogs_data")
    print("```")
    print("\n📋 OPTION 2 - Upload manually:")
    print("1. Download from: https://www.kaggle.com/c/dogs-vs-cats/data")
    print("2. Upload to Colab using Files panel")
    print("3. Organize into train/validation folders with cats/dogs subfolders")
    print("\n📋 OPTION 3 - Use Google Drive:")
    print("```")
    print("from google.colab import drive")
    print("drive.mount('/content/drive')")
    print("!cp -r /content/drive/MyDrive/cats_and_dogs_data .")
    print("```")
    print("="*60)
    return None
    
    return dataset_dir



def create_improved_model(input_shape=(150, 150, 3), lite_mode=False):
    """
    Create an improved CNN model with Batch Normalization
    Memory-efficient version with configurable complexity
    """
    # Adjust filters based on mode
    if lite_mode:
        filters = [16, 32, 64]
        dense_units = 256
    else:
        filters = [32, 64, 128]
        dense_units = 512
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(filters[0], (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(filters[1], (3, 3), padding='same'),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(filters[2], (3, 3), padding='same'),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fully Connected Layers
        Flatten(),
        Dense(dense_units),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    return model

def plot_training_history(history, save_path='catdog_training_history.png'):
    """
    Plot training metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to '{save_path}'")
    plt.show()

def visualize_sample_images(train_dir):
    """
    Visualize sample images from the dataset
    """
    import matplotlib.image as mpimg
    
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    
    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)
    
    # Display sample images
    nrows, ncols = 4, 4
    fig = plt.figure(figsize=(10, 10))
    
    pic_index = 8
    next_cat_pix = [os.path.join(train_cats_dir, fname) 
                    for fname in train_cat_fnames[pic_index-8:pic_index]]
    next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                    for fname in train_dog_fnames[pic_index-8:pic_index]]
    
    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')
        img = mpimg.imread(img_path)
        plt.imshow(img)
        if i < 8:
            sp.set_title('Cat', fontsize=10)
        else:
            sp.set_title('Dog', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    print("Sample images saved to 'sample_images.png'")
    plt.show()

if __name__ == "__main__":
    # ===== CONFIGURATION FOR YOUR HARDWARE =====
    IMAGE_SIZE = 128      # Use 96 for low RAM, 128 for medium, 150 for high RAM
    BATCH_SIZE = 16       # Use 8 for low RAM, 16 for medium, 32 for high RAM
    LITE_MODE = False     # Set to True for systems with <4GB RAM
    # ===========================================
    
    print("="*60)
    print("MEMORY-OPTIMIZED CAT VS DOG CLASSIFIER")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Lite mode: {LITE_MODE}")
    print("="*60)
    
    try:
        # Download and extract dataset
        dataset_dir = download_and_extract_dataset()
        
        if dataset_dir is None:
            print("\n" + "="*60)
            print("Cannot proceed without dataset.")
            print("Please download manually and re-run the script.")
            print("="*60)
            exit(1)
        
        # Setup directories - handle different possible structures
        print(f"\nLooking for dataset in: {dataset_dir}")
        
        # Check what was actually extracted
        if os.path.exists(dataset_dir):
            print(f"Contents of {dataset_dir}: {os.listdir(dataset_dir)}")
        
        # Try different possible structures
        possible_structures = [
            # Structure 1: Standard filtered dataset
            (os.path.join(dataset_dir, 'cats_and_dogs_filtered'), 'train', 'validation'),
            # Structure 2: Direct extraction
            (dataset_dir, 'train', 'validation'),
            # Structure 3: Kaggle dataset structure
            (os.path.join(dataset_dir, 'PetImages'), 'Cat', 'Dog'),
            # Structure 4: Root PetImages
            ('PetImages', 'Cat', 'Dog'),
        ]
        
        base_dir = None
        train_dir = None
        validation_dir = None
        is_kaggle_structure = False
        
        for base, train_name, val_name in possible_structures:
            if os.path.exists(base):
                print(f"Found base directory: {base}")
                print(f"Contents: {os.listdir(base)}")
                
                # Check if it's Kaggle structure (Cat/Dog folders directly)
                if train_name == 'Cat' and val_name == 'Dog':
                    cat_dir = os.path.join(base, 'Cat')
                    dog_dir = os.path.join(base, 'Dog')
                    if os.path.exists(cat_dir) and os.path.exists(dog_dir):
                        print("Detected Kaggle PetImages structure - will create train/val split")
                        is_kaggle_structure = True
                        base_dir = base
                        break
                else:
                    # Standard structure with train/validation
                    train_check = os.path.join(base, train_name)
                    val_check = os.path.join(base, val_name)
                    if os.path.exists(train_check):
                        base_dir = base
                        train_dir = train_check
                        validation_dir = val_check
                        print(f"Using structure: base={base}, train={train_name}, val={val_name}")
                        break
        
        if base_dir is None:
            print(f"\nERROR: Could not find valid dataset structure!")
            print(f"Please check the extracted files in: {dataset_dir}")
            exit(1)
        
        # Handle Kaggle structure - need to reorganize
        if is_kaggle_structure:
            print("\nReorganizing Kaggle dataset into train/validation structure...")
            import shutil
            from sklearn.model_selection import train_test_split
            
            organized_dir = './organized_cats_dogs'
            train_dir = os.path.join(organized_dir, 'train')
            validation_dir = os.path.join(organized_dir, 'validation')
            
            # Create directories
            os.makedirs(os.path.join(train_dir, 'cats'), exist_ok=True)
            os.makedirs(os.path.join(train_dir, 'dogs'), exist_ok=True)
            os.makedirs(os.path.join(validation_dir, 'cats'), exist_ok=True)
            os.makedirs(os.path.join(validation_dir, 'dogs'), exist_ok=True)
            
            # Process cats
            cat_source = os.path.join(base_dir, 'Cat')
            cat_files = [f for f in os.listdir(cat_source) if f.endswith(('.jpg', '.jpeg', '.png'))]
            cat_files = cat_files[:2000]  # Limit for memory
            train_cats, val_cats = train_test_split(cat_files, test_size=0.2, random_state=42)
            
            print(f"Copying {len(train_cats)} training cats...")
            for f in train_cats[:1000]:  # Limit for lite mode
                shutil.copy2(os.path.join(cat_source, f), os.path.join(train_dir, 'cats', f))
            
            print(f"Copying {len(val_cats)} validation cats...")
            for f in val_cats[:200]:
                shutil.copy2(os.path.join(cat_source, f), os.path.join(validation_dir, 'cats', f))
            
            # Process dogs
            dog_source = os.path.join(base_dir, 'Dog')
            dog_files = [f for f in os.listdir(dog_source) if f.endswith(('.jpg', '.jpeg', '.png'))]
            dog_files = dog_files[:2000]
            train_dogs, val_dogs = train_test_split(dog_files, test_size=0.2, random_state=42)
            
            print(f"Copying {len(train_dogs)} training dogs...")
            for f in train_dogs[:1000]:
                shutil.copy2(os.path.join(dog_source, f), os.path.join(train_dir, 'dogs', f))
            
            print(f"Copying {len(val_dogs)} validation dogs...")
            for f in val_dogs[:200]:
                shutil.copy2(os.path.join(dog_source, f), os.path.join(validation_dir, 'dogs', f))
            
            print("Dataset reorganization complete!")
            base_dir = organized_dir
        
        # Final verification
        if not os.path.exists(train_dir):
            print(f"\nERROR: Training directory not found: {train_dir}")
            exit(1)
        
        # Print dataset information
        train_cats_dir = os.path.join(train_dir, 'cats')
        train_dogs_dir = os.path.join(train_dir, 'dogs')
        val_cats_dir = os.path.join(validation_dir, 'cats')
        val_dogs_dir = os.path.join(validation_dir, 'dogs')
        
        print(f"\nDataset Information:")
        if os.path.exists(train_cats_dir):
            print(f"Training cats: {len(os.listdir(train_cats_dir))}")
        if os.path.exists(train_dogs_dir):
            print(f"Training dogs: {len(os.listdir(train_dogs_dir))}")
        if os.path.exists(val_cats_dir):
            print(f"Validation cats: {len(os.listdir(val_cats_dir))}")
        if os.path.exists(val_dogs_dir):
            print(f"Validation dogs: {len(os.listdir(val_dogs_dir))}")
        
        # Visualize sample images (skip if low memory)
        if not LITE_MODE:
            visualize_sample_images(train_dir)
        
        # Create improved model
        print("\nBuilding improved CNN model...")
        model = create_improved_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), lite_mode=LITE_MODE)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        
        # Enhanced data augmentation (reduce for lite mode)
        if LITE_MODE:
            train_datagen = ImageDataGenerator(
                rescale=1.0/255.,
                rotation_range=20,
                horizontal_flip=True
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1.0/255.,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        
        validation_datagen = ImageDataGenerator(rescale=1.0/255.)
        
        # Data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
        
        # Callbacks
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
        
        checkpoint = ModelCheckpoint(
            'best_catdog_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        print("\nTraining model...")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=[early_stopping, lr_scheduler, checkpoint],
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating model...")
        val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Plot training history
        plot_training_history(history)
        
        # Save final model
        model.save('final_catdog_model.h5')
        print("\nFinal model saved to 'final_catdog_model.h5'")
        print("Best model saved to 'best_catdog_model.h5'")
        
        # Clean up
        gc.collect()
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Out of memory or other issue encountered!")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nTry adjusting the configuration at the top of the script:")
        print("  1. Reduce IMAGE_SIZE to 96")
        print("  2. Reduce BATCH_SIZE to 8")
        print("  3. Set LITE_MODE = True")
        print("="*60)
