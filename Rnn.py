"""
Improved RNN Text Generation with:
- GRU layers (better than SimpleRNN)
- Multiple stacked layers for deeper learning
- Temperature-based sampling for controlled randomness
- Word-level and character-level generation
- Better training with dropout and callbacks
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import gc

# Configure TensorFlow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ===== CONFIGURATION =====
LITE_MODE = False  # Set to True for systems with <4GB RAM
# =========================

# Enhanced text corpus
text = """The beautiful girl whom I met last time is very intelligent also.
She enjoys reading books about science and technology.
The handsome boy whom I met last time is very intelligent also.
He loves playing chess and solving complex problems.
They both appreciate art and music in their free time.
Learning new things brings them great joy and satisfaction.
"""

print("=" * 60)
print("MEMORY-OPTIMIZED TEXT GENERATION WITH GRU")
print("=" * 60)
print(f"Configuration: {'LITE' if LITE_MODE else 'STANDARD'} mode")
print(f"\nTraining corpus length: {len(text)} characters")
print(f"Training text:\n{text}\n")

# Character-level processing
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"Unique characters: {chars}\n")

# Create training sequences
seq_length = 40 if not LITE_MODE else 20  # Shorter sequences for lite mode
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

print(f"Training sequences: {len(sequences)}")
print(f"Sequence shape: {X.shape}")

# One-hot encode
X_one_hot = tf.one_hot(X, vocab_size)
y_one_hot = tf.one_hot(y, vocab_size)

# Build improved GRU model (memory-efficient)
print("\nBuilding GRU Model...")

if LITE_MODE:
    print("Using LITE mode (reduced model size)")
    model = Sequential([
        GRU(64, input_shape=(seq_length, vocab_size), return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    batch_size = 16
    epochs = 100
else:
    model = Sequential([
        GRU(128, input_shape=(seq_length, vocab_size), return_sequences=True),
        Dropout(0.2),
        GRU(128, return_sequences=True),
        Dropout(0.2),
        GRU(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    batch_size = 32
    epochs = 200

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks for better training
early_stopping = EarlyStopping(
    monitor='loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# Train the model
print(f"\nTraining model (batch_size={batch_size}, max_epochs={epochs})...")
try:
    history = model.fit(
        X_one_hot, y_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    # Clean up
    del X_one_hot, y_one_hot
    gc.collect()
except Exception as e:
    print(f"\nERROR during training: {e}")
    print("Try setting LITE_MODE = True at the top of the script")
    raise

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], linewidth=2, color='blue')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], linewidth=2, color='green')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_training_history.png', dpi=300, bbox_inches='tight')
print("\nTraining history saved to 'rnn_training_history.png'")
plt.show()

def generate_text_with_temperature(seed_text, length=100, temperature=1.0):
    """
    Generate text with temperature-based sampling
    temperature < 1.0: more conservative (predictable)
    temperature > 1.0: more random (creative)
    """
    generated_text = seed_text
    
    for i in range(length):
        # Prepare input sequence
        if len(generated_text) < seq_length:
            input_seq = generated_text.ljust(seq_length)
        else:
            input_seq = generated_text[-seq_length:]
        
        x = np.array([[char_to_index.get(char, 0) for char in input_seq]])
        x_one_hot = tf.one_hot(x, vocab_size)
        
        # Predict next character
        prediction = model.predict(x_one_hot, verbose=0)[0]
        
        # Apply temperature
        prediction = np.log(prediction + 1e-10) / temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        
        # Sample from distribution
        next_index = np.random.choice(len(prediction), p=prediction)
        next_char = index_to_char[next_index]
        generated_text += next_char
    
    return generated_text

# Generate text with different seed texts and temperatures
print("\n" + "=" * 60)
print("TEXT GENERATION EXAMPLES")
print("=" * 60)

seed_texts = [
    "The beautiful girl whom I met ",
    "The handsome boy whom I met ",
    "She enjoys reading books about ",
    "He loves playing chess and "
]

temperatures = [0.5, 1.0, 1.5]

for seed in seed_texts:
    print(f"\n{'='*60}")
    print(f"Seed: '{seed}'")
    print(f"{'='*60}")
    
    for temp in temperatures:
        generated = generate_text_with_temperature(seed, length=100, temperature=temp)
        print(f"\nTemperature {temp}:")
        print(f"{generated}\n")
        print("-" * 60)

# Generate longer text samples
print("\n" + "=" * 60)
print("LONG-FORM GENERATION (Temperature = 1.0)")
print("=" * 60)

long_seed = "The "
long_generated = generate_text_with_temperature(long_seed, length=300, temperature=1.0)
print(f"\n{long_generated}\n")

# Save the model
model.save('improved_gru_text_generator.h5')
print("\n" + "=" * 60)
print("Model saved to 'improved_gru_text_generator.h5'")
print("=" * 60)

# Model performance summary
final_loss = history.history['loss'][-1]
final_accuracy = history.history['accuracy'][-1]
print(f"\nFinal Training Loss: {final_loss:.4f}")
print(f"Final Training Accuracy: {final_accuracy:.4f}")
print(f"\nTotal training epochs: {len(history.history['loss'])}")
