import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import time

# --- KONFIGURASI ---
DATA_ROOT = "data_mp_wajah"
MODEL_FILENAME = "face_cnn_model.h5"
METADATA_FILENAME = "face_cnn_metadata.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# CNN Hyperparameters
IMG_HEIGHT = 64  # Reshape landmarks ke format image-like
IMG_WIDTH = 68   # 468 landmarks -> 64x68 (closest to square)
CHANNELS = 3     # x, y, z coordinates

def load_data():
    """Memuat data fitur wajah dari direktori"""
    features = []
    labels_raw = []
    
    print("Memuat data fitur wajah...")
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Direktori '{DATA_ROOT}' tidak ditemukan!")
        print("Jalankan 'python data_collect.py' terlebih dahulu.")
        return None, None
    
    total_files = 0
    for name in os.listdir(DATA_ROOT):
        person_path = os.path.join(DATA_ROOT, name)
        if os.path.isdir(person_path):
            person_files = 0
            for filename in os.listdir(person_path):
                if filename.endswith(".npy"):
                    file_path = os.path.join(person_path, filename)
                    try:
                        feature = np.load(file_path)
                        features.append(feature)
                        labels_raw.append(name)
                        person_files += 1
                        total_files += 1
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            print(f"  {name}: {person_files} sampel")
    
    if not features:
        print("Tidak ada fitur yang berhasil dimuat!")
        return None, None
    
    print(f"Total sampel dimuat: {total_files}")
    return np.array(features), np.array(labels_raw)

def preprocess_landmarks_to_image(features):
    """Convert MediaPipe landmarks menjadi format image untuk CNN"""
    print("Preprocessing landmarks untuk CNN...")
    
    # MediaPipe face mesh memiliki 468 landmarks dengan koordinat x,y,z
    # Total: 468 * 3 = 1404 features
    n_landmarks = features.shape[1] // 3
    
    processed_features = []
    
    for feature in features:
        # Reshape dari (1404,) ke (468, 3)
        landmarks_3d = feature.reshape(n_landmarks, 3)
        
        # Normalisasi koordinat ke range [0, 1]
        x_coords = landmarks_3d[:, 0]
        y_coords = landmarks_3d[:, 1] 
        z_coords = landmarks_3d[:, 2]
        
        # Normalize to [0, 1]
        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-8)
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-8)
        z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-8)
        
        # Create image-like representation
        # Pad landmarks to make it rectangular (468 -> 64x68 = 4352, trim excess)
        target_size = IMG_HEIGHT * IMG_WIDTH
        
        if n_landmarks >= target_size:
            # Trim excess landmarks
            x_img = x_norm[:target_size].reshape(IMG_HEIGHT, IMG_WIDTH)
            y_img = y_norm[:target_size].reshape(IMG_HEIGHT, IMG_WIDTH)
            z_img = z_norm[:target_size].reshape(IMG_HEIGHT, IMG_WIDTH)
        else:
            # Pad with zeros if needed
            x_padded = np.pad(x_norm, (0, target_size - n_landmarks), 'constant')
            y_padded = np.pad(y_norm, (0, target_size - n_landmarks), 'constant')
            z_padded = np.pad(z_norm, (0, target_size - n_landmarks), 'constant')
            
            x_img = x_padded.reshape(IMG_HEIGHT, IMG_WIDTH)
            y_img = y_padded.reshape(IMG_HEIGHT, IMG_WIDTH)
            z_img = z_padded.reshape(IMG_HEIGHT, IMG_WIDTH)
        
        # Stack coordinates sebagai channels (height, width, channels)
        img_feature = np.stack([x_img, y_img, z_img], axis=-1)
        processed_features.append(img_feature)
    
    return np.array(processed_features)

def create_cnn_model(num_classes):
    """Membuat model CNN untuk face recognition"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
    ])
    
    return model

def train_cnn_model(X_train, X_test, y_train, y_test, label_encoder):
    """Melatih model CNN"""
    print(f"\n=== TRAINING CNN ===")
    print(f"Input shape: {X_train.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    start_time = time.time()
    
    # Create model
    num_classes = len(label_encoder.classes_)
    model = create_cnn_model(num_classes)
    
    # Compile model
    if num_classes > 1:
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=metrics
    )
    
    # Model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Data augmentation untuk meningkatkan generalisasi
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Wajah sebaiknya tidak di-flip
        fill_mode='nearest'
    )
    
    # Training
    print("Memulai training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluasi
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Prediksi untuk classification report
    y_pred_prob = model.predict(X_test)
    if num_classes > 1:
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        y_pred = (y_pred_prob > 0.5).astype(int)
    
    training_time = time.time() - start_time
    
    print(f"\nTraining time: {training_time:.2f} detik")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return model, history, test_accuracy

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history plot saved as 'cnn_training_history.png'")

def save_cnn_model(model, label_encoder, metadata):
    """Simpan model CNN dan metadata"""
    # Simpan model
    model.save(MODEL_FILENAME)
    
    # Simpan metadata
    metadata_data = {
        'encoder': label_encoder,
        'model_type': 'CNN',
        'classes': list(label_encoder.classes_),
        'img_height': IMG_HEIGHT,
        'img_width': IMG_WIDTH,
        'channels': CHANNELS,
        **metadata
    }
    
    with open(METADATA_FILENAME, 'wb') as f:
        pickle.dump(metadata_data, f)
    
    print(f"Model CNN disimpan: {MODEL_FILENAME}")
    print(f"Metadata disimpan: {METADATA_FILENAME}")

def main():
    print("=== CNN FACE RECOGNITION MODEL TRAINING ===")
    print("Note: CNN memerlukan lebih banyak data untuk hasil optimal (minimal 50+ sampel per kelas)")
    
    # Load data
    features, labels_raw = load_data()
    if features is None:
        return
    
    # Check data sufficiency
    unique_classes = len(set(labels_raw))
    min_samples = min([list(labels_raw).count(cls) for cls in set(labels_raw)])
    
    print(f"\nAnalisis Dataset:")
    print(f"  Jumlah kelas: {unique_classes}")
    print(f"  Sampel minimum per kelas: {min_samples}")
    
    if unique_classes < 2:
        print(f"\nError: CNN memerlukan minimal 2 kelas untuk training.")
        print("Kumpulkan data untuk minimal 1 orang lagi.")
        return
    
    if min_samples < 30:
        print(f"\nWarning: Data terlalu sedikit untuk CNN (minimal 30+ per kelas recommended)")
        print("CNN mungkin overfit. Pertimbangkan untuk:")
        print("1. Kumpulkan lebih banyak data")
        print("2. Gunakan KNN/SVM untuk dataset kecil")
        
        choice = input("\nLanjutkan training CNN? (y/n): ")
        if choice.lower() != 'y':
            return
    
    # Label encoding
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_raw)
    
    print(f"\nInfo Dataset:")
    print(f"  Jumlah sampel: {len(features)}")
    print(f"  Jumlah kelas: {len(label_encoder.classes_)}")
    print(f"  Kelas: {list(label_encoder.classes_)}")
    print(f"  Dimensi fitur asli: {features.shape[1]}")
    
    # Preprocess landmarks to image format
    processed_features = preprocess_landmarks_to_image(features)
    print(f"  Dimensi fitur CNN: {processed_features.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        processed_features, labels_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_encoded
    )
    
    print(f"\nSplit dataset:")
    print(f"  Training: {len(X_train)} sampel")
    print(f"  Testing: {len(X_test)} sampel")
    
    # Train CNN
    model, history, cnn_accuracy = train_cnn_model(X_train, X_test, y_train, y_test, label_encoder)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    metadata = {
        'test_accuracy': cnn_accuracy,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'epochs_trained': len(history.history['accuracy'])
    }
    
    save_cnn_model(model, label_encoder, metadata)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    print(f"Model type: Deep Convolutional Neural Network")
    print(f"Total parameters: {model.count_params():,}")
    print("\nTraining CNN selesai! Model siap digunakan.")
    print("Untuk testing: python test_recog.py (pilih CNN)")

if __name__ == "__main__":
    # Set memory growth untuk GPU jika tersedia
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()