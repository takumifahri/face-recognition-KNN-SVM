import numpy as np
import os
import pickle
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import time

# --- KONFIGURASI ---
DATA_ROOT = "data_mp_wajah"
SVM_MODEL_FILENAME = "face_svm_model.pkl"
ONECLASS_MODEL_FILENAME = "oneclass_svm_model.pkl"
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
TEST_SIZE = 0.2
RANDOM_STATE = 42

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

def train_svm_model(X_train, X_test, y_train, y_test, label_encoder, scaler):
    """Melatih model SVM"""
    print(f"\n=== TRAINING SVM (kernel={SVM_KERNEL}, C={SVM_C}) ===")
    start_time = time.time()
    
    # Normalisasi data untuk SVM
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inisialisasi dan training
    svm_model = SVC(
        kernel=SVM_KERNEL, 
        C=SVM_C, 
        gamma=SVM_GAMMA,
        probability=True
    )
    svm_model.fit(X_train_scaled, y_train)
    
    # Prediksi dan evaluasi
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
    
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} detik")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Score (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return svm_model, accuracy

def train_oneclass_svm(features, labels_raw):
    """Melatih One-Class SVM untuk satu kelas"""
    print(f"\n=== TRAINING ONE-CLASS SVM ===")
    print("Note: Hanya satu kelas terdeteksi, menggunakan One-Class SVM untuk anomaly detection")
    
    start_time = time.time()
    
    # Normalisasi data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train One-Class SVM
    model = OneClassSVM(kernel=SVM_KERNEL, gamma=SVM_GAMMA, nu=0.1)
    model.fit(features_scaled)
    
    # Prediksi pada training data untuk evaluasi
    predictions = model.predict(features_scaled)
    outliers = np.sum(predictions == -1)
    
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} detik")
    print(f"Total sampel: {len(features)}")
    print(f"Outliers detected: {outliers} ({outliers/len(features)*100:.1f}%)")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels_raw)
    
    return model, label_encoder, scaler

def save_model(model, label_encoder, scaler, filename, model_type):
    """Simpan model dan komponen pendukung"""
    model_data = {
        'model': model,
        'encoder': label_encoder,
        'scaler': scaler,
        'model_type': model_type,
        'classes': list(label_encoder.classes_)
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model {model_type} disimpan: {filename}")

def main():
    print("=== SVM FACE RECOGNITION MODEL TRAINING ===")
    
    # Load data
    features, labels_raw = load_data()
    if features is None:
        return
    
    # Check number of unique classes
    unique_classes = len(set(labels_raw))
    
    if unique_classes == 1:
        print(f"\nHanya {unique_classes} kelas ditemukan: {list(set(labels_raw))}")
        print("Pilihan:")
        print("1. Kumpulkan data untuk orang lain")
        print("2. Lanjutkan dengan One-Class SVM")
        
        choice = input("\nPilih (1/2): ")
        if choice == "1":
            print("Kumpulkan data untuk orang lain menggunakan data_collect.py")
            return
        elif choice == "2":
            model, label_encoder, scaler = train_oneclass_svm(features, labels_raw)
            save_model(model, label_encoder, scaler, ONECLASS_MODEL_FILENAME, "OneClassSVM")
            print(f"\nTraining OneClass SVM selesai!")
            print(f"Untuk testing: python test_recog.py (ubah USE_MODEL = 'OneClassSVM')")
        else:
            print("Pilihan tidak valid.")
        return
    
    # Multi-class SVM training
    print(f"\n{unique_classes} kelas ditemukan: {list(set(labels_raw))}")
    
    # Label encoding
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_raw)
    
    print(f"\nInfo Dataset:")
    print(f"  Jumlah sampel: {len(features)}")
    print(f"  Jumlah kelas: {len(label_encoder.classes_)}")
    print(f"  Kelas: {list(label_encoder.classes_)}")
    print(f"  Dimensi fitur: {features.shape[1]}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=labels_encoded
    )
    
    print(f"\nSplit dataset:")
    print(f"  Training: {len(X_train)} sampel")
    print(f"  Testing: {len(X_test)} sampel")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Train SVM
    svm_model, svm_accuracy = train_svm_model(X_train, X_test, y_train, y_test, label_encoder, scaler)
    
    # Save model
    save_model(svm_model, label_encoder, scaler, SVM_MODEL_FILENAME, "SVM")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    print("\nTraining SVM selesai! Model siap digunakan.")
    print(f"Untuk testing: python test_recog.py (ubah USE_MODEL = 'SVM')")

if __name__ == "__main__":
    main()