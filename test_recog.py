import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURASI ---
KNN_MODEL_FILE = "face_knn_optimized_model.pkl"
SVM_MODEL_FILE = "face_svm_model.pkl"
ONECLASS_MODEL_FILE = "oneclass_svm_model.pkl"
CONFIDENCE_THRESHOLD = 0.7
DISTANCE_THRESHOLD = 0.5
ONECLASS_THRESHOLD = 0.0

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

class FeatureProcessor:
    """Class untuk menangani preprocessing fitur yang sama dengan training"""
    
    def __init__(self, preprocessing_config=None):
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        self.constant_features_mask = None
        self.correlation_features_mask = None
        
        # Default config jika tidak ada
        if preprocessing_config is None:
            self.config = {
                'use_pca': True,
                'pca_components': 0.95,
                'use_feature_selection': True,
                'feature_selection_k': 500,
                'use_robust_scaling': True
            }
        else:
            self.config = preprocessing_config
    
    def fit_transform(self, features, labels=None):
        """Fit transformer dan transform features (untuk training)"""
        print("Fitting feature processor...")
        
        # 1. Remove constant features
        feature_vars = np.var(features, axis=0)
        self.constant_features_mask = feature_vars >= 1e-8
        features = features[:, self.constant_features_mask]
        print(f"After removing constant features: {features.shape}")
        
        # 2. Remove highly correlated features
        corr_matrix = np.corrcoef(features.T)
        highly_corr_pairs = np.where((np.abs(corr_matrix) > 0.95) & 
                                   (np.abs(corr_matrix) < 1.0))
        
        features_to_remove = set()
        for i, j in zip(highly_corr_pairs[0], highly_corr_pairs[1]):
            if i < j:
                features_to_remove.add(j)
        
        remaining_features = [i for i in range(features.shape[1]) if i not in features_to_remove]
        self.correlation_features_mask = remaining_features
        features = features[:, remaining_features]
        print(f"After removing correlated features: {features.shape}")
        
        # 3. Scaling
        if self.config['use_robust_scaling']:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        features = self.scaler.fit_transform(features)
        
        # 4. Feature Selection
        if self.config['use_feature_selection'] and labels is not None:
            k_features = min(self.config['feature_selection_k'], features.shape[1])
            self.feature_selector = SelectKBest(score_func=f_classif, k=k_features)
            features = self.feature_selector.fit_transform(features, labels)
            print(f"After feature selection: {features.shape}")
        
        # 5. PCA
        if self.config['use_pca']:
            self.pca = PCA(n_components=self.config['pca_components'], random_state=42)
            features = self.pca.fit_transform(features)
            print(f"After PCA: {features.shape}")
        
        return features
    
    def transform(self, features):
        """Transform features menggunakan fitted transformers"""
        # 1. Remove constant features
        if self.constant_features_mask is not None:
            features = features[:, self.constant_features_mask]
        
        # 2. Remove correlated features
        if self.correlation_features_mask is not None:
            features = features[:, self.correlation_features_mask]
        
        # 3. Scaling
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # 4. Feature Selection
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)
        
        # 5. PCA
        if self.pca is not None:
            features = self.pca.transform(features)
        
        return features

def check_available_models():
    """Cek model mana saja yang tersedia"""
    models = {
        "KNN": KNN_MODEL_FILE,
        "SVM": SVM_MODEL_FILE, 
        "OneClassSVM": ONECLASS_MODEL_FILE
    }
    
    available = []
    for name, file in models.items():
        if os.path.exists(file):
            available.append(name)
    
    return available

def select_model():
    """Meminta user memilih model yang akan digunakan"""
    available_models = check_available_models()
    
    if not available_models:
        print("ERROR: Tidak ada model yang tersedia!")
        print("\nJalankan training script terlebih dahulu:")
        print("- Untuk KNN: python KNN_Trainer.py")
        print("- Untuk SVM: python SVM_Trainer.py")
        return None
    
    print("=== FACE RECOGNITION MODEL SELECTOR ===")
    print(f"Model yang tersedia: {available_models}")
    print()
    
    # Tampilkan pilihan
    for i, model in enumerate(available_models, 1):
        model_file = ""
        if model == "KNN":
            model_file = KNN_MODEL_FILE
        elif model == "SVM":
            model_file = SVM_MODEL_FILE
        elif model == "OneClassSVM":
            model_file = ONECLASS_MODEL_FILE
        
        print(f"{i}. {model} ({model_file})")
    
    while True:
        try:
            choice = input(f"\nPilih model (1-{len(available_models)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx]
                print(f"Model terpilih: {selected_model}")
                return selected_model
            else:
                print(f"Pilihan tidak valid. Masukkan angka 1-{len(available_models)}")
        except ValueError:
            print("Input harus berupa angka!")
        except KeyboardInterrupt:
            print("\nProgram dibatalkan.")
            return None

def load_optimized_model(model_file):
    """Load optimized model beserta semua komponen preprocessing"""
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        model = data['model']
        label_encoder = data['encoder']
        model_type = data['model_type']
        
        # Load preprocessing components jika ada
        preprocessing_config = data.get('preprocessing_config', None)
        distance_threshold = data.get('distance_threshold', DISTANCE_THRESHOLD)
        best_params = data.get('best_params', {'n_neighbors': model.n_neighbors if hasattr(model, 'n_neighbors') else 5})
        
        # LOAD FEATURE PROCESSOR YANG SUDAH DI-FIT!
        feature_processor = data.get('feature_processor', None)
        
        return model, label_encoder, preprocessing_config, model_type, distance_threshold, best_params, feature_processor
        
    except FileNotFoundError:
        print(f"Error: File '{model_file}' tidak ditemukan!")
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None, None, None

def load_legacy_model(model_file):
    """Load model lama (tanpa preprocessing pipeline)"""
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        model = data['model']
        label_encoder = data['encoder']
        scaler = data.get('scaler', None)
        model_type = data['model_type']
        
        return model, label_encoder, scaler, model_type
        
    except FileNotFoundError:
        print(f"Error: File '{model_file}' tidak ditemukan!")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

def predict_optimized_knn(model, feature_vector, label_encoder, distance_threshold, best_params, feature_processor):
    """Prediksi menggunakan Optimized KNN dengan preprocessing yang benar"""
    try:
        if feature_processor is None:
            return "No Processor", 0.0
        
        # Transform fitur menggunakan pipeline yang sama dengan training
        processed_features = feature_processor.transform(feature_vector)
        
        # Get distances to neighbors
        k_neighbors = best_params['n_neighbors']
        distances, indices = model.kneighbors(processed_features, n_neighbors=k_neighbors)
        
        # Prediksi
        predicted_id = model.predict(processed_features)[0]
        predicted_name = label_encoder.inverse_transform([predicted_id])[0]
        
        # Calculate confidence
        avg_distance = np.mean(distances[0])
        
        if avg_distance > distance_threshold:
            return "Unknown", 0.0
        
        # Improved confidence calculation
        distance_confidence = max(0, 1 - (avg_distance / distance_threshold))
        
        # Apply non-linear scaling for better confidence distribution
        final_confidence = np.power(distance_confidence, 0.5)
        final_confidence = max(0.1, min(1.0, final_confidence))
        
        return predicted_name, final_confidence
        
    except Exception as e:
        print(f"OptimizedKNN Error: {e}")
        return "Error", 0.0

def predict_legacy_knn(model, feature_vector, label_encoder, distance_threshold):
    """Prediksi menggunakan KNN lama (tanpa preprocessing)"""
    try:
        # Prediksi langsung tanpa preprocessing tambahan
        distances, indices = model.kneighbors(feature_vector, n_neighbors=model.n_neighbors)
        predicted_id = model.predict(feature_vector)[0]
        predicted_name = label_encoder.inverse_transform([predicted_id])[0]
        
        avg_distance = np.mean(distances[0])
        
        if avg_distance > distance_threshold:
            return "Unknown", 0.0
        
        confidence = max(0, 1 - (avg_distance / distance_threshold))
        return predicted_name, confidence
        
    except Exception as e:
        print(f"KNN Error: {e}")
        return "Error", 0.0

def predict_svm(model, feature_vector, label_encoder, scaler):
    """Prediksi menggunakan SVM dengan probabilitas"""
    try:
        feature_scaled = scaler.transform(feature_vector)
        
        probabilities = model.predict_proba(feature_scaled)[0]
        max_prob = np.max(probabilities)
        
        if max_prob > CONFIDENCE_THRESHOLD:
            predicted_id = model.predict(feature_scaled)[0]
            predicted_name = label_encoder.inverse_transform([predicted_id])[0]
            return predicted_name, max_prob
        else:
            return "Unknown", max_prob
    except Exception as e:
        return f"SVM Error", 0.0

def main():
    # User memilih model
    USE_MODEL = select_model()
    if USE_MODEL is None:
        return
    
    print(f"\n=== FACE RECOGNITION REAL-TIME ({USE_MODEL}) ===")
    
    # Tentukan file model berdasarkan pilihan
    if USE_MODEL == "KNN":
        model_file = KNN_MODEL_FILE
    elif USE_MODEL == "SVM":
        model_file = SVM_MODEL_FILE
    elif USE_MODEL == "OneClassSVM":
        model_file = ONECLASS_MODEL_FILE
    else:
        print(f"Error: Model '{USE_MODEL}' tidak dikenali!")
        return
    
    # Load model
    feature_processor = None
    
    if USE_MODEL == "KNN":
        # Load optimized model dengan feature processor
        model, label_encoder, preprocessing_config, model_type, distance_threshold, best_params, feature_processor = load_optimized_model(model_file)
        
        if model is not None:
            if feature_processor is not None:
                print(f"✓ Model Optimized {model_type} berhasil dimuat!")
                print("✓ Feature processor yang sudah di-fit berhasil dimuat!")
            else:
                print(f"✓ Model Optimized {model_type} berhasil dimuat!")
                print("⚠️  Feature processor tidak tersedia - menggunakan fallback legacy mode")
                # Fallback ke model lama
                model, label_encoder, scaler, model_type = load_legacy_model(model_file)
                if model is None:
                    print("Gagal memuat model!")
                    return
                print(f"✓ Model Legacy {model_type} berhasil dimuat!")
                distance_threshold = DISTANCE_THRESHOLD
                best_params = {'n_neighbors': model.n_neighbors}
        else:
            # Fallback ke model lama
            print("Mencoba load model KNN lama...")
            model, label_encoder, scaler, model_type = load_legacy_model(model_file)
            if model is None:
                print("Gagal memuat model!")
                return
            print(f"✓ Model Legacy {model_type} berhasil dimuat!")
            distance_threshold = DISTANCE_THRESHOLD
            best_params = {'n_neighbors': model.n_neighbors}
            
    else:
        # SVM atau OneClass
        model, label_encoder, scaler, model_type = load_legacy_model(model_file)
        if model is None:
            print("Gagal memuat model!")
            return
        print(f"✓ Model {model_type} berhasil dimuat!")
    
    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
        print(f"✓ Dapat mengenali: {list(label_encoder.classes_)}")
    else:
        print("✓ Dapat mengenali: [Label encoder tidak tersedia]")
    print(f"✓ Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    print("\nKontrol:")
    print("- 'q' untuk keluar")
    print("- 's' untuk screenshot")
    print("- 'c' untuk ganti confidence threshold")
    print("- 'm' untuk info model")
    print("- 'r' untuk restart dan pilih model lain")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera!")
        return
    
    # Variabel untuk FPS dan display
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0
    
    print("\nMembuka kamera...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera!")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Deteksi wajah
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Ekstraksi fitur (1434 features)
            feature_vector = []
            for lm in landmarks:
                feature_vector.extend([lm.x, lm.y, lm.z])
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Prediksi berdasarkan tipe model
            try:
                if USE_MODEL == "KNN":
                    if feature_processor is not None:
                        # Optimized KNN dengan feature processor yang sudah di-fit
                        predicted_name, confidence = predict_optimized_knn(
                            model, feature_vector, label_encoder, distance_threshold, 
                            best_params, feature_processor
                        )
                    else:
                        # Legacy KNN
                        predicted_name, confidence = predict_legacy_knn(
                            model, feature_vector, label_encoder, distance_threshold
                        )
                elif USE_MODEL == "SVM":
                    predicted_name, confidence = predict_svm(model, feature_vector, label_encoder, scaler)
                else:
                    predicted_name, confidence = "Error", 0.0
                
                # Warna teks berdasarkan prediksi
                color = (0, 255, 0) if predicted_name != "Unknown" and "Error" not in predicted_name else (0, 0, 255)
                
                # Tampilkan hasil dengan background
                cv2.rectangle(frame, (10, 10), (450, 130), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (450, 130), color, 2)
                
                cv2.putText(frame, f"Name: {predicted_name}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Model: {USE_MODEL}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
                cv2.putText(frame, f"Prediction Error", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
            cv2.putText(frame, "No face detected", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # FPS calculation dan display
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        cv2.putText(frame, f"FPS: {fps_display}", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instruksi kontrol
        cv2.putText(frame, "q=quit, s=screenshot, c=confidence, m=info, r=restart", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow(f"Face Recognition - {USE_MODEL}", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{USE_MODEL}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('r'):
            print("\nRestarting...")
            cap.release()
            cv2.destroyAllWindows()
            main()  # Restart program
            return
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh user.")
    except Exception as e:
        print(f"Error: {e}")