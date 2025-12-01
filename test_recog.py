import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# --- KONFIGURASI ---
KNN_MODEL_FILE = "face_knn_robust_model.pkl"
SVM_MODEL_FILE = "face_svm_optimized_model.pkl"
CONFIDENCE_THRESHOLD = 0.7

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================================================
# 🔧 COMPATIBILITY CLASSES (MUST MATCH TRAINER!)
# ============================================================================

class RobustFeatureProcessor:
    """Feature processor - ENHANCED untuk KNN"""
    
    def __init__(self, preprocessing_config=None):
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        self.constant_features_mask = None
        self.correlation_features_mask = None
        self.config = preprocessing_config or {}
    
    def transform(self, features):
        """Transform features - ROBUST ERROR HANDLING"""
        try:
            # 1. Remove constant features (BOOLEAN MASK)
            if self.constant_features_mask is not None:
                if isinstance(self.constant_features_mask, np.ndarray):
                    if self.constant_features_mask.dtype == bool:
                        if len(self.constant_features_mask) == features.shape[1]:
                            features = features[:, self.constant_features_mask]
                        else:
                            print(f"⚠️  Constant mask mismatch: {len(self.constant_features_mask)} vs {features.shape[1]}")
                    else:
                        # Convert index array to boolean
                        indices = self.constant_features_mask.astype(int)
                        if np.max(indices) < features.shape[1]:
                            features = features[:, indices]
            
            # 2. Remove correlated features (INDEX ARRAY)
            if self.correlation_features_mask is not None:
                if isinstance(self.correlation_features_mask, (list, np.ndarray)):
                    indices = np.array(self.correlation_features_mask, dtype=int)
                    if len(indices) > 0 and np.max(indices) < features.shape[1]:
                        features = features[:, indices]
            
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
            
        except Exception as e:
            print(f"⚠️  Processor transform error: {e}")
            return features

class RobustKNNWithUnknown:
    """KNN classifier dengan unknown detection - ENHANCED"""
    
    def __init__(self, knn_model, distance_threshold, enable_unknown=True):
        self.knn = knn_model
        self.distance_threshold = distance_threshold
        self.enable_unknown = enable_unknown
    
    def predict(self, X):
        """Predict dengan unknown detection"""
        predictions = self.knn.predict(X)
        
        if not self.enable_unknown:
            return predictions
        
        distances, _ = self.knn.kneighbors(X)
        avg_distances = np.mean(distances, axis=1)
        
        unknown_mask = avg_distances > self.distance_threshold
        predictions = predictions.astype(object)
        predictions[unknown_mask] = -1
        
        return predictions
    
    def predict_with_distance(self, X):
        """Predict dengan info distance"""
        predictions = self.knn.predict(X)
        distances, _ = self.knn.kneighbors(X)
        avg_distances = np.mean(distances, axis=1)
        
        if self.enable_unknown:
            unknown_mask = avg_distances > self.distance_threshold
            predictions = predictions.astype(object)
            predictions[unknown_mask] = -1
        
        return predictions, avg_distances
    
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Passthrough to base KNN"""
        return self.knn.kneighbors(X, n_neighbors, return_distance)

class RobustSVMWithUnknown:
    """SVM classifier dengan unknown detection - ENHANCED"""
    
    def __init__(self, svm_model, decision_threshold, enable_unknown=True):
        self.svm = svm_model
        self.decision_threshold = decision_threshold
        self.enable_unknown = enable_unknown
    
    def predict(self, X):
        """Predict dengan unknown detection"""
        predictions = self.svm.predict(X)
        
        if not self.enable_unknown:
            return predictions
        
        decision_scores = self.svm.decision_function(X)
        
        if decision_scores.ndim > 1:
            max_scores = np.max(decision_scores, axis=1)
        else:
            max_scores = decision_scores
        
        unknown_mask = max_scores < self.decision_threshold
        predictions = predictions.astype(object)
        predictions[unknown_mask] = -1
        
        return predictions
    
    def predict_with_confidence(self, X):
        """Predict dengan confidence score"""
        predictions = self.svm.predict(X)
        decision_scores = self.svm.decision_function(X)
        
        if decision_scores.ndim > 1:
            max_scores = np.max(decision_scores, axis=1)
        else:
            max_scores = decision_scores
        
        if self.enable_unknown:
            unknown_mask = max_scores < self.decision_threshold
            predictions = predictions.astype(object)
            predictions[unknown_mask] = -1
        
        # Sigmoid-like confidence
        confidence = 1 / (1 + np.exp(-max_scores))
        
        return predictions, confidence
    
    def predict_proba(self, X):
        """Compatibility method"""
        return self.svm.predict_proba(X)
    
    def decision_function(self, X):
        """Get decision scores"""
        return self.svm.decision_function(X)

# ============================================================================
# 🎯 FEATURE EXTRACTION (MUST MATCH data_collect.py!)
# ============================================================================

def extract_enhanced_features(landmarks):
    """
    ✅ CONSISTENT: 1442 features
    - 478 landmarks × 3 coordinates = 1434
    - 8 geometric features
    Total: 1442 features
    """
    features = []
    
    # 1. Raw coordinates (478 × 3 = 1434)
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    
    # 2. Geometric features (8 features)
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    chin = landmarks[152]
    forehead = landmarks[10]
    
    # Distances
    eye_distance = np.sqrt((right_eye.x - left_eye.x)**2 + (right_eye.y - left_eye.y)**2)
    face_width = np.sqrt((landmarks[234].x - landmarks[454].x)**2 + (landmarks[234].y - landmarks[454].y)**2)
    face_height = np.sqrt((forehead.x - chin.x)**2 + (forehead.y - chin.y)**2)
    mouth_width = np.sqrt((mouth_right.x - mouth_left.x)**2 + (mouth_right.y - mouth_left.y)**2)
    
    features.extend([eye_distance, face_width, face_height, mouth_width])
    
    # Ratios (3 features)
    features.append(eye_distance / face_width if face_width > 0 else 0)
    features.append(mouth_width / face_width if face_width > 0 else 0)
    features.append(face_width / face_height if face_height > 0 else 0)
    
    # Symmetry (1 feature)
    left_eye_nose = np.sqrt((left_eye.x - nose_tip.x)**2 + (left_eye.y - nose_tip.y)**2)
    right_eye_nose = np.sqrt((right_eye.x - nose_tip.x)**2 + (right_eye.y - nose_tip.y)**2)
    symmetry_ratio = left_eye_nose / right_eye_nose if right_eye_nose > 0 else 1.0
    features.append(symmetry_ratio)
    
    return np.array(features)

# ============================================================================
# 🔍 MODEL DETECTION & LOADING
# ============================================================================

def check_available_models():
    """Cek model yang tersedia"""
    models = {
        "KNN": KNN_MODEL_FILE,
        "SVM": SVM_MODEL_FILE
    }
    
    available = []
    for name, file in models.items():
        if os.path.exists(file):
            available.append(name)
    
    return available

def select_model():
    """User memilih model"""
    available = check_available_models()
    
    if not available:
        print("❌ ERROR: Tidak ada model tersedia!")
        print("\nJalankan salah satu:")
        print("  python KNN_Trainer.py")
        print("  python SVM_Trainer.py")
        return None
    
    print("\n" + "="*60)
    print("🔍 FACE RECOGNITION MODEL SELECTOR v3.0")
    print("="*60)
    print(f"Model tersedia: {', '.join(available)}\n")
    
    for i, model in enumerate(available, 1):
        model_file = KNN_MODEL_FILE if model == "KNN" else SVM_MODEL_FILE
        try:
            with open(model_file, 'rb') as f:
                data = pickle.load(f)
                version = data.get('version', 'Unknown')
                accuracy = data.get('best_params', {}).get('accuracy', 'N/A')
                print(f"  {i}. {model} (v{version})")
        except:
            print(f"  {i}. {model}")
    
    print()
    while True:
        try:
            choice = input(f"Pilih model (1-{len(available)}): ").strip()
            if not choice:
                continue
            
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                selected = available[idx]
                print(f"✓ Model terpilih: {selected}")
                return selected
            else:
                print(f"❌ Pilih 1-{len(available)}")
        except ValueError:
            print("❌ Input harus angka!")
        except KeyboardInterrupt:
            print("\n⚠️  Dibatalkan.")
            return None

def load_model(model_file):
    """Load model dengan ENHANCED compatibility"""
    try:
        print(f"\n📦 Loading: {model_file}")
        
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract components
        model = data['model']
        encoder = data['encoder']
        model_type = data.get('model_type', 'Unknown')
        version = data.get('version', 'Unknown')
        feature_processor = data.get('feature_processor', None)
        distance_threshold = data.get('distance_threshold', 0.5)
        best_params = data.get('best_params', {})
        unknown_enabled = data.get('unknown_detection_enabled', False)
        
        print(f"✓ Model: {model_type}")
        print(f"✓ Version: {version}")
        print(f"✓ Feature Processor: {type(feature_processor).__name__ if feature_processor else 'None'}")
        print(f"✓ Unknown Detection: {'ENABLED ✅' if unknown_enabled else 'DISABLED ❌'}")
        print(f"✓ Threshold: {distance_threshold:.4f}")
        print(f"✓ Classes: {list(encoder.classes_)}")
        
        if best_params:
            print(f"✓ Best Params: {best_params}")
        
        return {
            'model': model,
            'encoder': encoder,
            'type': model_type,
            'version': version,
            'processor': feature_processor,
            'threshold': distance_threshold,
            'params': best_params,
            'unknown': unknown_enabled
        }
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 🎯 ENHANCED PREDICTION FUNCTIONS
# ============================================================================

def predict_knn_enhanced(model_data, raw_features):
    """KNN prediction dengan ENHANCED features"""
    try:
        model = model_data['model']
        encoder = model_data['encoder']
        processor = model_data['processor']
        threshold = model_data['threshold']
        
        # Transform features
        if processor is not None:
            processed = processor.transform(raw_features)
        else:
            processed = raw_features
        
        # Predict with distance
        if hasattr(model, 'predict_with_distance'):
            pred, distances = model.predict_with_distance(processed)
            pred = pred[0]
            avg_dist = distances[0]
        else:
            # Fallback to base KNN
            if hasattr(model, 'knn'):
                distances, _ = model.knn.kneighbors(processed)
                avg_dist = np.mean(distances[0])
                pred = model.knn.predict(processed)[0]
            else:
                distances, _ = model.kneighbors(processed)
                avg_dist = np.mean(distances[0])
                pred = model.predict(processed)[0]
        
        # Handle unknown
        if pred == -1 or (model_data['unknown'] and avg_dist > threshold):
            return "Unknown", 0.0, avg_dist
        
        # Get name
        name = encoder.inverse_transform([int(pred)])[0]
        
        # Calculate confidence (inverse of distance, normalized)
        confidence = max(0.1, min(1.0, 1 - (avg_dist / (threshold * 2))))
        
        return name, confidence, avg_dist
        
    except Exception as e:
        print(f"❌ KNN prediction error: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 0.0, 0.0

def predict_svm_enhanced(model_data, raw_features):
    """SVM prediction dengan ENHANCED features"""
    try:
        model = model_data['model']
        encoder = model_data['encoder']
        processor = model_data['processor']
        threshold = model_data['threshold']
        
        # Transform features (SVM uses RobustScaler)
        if processor is not None:
            processed = processor.transform(raw_features)
        else:
            processed = raw_features
        
        # Predict with confidence
        if hasattr(model, 'predict_with_confidence'):
            pred, confidence = model.predict_with_confidence(processed)
            pred = pred[0]
            confidence = confidence[0]
        else:
            # Fallback to base SVM
            if hasattr(model, 'svm'):
                probs = model.svm.predict_proba(processed)[0]
                pred = model.svm.predict(processed)[0]
                decision = model.svm.decision_function(processed)[0]
            else:
                probs = model.predict_proba(processed)[0]
                pred = model.predict(processed)[0]
                decision = model.decision_function(processed)[0]
            
            # Get max decision score
            if isinstance(decision, np.ndarray):
                max_decision = np.max(decision)
            else:
                max_decision = decision
            
            # Check unknown
            if model_data['unknown'] and max_decision < threshold:
                return "Unknown", 0.0, max_decision
            
            confidence = np.max(probs)
        
        # Handle unknown
        if pred == -1:
            decision = model.decision_function(processed)[0]
            max_decision = np.max(decision) if isinstance(decision, np.ndarray) else decision
            return "Unknown", 0.0, max_decision
        
        # Get name
        name = encoder.inverse_transform([int(pred)])[0]
        
        # Get decision score
        decision = model.decision_function(processed)[0] if hasattr(model, 'decision_function') else 0.0
        max_decision = np.max(decision) if isinstance(decision, np.ndarray) else decision
        
        return name, confidence, max_decision
        
    except Exception as e:
        print(f"❌ SVM prediction error: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 0.0, 0.0

def predict_face_universal(model_data, raw_features):
    """Universal prediction wrapper"""
    model_type = model_data['type']
    
    if 'KNN' in model_type:
        return predict_knn_enhanced(model_data, raw_features)
    elif 'SVM' in model_type:
        return predict_svm_enhanced(model_data, raw_features)
    else:
        return "Unsupported Model", 0.0, 0.0

# ============================================================================
# 🎥 RECOGNITION LOOP
# ============================================================================

def run_recognition(model_name, model_file):
    """Main recognition loop dengan ENHANCED display"""
    
    model_data = load_model(model_file)
    if model_data is None:
        return False
    
    print("\n" + "="*60)
    print("📹 KONTROL:")
    print("  [Q] Ganti model | [S] Screenshot | [R] Show stats | [ESC] Keluar")
    print("="*60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Tidak bisa buka kamera!")
        return False
    
    # Stats tracking
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0
    prediction_times = []
    show_stats = False
    
    # Recognition history (last 10 predictions)
    history = []
    max_history = 10
    
    print("\n🎥 Kamera aktif...\n")
    restart = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Process face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Extract features
            pred_start = time.time()
            raw_features = extract_enhanced_features(landmarks).reshape(1, -1)
            
            # Predict
            name, confidence, metric = predict_face_universal(model_data, raw_features)
            pred_time = (time.time() - pred_start) * 1000  # ms
            prediction_times.append(pred_time)
            if len(prediction_times) > 30:
                prediction_times.pop(0)
            
            # Update history
            history.append((name, confidence))
            if len(history) > max_history:
                history.pop(0)
            
            # Color coding
            if name == "Unknown":
                color = (0, 165, 255)  # Orange
                status = "⚠️  UNKNOWN"
            elif "Error" in name:
                color = (0, 0, 255)  # Red
                status = "❌ ERROR"
            else:
                color = (0, 255, 0)  # Green
                status = "✅ RECOGNIZED"
            
            # Main display box
            cv2.rectangle(frame, (10, 10), (550, 180), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (550, 180), color, 3)
            
            cv2.putText(frame, f"Name: {name}", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show metric based on model type
            if 'KNN' in model_data['type']:
                cv2.putText(frame, f"Distance: {metric:.4f}", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            else:
                cv2.putText(frame, f"Decision: {metric:.4f}", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            cv2.putText(frame, f"Model: {model_data['type']} v{model_data['version']}", 
                       (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Confidence bar
            bar_width = int(confidence * 520)
            cv2.rectangle(frame, (15, 190), (545, 220), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, 190), (15 + bar_width, 220), color, -1)
            cv2.putText(frame, f"{confidence*100:.1f}%", (250, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        else:
            cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 255), 2)
            cv2.putText(frame, "No face detected", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # FPS counter
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start = time.time()
        
        cv2.rectangle(frame, (w - 150, 10), (w - 10, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps_display}", (w - 140, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show stats panel (optional)
        if show_stats and len(prediction_times) > 0:
            cv2.rectangle(frame, (w - 300, 80), (w - 10, 280), (0, 0, 0), -1)
            cv2.rectangle(frame, (w - 300, 80), (w - 10, 280), (100, 100, 100), 2)
            
            cv2.putText(frame, "📊 STATISTICS", (w - 290, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            avg_pred = np.mean(prediction_times)
            cv2.putText(frame, f"Pred Time: {avg_pred:.1f}ms", (w - 290, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            if len(history) > 0:
                recent_conf = np.mean([c for _, c in history[-5:]])
                cv2.putText(frame, f"Avg Conf: {recent_conf:.3f}", (w - 290, 165), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                unique_names = len(set([n for n, _ in history if n != "Unknown"]))
                cv2.putText(frame, f"Faces: {unique_names}", (w - 290, 195), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Threshold: {model_data['threshold']:.4f}", 
                       (w - 290, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Unknown: {'ON' if model_data['unknown'] else 'OFF'}", 
                       (w - 290, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "[Q] Change | [S] Screenshot | [R] Stats | [ESC] Exit", 
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(f"Face Recognition v3.0 - {model_data['type']}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n🔄 Restart untuk ganti model...")
            restart = True
            break
        elif key == ord('s') or key == ord('S'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{model_name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        elif key == ord('r') or key == ord('R'):
            show_stats = not show_stats
            print(f"📊 Stats: {'ON' if show_stats else 'OFF'}")
        elif key == 27:  # ESC
            print("\n⚠️  Keluar...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    if len(prediction_times) > 0:
        print(f"\n📊 SESSION STATISTICS:")
        print(f"  • Avg Prediction Time: {np.mean(prediction_times):.2f}ms")
        print(f"  • Total Predictions: {len(history)}")
        if len(history) > 0:
            recognized = sum(1 for n, _ in history if n not in ["Unknown", "Error"])
            print(f"  • Recognized: {recognized}/{len(history)} ({recognized/len(history)*100:.1f}%)")
    
    return restart

# ============================================================================
# 🚀 MAIN FUNCTION
# ============================================================================

def main():
    """Main loop"""
    print("\n" + "="*60)
    print(" 🎯 FACE RECOGNITION v3.0 - ENHANCED KNN/SVM")
    print("="*60)
    print("\n✨ Features:")
    print("  • 1442-dimensional face features")
    print("  • Unknown face detection")
    print("  • Real-time confidence display")
    print("  • Performance statistics")
    print("  • Model switching")
    
    while True:
        model_name = select_model()
        if model_name is None:
            break
        
        model_file = KNN_MODEL_FILE if model_name == "KNN" else SVM_MODEL_FILE
        
        print(f"\n🚀 Starting {model_name}...")
        restart = run_recognition(model_name, model_file)
        
        if not restart:
            break
    
    print("\n✅ Selesai!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Dibatalkan.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()