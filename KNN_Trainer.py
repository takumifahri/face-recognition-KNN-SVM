import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import cv2
import mediapipe as mp
warnings.filterwarnings('ignore')

# --- KONFIGURASI ROBUST ---
DATA_ROOT = "data_mp_wajah"
MODEL_FILENAME = "face_knn_robust_model.pkl"
TEST_SIZE = 0.3
RANDOM_STATE = 42

# 🎯 KAGGLE SETTINGS (SAMA SEPERTI SVM)
KAGGLE_SUBDIR = "kaggle/Faces/Faces"
KAGGLE_MAX_SAMPLES_PER_CLASS = 200

# TUNING UNTUK ROBUSTNESS
USE_PCA = True
PCA_COMPONENTS = 0.99
USE_FEATURE_SELECTION = True
FEATURE_SELECTION_K = 'all'
USE_ROBUST_SCALING = True

# ✨ DATA AUGMENTATION untuk variasi pose
USE_DATA_AUGMENTATION = True
AUGMENTATION_NOISE_LEVEL = 0.02

# 🔒 UNKNOWN DETECTION
ENABLE_UNKNOWN_DETECTION = True
UNKNOWN_THRESHOLD_PERCENTILE = 85

# 📁 OUTPUT DIRECTORIES
OUTPUT_DIR = "knn_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🎯 MEDIAPIPE SETUP (SAMA SEPERTI SVM)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def extract_enhanced_features_from_image(image_path):
    """Extract 1442-dim features from image using MediaPipe"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        
        # 1. Landmark coordinates (478 * 3 = 1434)
        coords = []
        for landmark in landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        
        # 2. Geometric features (8 additional)
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        face_width = np.max(lm_array[:, 0]) - np.min(lm_array[:, 0])
        face_height = np.max(lm_array[:, 1]) - np.min(lm_array[:, 1])
        
        left_eye = lm_array[33]
        right_eye = lm_array[263]
        eye_distance = np.linalg.norm(left_eye - right_eye)
        
        nose_tip = lm_array[1]
        mouth_center = lm_array[13]
        nose_mouth_dist = np.linalg.norm(nose_tip - mouth_center)
        
        aspect_ratio = face_width / face_height if face_height > 0 else 0
        
        left_half = lm_array[:234]
        right_half = lm_array[234:468]
        symmetry = np.mean(np.abs(left_half[:, 0] - (1 - right_half[:, 0])))
        
        depth_var = np.var(lm_array[:, 2])
        
        centroid = np.mean(lm_array, axis=0)
        centroid_dist = np.mean([np.linalg.norm(lm - centroid) for lm in lm_array])
        
        geometric_features = [
            face_width, face_height, eye_distance, nose_mouth_dist,
            aspect_ratio, symmetry, depth_var, centroid_dist
        ]
        
        features = np.array(coords + geometric_features, dtype=np.float32)
        
        return features
        
    except Exception as e:
        return None

def load_kaggle_format_data(kaggle_dir, max_samples_per_class=200):
    """Load Kaggle format: Faces/Akshay Kumar_0.jpg"""
    print(f"\n📥 LOADING KAGGLE FORMAT DATA")
    print(f"   Path: {kaggle_dir}")
    print(f"   Max samples per class: {max_samples_per_class}")
    
    if not os.path.exists(kaggle_dir):
        print(f"   ❌ Directory not found: {kaggle_dir}")
        return [], [], {}
    
    features = []
    labels = []
    class_counts = {}
    skipped = 0
    
    image_files = [f for f in os.listdir(kaggle_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"   Found {len(image_files)} image files")
    
    person_files = {}
    for filename in image_files:
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            name_part = parts[0]
            person_name = name_part.replace('_', ' ')
        else:
            person_name = filename.split('.')[0]
        
        if person_name not in person_files:
            person_files[person_name] = []
        person_files[person_name].append(filename)
    
    print(f"   Detected {len(person_files)} unique persons")
    
    for person_name, file_list in sorted(person_files.items()):
        person_features = []
        
        files_to_process = file_list[:max_samples_per_class]
        
        for filename in files_to_process:
            img_path = os.path.join(kaggle_dir, filename)
            feature = extract_enhanced_features_from_image(img_path)
            
            if feature is not None:
                person_features.append(feature)
            else:
                skipped += 1
        
        if len(person_features) > 0:
            features.extend(person_features)
            labels.extend([person_name] * len(person_features))
            class_counts[person_name] = len(person_features)
            print(f"   ✅ {person_name}: {len(person_features)}/{len(files_to_process)} samples")
    
    print(f"\n   Summary:")
    print(f"   • Processed: {len(features)} samples")
    print(f"   • Skipped: {skipped} samples")
    print(f"   • Classes: {len(class_counts)}")
    
    return features, labels, class_counts

def load_local_format_data(data_root):
    """Load local format: Fahri_Radiansyah/image.npy"""
    print(f"\n📥 LOADING LOCAL FORMAT DATA")
    print(f"   Path: {data_root}")
    
    features = []
    labels = []
    class_counts = {}
    feature_lengths = []
    
    for person_name in os.listdir(data_root):
        person_path = os.path.join(data_root, person_name)
        
        if person_name.lower() == 'kaggle':
            continue
        
        if not os.path.isdir(person_path):
            continue
        
        person_features = []
        
        for filename in os.listdir(person_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(person_path, filename)
                try:
                    feature = np.load(file_path)
                    
                    if np.any(np.isnan(feature)) or np.any(np.isinf(feature)):
                        continue
                    
                    feature_lengths.append(len(feature))
                    person_features.append(feature)
                    
                except Exception as e:
                    print(f"   ⚠️  Error loading {file_path}: {e}")
        
        if len(person_features) > 0:
            features.extend(person_features)
            labels.extend([person_name] * len(person_features))
            class_counts[person_name] = len(person_features)
            print(f"   ✅ {person_name}: {len(person_features)} samples")
    
    if len(set(feature_lengths)) > 1:
        print(f"\n   ⚠️  Inconsistent feature lengths detected: {set(feature_lengths)}")
        most_common = max(set(feature_lengths), key=feature_lengths.count)
        print(f"   Filtering to most common length: {most_common}")
        
        filtered_features = []
        filtered_labels = []
        for feat, label in zip(features, labels):
            if len(feat) == most_common:
                filtered_features.append(feat)
                filtered_labels.append(label)
        
        features = filtered_features
        labels = filtered_labels
        print(f"   Filtered: {len(features)} samples")
    
    print(f"\n   Summary:")
    print(f"   • Total samples: {len(features)}")
    print(f"   • Classes: {len(class_counts)}")
    
    return features, labels, class_counts

def compare_and_merge_datasets(local_features, local_labels, local_counts,
                               kaggle_features, kaggle_labels, kaggle_counts):
    """Compare and merge datasets"""
    print("\n" + "="*70)
    print("📊 DATASET COMPARISON & MERGE")
    print("="*70)
    
    print("\n1️⃣  SIZE COMPARISON:")
    print(f"{'Metric':<30} {'Local (npy)':<20} {'Kaggle (jpg)':<20}")
    print("-" * 70)
    print(f"{'Total Samples':<30} {len(local_labels):<20} {len(kaggle_labels):<20}")
    print(f"{'Number of Classes':<30} {len(local_counts):<20} {len(kaggle_counts):<20}")
    
    if len(local_features) > 0 and len(kaggle_features) > 0:
        print(f"{'Feature Dimensions':<30} {local_features[0].shape[0]:<20} {kaggle_features[0].shape[0]:<20}")
    
    print("\n2️⃣  CLASS OVERLAP:")
    local_set = set(local_labels)
    kaggle_set = set(kaggle_labels)
    overlap = local_set.intersection(kaggle_set)
    
    print(f"   • Local-only classes: {len(local_set - kaggle_set)}")
    if len(local_set - kaggle_set) > 0:
        print(f"     {', '.join(sorted(list(local_set - kaggle_set))[:5])}")
    
    print(f"   • Kaggle-only classes: {len(kaggle_set - local_set)}")
    if len(kaggle_set - local_set) > 0:
        kaggle_only_list = sorted(list(kaggle_set - local_set))[:5]
        print(f"     {', '.join(kaggle_only_list)}{'...' if len(kaggle_set - local_set) > 5 else ''}")
    
    print(f"   • Shared classes: {len(overlap)}")
    if len(overlap) > 0:
        print(f"     {', '.join(sorted(list(overlap)))}")
    
    print("\n3️⃣  MERGE STRATEGY:")
    
    if len(local_features) == 0:
        print("   ✅ Using KAGGLE dataset only (no local data)")
        return kaggle_features, kaggle_labels, kaggle_counts
    
    if len(kaggle_features) == 0:
        print("   ✅ Using LOCAL dataset only (no Kaggle data)")
        return local_features, local_labels, local_counts
    
    if local_features[0].shape[0] != kaggle_features[0].shape[0]:
        print(f"   ❌ Feature dimension mismatch!")
        print(f"      Local: {local_features[0].shape[0]}, Kaggle: {kaggle_features[0].shape[0]}")
        print(f"   ⚠️  Using LOCAL dataset only (preprocessed)")
        return local_features, local_labels, local_counts
    
    print("   🔀 MERGING both datasets...")
    
    combined_features = local_features + kaggle_features
    combined_labels = local_labels + kaggle_labels
    
    combined_counts = {}
    for name, count in local_counts.items():
        combined_counts[name] = count
    for name, count in kaggle_counts.items():
        combined_counts[name] = combined_counts.get(name, 0) + count
    
    print(f"\n   ✅ COMBINED DATASET:")
    print(f"      • Total samples: {len(combined_labels)}")
    print(f"      • Total classes: {len(combined_counts)}")
    print(f"      • Local contribution: {len(local_labels)} ({len(local_labels)/len(combined_labels)*100:.1f}%)")
    print(f"      • Kaggle contribution: {len(kaggle_labels)} ({len(kaggle_labels)/len(combined_labels)*100:.1f}%)")
    
    return combined_features, combined_labels, combined_counts

def load_and_analyze_data():
    """Auto-detect and load both formats"""
    print("\n" + "="*70)
    print("📥 AUTO-LOADING DATA FROM MULTIPLE SOURCES")
    print("="*70)
    
    local_features, local_labels, local_counts = load_local_format_data(DATA_ROOT)
    
    kaggle_dir = os.path.join(DATA_ROOT, KAGGLE_SUBDIR)
    kaggle_features, kaggle_labels, kaggle_counts = load_kaggle_format_data(
        kaggle_dir, KAGGLE_MAX_SAMPLES_PER_CLASS
    )
    
    if len(local_features) > 0 or len(kaggle_features) > 0:
        final_features, final_labels, final_counts = compare_and_merge_datasets(
            local_features, local_labels, local_counts,
            kaggle_features, kaggle_labels, kaggle_counts
        )
        
        if len(final_features) > 0:
            return np.array(final_features), np.array(final_labels), final_counts
    
    print("\n❌ No valid data found!")
    return None, None, None

class RobustFeatureProcessor:
    """Feature processor yang lebih robust terhadap variasi pose"""
    
    def __init__(self, preprocessing_config=None):
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        self.constant_features_mask = None
        self.correlation_features_mask = None
        
        if preprocessing_config is None:
            self.config = {
                'use_pca': USE_PCA,
                'pca_components': PCA_COMPONENTS,
                'use_feature_selection': USE_FEATURE_SELECTION,
                'feature_selection_k': FEATURE_SELECTION_K,
                'use_robust_scaling': USE_ROBUST_SCALING,
                'use_data_augmentation': USE_DATA_AUGMENTATION,
                'augmentation_noise': AUGMENTATION_NOISE_LEVEL
            }
        else:
            self.config = preprocessing_config
    
    def augment_data(self, features, labels):
        """Data augmentation untuk meningkatkan robustness"""
        if not self.config.get('use_data_augmentation', False):
            return features, labels
        
        print("\n=== DATA AUGMENTATION FOR ROBUSTNESS ===")
        augmented_features = [features]
        augmented_labels = [labels]
        
        noise_level = self.config.get('augmentation_noise', 0.02)
        
        # Augmentation 1: Small random noise
        noise = np.random.normal(0, noise_level, features.shape)
        augmented_features.append(features + noise)
        augmented_labels.append(labels)
        
        # Augmentation 2: Slight scaling
        scale_factors = np.random.uniform(0.95, 1.05, (features.shape[0], 1))
        augmented_features.append(features * scale_factors)
        augmented_labels.append(labels)
        
        # Augmentation 3: Combined
        augmented_features.append((features + noise * 0.5) * scale_factors)
        augmented_labels.append(labels)
        
        all_features = np.vstack(augmented_features)
        all_labels = np.concatenate(augmented_labels)
        
        print(f"Original samples: {features.shape[0]}")
        print(f"Augmented samples: {all_features.shape[0]}")
        print(f"Augmentation ratio: {all_features.shape[0] / features.shape[0]:.1f}x")
        
        return all_features, all_labels
    
    def fit_transform(self, features, labels=None):
        """Fit dan transform dengan robust preprocessing"""
        print("\n=== ROBUST PREPROCESSING WITH FEATURE PROCESSOR ===")
        print(f"Original features shape: {features.shape}")
        
        if labels is not None and self.config.get('use_data_augmentation', False):
            features, labels = self.augment_data(features, labels)
        
        # 1. Remove constant features
        feature_vars = np.var(features, axis=0)
        self.constant_features_mask = feature_vars >= 1e-8
        features = features[:, self.constant_features_mask]
        print(f"After removing constant features: {features.shape}")
        
        # 2. Remove correlated features
        corr_matrix = np.corrcoef(features.T)
        highly_corr_pairs = np.where((np.abs(corr_matrix) > 0.98) & 
                                   (np.abs(corr_matrix) < 1.0))
        
        features_to_remove = set()
        for i, j in zip(highly_corr_pairs[0], highly_corr_pairs[1]):
            if i < j:
                features_to_remove.add(j)
        
        remaining_features = [i for i in range(features.shape[1]) if i not in features_to_remove]
        self.correlation_features_mask = np.array(remaining_features)
        features = features[:, remaining_features]
        print(f"After removing correlated features: {features.shape}")
        
        # 3. Robust Scaling
        self.scaler = RobustScaler(quantile_range=(10, 90))
        features = self.scaler.fit_transform(features)
        
        # 4. Feature Selection
        if self.config['use_feature_selection'] and labels is not None:
            if self.config['feature_selection_k'] == 'all':
                k_features = features.shape[1]
            else:
                k_features = min(self.config['feature_selection_k'], features.shape[1])
            
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
            features = self.feature_selector.fit_transform(features, labels)
            print(f"After feature selection: {features.shape}")
        
        # 5. PCA
        if self.config['use_pca']:
            self.pca = PCA(n_components=self.config['pca_components'], random_state=RANDOM_STATE)
            features = self.pca.fit_transform(features)
            print(f"After PCA: {features.shape}")
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
            print(f"Number of components: {self.pca.n_components_}")
        
        print(f"Final processed features shape: {features.shape}")
        return features, labels
    
    def transform(self, features):
        """Transform features untuk inference"""
        if self.constant_features_mask is not None:
            features = features[:, self.constant_features_mask]
        
        if self.correlation_features_mask is not None:
            features = features[:, self.correlation_features_mask]
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)
        
        if self.pca is not None:
            features = self.pca.transform(features)
        
        return features

class RobustKNNWithUnknown:
    """KNN classifier dengan unknown detection"""
    
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

def analyze_confusion_matrix_detailed(cm, class_names):
    """📊 ANALISIS CONFUSION MATRIX MENDALAM"""
    print("\n" + "="*70)
    print("📊 CONFUSION MATRIX - DETAILED ANALYSIS")
    print("="*70)
    
    n_classes = len(class_names)
    total_samples = np.sum(cm)
    
    print("\n1️⃣  OVERALL STATISTICS:")
    print(f"   Total Samples: {total_samples}")
    print(f"   Total Classes: {n_classes}")
    print(f"   Correctly Classified: {np.trace(cm)} ({np.trace(cm)/total_samples*100:.2f}%)")
    print(f"   Misclassified: {total_samples - np.trace(cm)} ({(total_samples - np.trace(cm))/total_samples*100:.2f}%)")
    
    print("\n2️⃣  PER-CLASS ANALYSIS:")
    print(f"{'Class':<20} {'True Pos':<10} {'False Pos':<10} {'False Neg':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 95)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total_samples - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1
        }
        
        print(f"{class_name:<20} {tp:<10} {fp:<10} {fn:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    return class_metrics

def interpret_knn_algorithm(best_knn, X_train, y_train, X_test, y_test, class_names, feature_processor):
    """🧠 INTERPRETASI ALGORITMA KNN"""
    print("\n" + "="*70)
    print("🧠 K-NEAREST NEIGHBORS (KNN) - ALGORITHM INTERPRETATION")
    print("="*70)
    
    k = best_knn.n_neighbors
    metric = best_knn.metric
    weights = best_knn.weights
    
    print("\n1️⃣  ALGORITHM OVERVIEW:")
    print(f"""
   KNN finds K={k} nearest neighbors using '{metric}' metric
   Predicts class based on majority vote (weighted by '{weights}')
   Effective in {X_train.shape[1]}-dimensional space
    """)
    
    print("\n2️⃣  MODEL PARAMETERS:")
    print(f"   K = {k}")
    print(f"   Distance Metric = '{metric}'")
    print(f"   Weight Function = '{weights}'")

# ...existing code...

def plot_separated_visualizations(best_knn, X_train, y_train, X_test, y_test, 
                                  label_encoder, best_params, adaptive_threshold, 
                                  feature_processor):
    """📊 CREATE ALL 12 SEPARATED VISUALIZATIONS"""
    print("\n=== CREATING 12 SEPARATED VISUALIZATIONS ===")
    
    y_pred = best_knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    distances_train, _ = best_knn.kneighbors(X_train)
    distances_test, _ = best_knn.kneighbors(X_test)
    avg_dist_train = np.mean(distances_train, axis=1)
    avg_dist_test = np.mean(distances_test, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    classes = label_encoder.classes_
    imbalance_ratio = cm.sum(axis=1).max() / cm.sum(axis=1).min() if cm.sum(axis=1).min() > 0 else float('inf')
    
    # 1. Confusion Matrix
    print("  [1/12] Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={'label': 'Count'})
    title_text = f'Confusion Matrix\nAcc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    for i in range(len(classes)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. K-Value Optimization
    print("  [2/12] K-Value Optimization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    k_values = [3, 5, 7, 9, 11, 15, 21, 25, 31]
    train_accs = []
    test_accs = []
    cv_accs = []
    
    optimal_params = best_params.copy()
    for k in k_values:
        if k > len(X_train) // 2:
            break
        optimal_params['n_neighbors'] = k
        knn_temp = KNeighborsClassifier(**optimal_params)
        knn_temp.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, knn_temp.predict(X_train)))
        test_accs.append(accuracy_score(y_test, knn_temp.predict(X_test)))
        cv_accs.append(cross_val_score(knn_temp, X_train, y_train, cv=5).mean())
    
    ax.plot(k_values[:len(train_accs)], train_accs, 'bo-', label='Train', linewidth=2, markersize=8)
    ax.plot(k_values[:len(test_accs)], test_accs, 'ro-', label='Test', linewidth=2, markersize=8)
    ax.plot(k_values[:len(cv_accs)], cv_accs, 'go-', label='CV', linewidth=2, markersize=8)
    ax.axvline(x=best_params['n_neighbors'], color='purple', linestyle='--', linewidth=2,
                label=f"Optimal K={best_params['n_neighbors']}")
    ax.set_title('K-Value Optimization (Bias-Variance Tradeoff)', fontsize=14, fontweight='bold')
    ax.set_xlabel('K (Number of Neighbors)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_k_value_optimization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distance Distribution
    print("  [3/12] Distance Distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(avg_dist_train, bins=40, alpha=0.6, label='Train', color='blue', edgecolor='black')
    ax.hist(avg_dist_test, bins=40, alpha=0.6, label='Test', color='red', edgecolor='black')
    ax.axvline(x=adaptive_threshold, color='green', linestyle='--', linewidth=3,
                label=f'Threshold={adaptive_threshold:.3f}')
    ax.axvline(x=np.mean(avg_dist_train), color='blue', linestyle='-', linewidth=2,
                label=f'Mean Train={np.mean(avg_dist_train):.3f}')
    ax.axvline(x=np.mean(avg_dist_test), color='red', linestyle='-', linewidth=2,
                label=f'Mean Test={np.mean(avg_dist_test):.3f}')
    if ENABLE_UNKNOWN_DETECTION:
        ax.axvspan(adaptive_threshold, ax.get_xlim()[1], alpha=0.3, color='orange', 
                   label='Unknown Detection Zone')
    ax.set_title('Distance Distribution (Prediction Confidence)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Distance to K Neighbors', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_distance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. PCA Variance Analysis
    print("  [4/12] PCA Variance Analysis...")
    fig, ax = plt.subplots(figsize=(10, 6))
    if feature_processor.pca is not None:
        explained_var = feature_processor.pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        n_components = len(explained_var)
        
        ax.bar(range(1, min(51, n_components+1)), explained_var[:50], 
                alpha=0.7, color='skyblue', edgecolor='black', label='Individual')
        ax_twin = ax.twinx()
        ax_twin.plot(range(1, min(51, n_components+1)), cumsum_var[:50], 
                     'ro-', linewidth=3, markersize=6, label='Cumulative')
        ax_twin.axhline(y=0.99, color='green', linestyle='--', linewidth=2, label='99% Target')
        
        ax.set_title(f'PCA Variance (Total: {n_components} components)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Individual Variance Ratio', fontsize=12)
        ax_twin.set_ylabel('Cumulative Variance', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax_twin.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_pca_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Per-Class Performance
    print("  [5/12] Per-Class Performance...")
    fig, ax = plt.subplots(figsize=(12, 6))
    report = classification_report(y_test, y_pred, 
                           labels=range(len(classes)),
                           target_names=classes, 
                           output_dict=True, zero_division=0)
    
    precision_vals = [report[c]['precision'] for c in classes]
    recall_vals = [report[c]['recall'] for c in classes]
    f1_vals = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_vals, width, label='Precision', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x, recall_vals, width, label='Recall', alpha=0.8, color='#e74c3c')
    bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8, color='#2ecc71')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.15])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_per_class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Prediction Confidence
    print("  [6/12] Prediction Confidence...")
    fig, ax = plt.subplots(figsize=(10, 6))
    correct_mask = y_pred == y_test
    correct_distances = avg_dist_test[correct_mask]
    incorrect_distances = avg_dist_test[~correct_mask] if np.sum(~correct_mask) > 0 else []
    
    box_data = [correct_distances]
    labels_box = ['Correct\nPredictions']
    colors = ['green']
    
    if len(incorrect_distances) > 0:
        box_data.append(incorrect_distances)
        labels_box.append('Incorrect\nPredictions')
        colors.append('red')
    
    bp = ax.boxplot(box_data, labels=labels_box, patch_artist=True, 
                     widths=0.6, showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(y=adaptive_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Threshold={adaptive_threshold:.3f}')
    ax.set_title('Prediction Confidence Analysis (Distance)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Distance to K Neighbors', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_prediction_confidence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Class Balance
    print("  [7/12] Class Balance...")
    fig, ax = plt.subplots(figsize=(10, 6))
    class_totals = cm.sum(axis=1)
    bars = ax.barh(classes, class_totals, color='steelblue', alpha=0.7, edgecolor='black')
    
    for i, (bar, count) in enumerate(zip(bars, class_totals)):
        percentage = (count / np.sum(class_totals)) * 100
        ax.text(count, bar.get_y() + bar.get_height()/2, 
                f' {count} ({percentage:.1f}%)', 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_title(f'Class Distribution\nImbalance Ratio: {imbalance_ratio:.2f}x', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_class_balance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Learning Curve
    print("  [8/12] Learning Curve...")
    fig, ax = plt.subplots(figsize=(10, 6))
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []
    
    for size in train_sizes:
        n_samples = int(len(X_train) * size)
        if n_samples < best_params['n_neighbors']:
            continue
        knn_temp = KNeighborsClassifier(**best_params)
        knn_temp.fit(X_train[:n_samples], y_train[:n_samples])
        train_scores.append(accuracy_score(y_train[:n_samples], knn_temp.predict(X_train[:n_samples])))
        test_scores.append(accuracy_score(y_test, knn_temp.predict(X_test)))
    
    valid_sizes = train_sizes[:len(train_scores)]
    ax.plot([int(s * len(X_train)) for s in valid_sizes], train_scores, 'bo-', label='Train', linewidth=2, markersize=8)
    ax.plot([int(s * len(X_train)) for s in valid_sizes], test_scores, 'ro-', label='Test', linewidth=2, markersize=8)
    ax.fill_between([int(s * len(X_train)) for s in valid_sizes], train_scores, test_scores, alpha=0.2, color='gray')
    ax.set_title('Learning Curve (Model Scalability)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Feature Importance (PCA)
    print("  [9/12] Feature Importance (PCA)...")
    fig, ax = plt.subplots(figsize=(10, 6))
    if feature_processor.pca is not None:
        top_n = 20
        components = feature_processor.pca.components_[:top_n]
        importance = np.sum(np.abs(components), axis=1)
        
        bars = ax.barh(range(top_n), importance, color='coral', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f'PC{i+1}' for i in range(top_n)], fontsize=10)
        ax.set_title(f'Top {top_n} Principal Components (Feature Importance)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Absolute Weight Sum', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Misclassification Matrix
    print("  [10/12] Misclassification Matrix...")
    fig, ax = plt.subplots(figsize=(14, 12))
    misclass_matrix = cm.copy()
    np.fill_diagonal(misclass_matrix, 0)
    sns.heatmap(misclass_matrix, annot=True, fmt='d', cmap='Reds', 
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={'label': 'Misclassifications'})
    ax.set_title('Misclassification Heatmap (Off-diagonal Confusion)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted As', fontsize=12)
    ax.set_ylabel('Actually Is', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '10_misclassification_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 11. Normalized Confusion Matrix
    print("  [11/12] Normalized Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(14, 12))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
    ax.set_title('Normalized Confusion Matrix (Row Percentages)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '11_normalized_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 12. Model Summary (TEXT FILE)
    print("  [12/12] Model Summary (Text File)...")
    cv_score = cross_val_score(best_knn, X_train, y_train, cv=5).mean()
    cv_std = cross_val_score(best_knn, X_train, y_train, cv=5).std()
    variance_explained = feature_processor.pca.explained_variance_ratio_.sum() if feature_processor.pca else 0.0
    unknown_status = "ENABLED ✅" if ENABLE_UNKNOWN_DETECTION else "DISABLED ❌"
    
    summary_text = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           ENHANCED KNN MODEL SUMMARY - DUAL SOURCE v3.0             ║
╚══════════════════════════════════════════════════════════════════════╝

📊 DATA SOURCES:
═══════════════════════════════════════════════════════════════════════
  • Local format: {DATA_ROOT}/<person_name>/*.npy
  • Kaggle format: {DATA_ROOT}/{KAGGLE_SUBDIR}/<Name_Number>.jpg
  • Auto-detection: ✅ ENABLED
  • Auto-merge: ✅ ENABLED

📈 DATASET INFORMATION:
═══════════════════════════════════════════════════════════════════════
  • Training Samples: {len(X_train)} (70%)
  • Test Samples: {len(X_test)} (30%)
  • Feature Dimensions: {X_train.shape[1]}
  • Number of Classes: {len(classes)}
  • Classes: {', '.join(classes[:10])}{'...' if len(classes) > 10 else ''}
  • Class Balance Ratio: {imbalance_ratio:.2f}x

🎯 PERFORMANCE METRICS:
═══════════════════════════════════════════════════════════════════════
  • Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
  • Cross-Validation (5-fold): {cv_score:.4f} ± {cv_std:.4f}
  • Macro Precision: {precision:.4f}
  • Macro Recall: {recall:.4f}
  • Macro F1-Score: {f1:.4f}

🔧 MODEL CONFIGURATION:
═══════════════════════════════════════════════════════════════════════
  • Algorithm: K-Nearest Neighbors (KNN)
  • K (Neighbors): {best_params['n_neighbors']}
  • Weight Function: {best_params['weights']}
  • Distance Metric: {best_params['metric']}

📏 DISTANCE STATISTICS:
═══════════════════════════════════════════════════════════════════════
  • Train Mean Distance: {np.mean(avg_dist_train):.4f}
  • Test Mean Distance: {np.mean(avg_dist_test):.4f}
  • Train Std Dev: {np.std(avg_dist_train):.4f}
  • Test Std Dev: {np.std(avg_dist_test):.4f}
  • Distance Ratio (Test/Train): {np.mean(avg_dist_test)/np.mean(avg_dist_train):.2f}x

🔧 PREPROCESSING PIPELINE:
═══════════════════════════════════════════════════════════════════════
  • Original Features: 1442 (face landmarks + geometry)
  • PCA Components: {X_train.shape[1]}
  • Variance Explained: {variance_explained:.4f} ({variance_explained*100:.2f}%)
  • Dimensionality Reduction: {1442/X_train.shape[1]:.1f}x
  • Scaler: RobustScaler (10-90 percentile)
  • Data Augmentation: {'YES (4x)' if USE_DATA_AUGMENTATION else 'NO'}

🔒 UNKNOWN DETECTION:
═══════════════════════════════════════════════════════════════════════
  • Status: {unknown_status}
  • Threshold: {adaptive_threshold:.4f}
  • Percentile: {UNKNOWN_THRESHOLD_PERCENTILE}th
  • Method: Distance-based threshold

📂 OUTPUT FILES:
═══════════════════════════════════════════════════════════════════════
  01. Confusion Matrix
  02. K-Value Optimization
  03. Distance Distribution
  04. PCA Variance Analysis
  05. Per-Class Performance
  06. Prediction Confidence
  07. Class Balance Analysis
  08. Learning Curve
  09. Feature Importance (PCA)
  10. Misclassification Matrix
  11. Normalized Confusion Matrix
  12. Model Summary (This File)

✅ MODEL READY FOR DEPLOYMENT!
═══════════════════════════════════════════════════════════════════════
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
    """
    
    with open(os.path.join(OUTPUT_DIR, '12_model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\n✅ All 12 visualizations saved to: {OUTPUT_DIR}/")
    print("   01_confusion_matrix.png")
    print("   02_k_value_optimization.png")
    print("   03_distance_distribution.png")
    print("   04_pca_variance.png")
    print("   05_per_class_performance.png")
    print("   06_prediction_confidence.png")
    print("   07_class_balance.png")
    print("   08_learning_curve.png")
    print("   09_feature_importance.png")
    print("   10_misclassification_matrix.png")
    print("   11_normalized_confusion_matrix.png")
    print("   12_model_summary.txt")

# ...existing code...
def robust_knn_grid_search(X_train, y_train):
    """Grid search dengan parameter yang lebih robust"""
    print("\n=== ROBUST KNN GRID SEARCH ===")
    
    param_grid = {
        'n_neighbors': [5, 7, 9, 11, 15, 21],
        'weights': ['distance'],
        'metric': ['euclidean', 'manhattan'],
        'algorithm': ['auto'],
        'p': [1, 2]
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn, param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def calculate_adaptive_threshold(best_knn, X_train, y_train, X_test):
    """Calculate adaptive threshold untuk unknown detection"""
    print("\n=== CALCULATING ADAPTIVE THRESHOLD ===")
    
    distances_train, _ = best_knn.kneighbors(X_train)
    avg_distances_train = np.mean(distances_train, axis=1)
    
    distances_test, _ = best_knn.kneighbors(X_test)
    avg_distances_test = np.mean(distances_test, axis=1)
    
    threshold_percentile = UNKNOWN_THRESHOLD_PERCENTILE
    adaptive_threshold = np.percentile(
        np.concatenate([avg_distances_train, avg_distances_test]), 
        threshold_percentile
    )
    
    print(f"Train distance - Mean: {np.mean(avg_distances_train):.4f}, Std: {np.std(avg_distances_train):.4f}")
    print(f"Test distance - Mean: {np.mean(avg_distances_test):.4f}, Std: {np.std(avg_distances_test):.4f}")
    print(f"Unknown threshold ({threshold_percentile}%): {adaptive_threshold:.4f}")
    
    return adaptive_threshold

def train_robust_knn(X_train, X_test, y_train, y_test, label_encoder, feature_processor):
    """Train KNN dengan analisis lengkap"""
    print("\n=== ROBUST KNN TRAINING ===")
    start_time = time.time()
    
    best_knn, best_params = robust_knn_grid_search(X_train, y_train)
    adaptive_threshold = calculate_adaptive_threshold(best_knn, X_train, y_train, X_test)
    
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)
    
    training_time = time.time() - start_time
    
    print(f"\n=== TRAINING RESULTS ===")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    analyze_confusion_matrix_detailed(cm, label_encoder.classes_)
    interpret_knn_algorithm(best_knn, X_train, y_train, X_test, y_test,
                           label_encoder.classes_, feature_processor)
    plot_separated_visualizations(best_knn, X_train, y_train, X_test, y_test,
                                  label_encoder, best_params, adaptive_threshold,
                                  feature_processor)
    
    return best_knn, accuracy, adaptive_threshold, best_params

def save_robust_model(model, label_encoder, threshold, best_params, feature_processor, filename):
    """Save model dengan Unknown Detection"""
    
    robust_model = RobustKNNWithUnknown(
        knn_model=model,
        distance_threshold=threshold,
        enable_unknown=ENABLE_UNKNOWN_DETECTION
    )
    
    model_data = {
        'model': robust_model,
        'knn_base': model,
        'encoder': label_encoder,
        'model_type': 'EnhancedKNN_DualSource',
        'classes': list(label_encoder.classes_),
        'distance_threshold': threshold,
        'best_params': best_params,
        'feature_processor': feature_processor,
        'version': '3.0_dual_source',
        'data_sources': {
            'local': DATA_ROOT,
            'kaggle': os.path.join(DATA_ROOT, KAGGLE_SUBDIR)
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✅ Model saved: {filename}")

def main():
    print("=" * 70)
    print(" 🎯 ENHANCED KNN TRAINER v3.0 - DUAL SOURCE")
    print("=" * 70)
    print("\n📋 FEATURES:")
    print("  ✅ Auto-detect LOCAL (.npy) + KAGGLE (.jpg) formats")
    print("  ✅ Smart merge of both datasets")
    print("  ✅ 1442-dim MediaPipe features")
    print("  ✅ 70% Training / 30% Testing")
    print("  ✅ 12 separated visualizations")
    print(f"\n🔒 Unknown Detection: {'ENABLED ✅' if ENABLE_UNKNOWN_DETECTION else 'DISABLED ❌'}")
    
    features, labels_raw, class_counts = load_and_analyze_data()
    if features is None:
        return
    
    if len(set(labels_raw)) < 2:
        print("\n❌ Need at least 2 classes!")
        return
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_raw)
    
    print(f"\n📊 Final Dataset:")
    print(f"  • Samples: {len(features)}")
    print(f"  • Classes: {len(label_encoder.classes_)}")
    print(f"  • Features: {features.shape[1]}")
    
    feature_processor = RobustFeatureProcessor()
    processed_features, processed_labels = feature_processor.fit_transform(features, labels_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        processed_features, processed_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=processed_labels
    )
    
    print(f"\n📂 Split (70/30):")
    print(f"  • Training: {len(X_train)}")
    print(f"  • Testing: {len(X_test)}")
    
    model, accuracy, threshold, best_params = train_robust_knn(
        X_train, X_test, y_train, y_test, label_encoder, feature_processor
    )
    
    save_robust_model(model, label_encoder, threshold, best_params, feature_processor, MODEL_FILENAME)
    
    print("\n" + "=" * 70)
    print(" ✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model Type: Enhanced KNN v3.0 with Full Analysis")
    print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Distance Threshold: {threshold:.4f}")
    print(f"Unknown Detection: {'ENABLED ✅' if ENABLE_UNKNOWN_DETECTION else 'DISABLED ❌'}")
    print(f"\n📊 Outputs Generated:")
    print(f"  • Model: {MODEL_FILENAME}")
    print(f"  • Visualizations: {OUTPUT_DIR}/ (12 files)")
    print(f"  • Summary: {OUTPUT_DIR}/12_model_summary.txt")
    print(f"\n🚀 Model ready for deployment!")

if __name__ == "__main__":
    plt.style.use('default')
    main()