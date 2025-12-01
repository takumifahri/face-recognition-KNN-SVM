import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import cv2
import mediapipe as mp
warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
DATA_ROOT = "data_mp_wajah"
SVM_MODEL_FILENAME = "face_svm_optimized_model.pkl"
TEST_SIZE = 0.3
RANDOM_STATE = 42

# 🎯 KAGGLE SETTINGS
KAGGLE_SUBDIR = "kaggle/Faces/Faces"
KAGGLE_MAX_SAMPLES_PER_CLASS = 200

# 🔒 UNKNOWN DETECTION
ENABLE_UNKNOWN_DETECTION = True
UNKNOWN_DECISION_THRESHOLD_PERCENTILE = 15

# 📁 OUTPUT DIRECTORIES
OUTPUT_DIR = "svm_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SVM Parameters
SVM_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

# 🎯 MEDIAPIPE SETUP
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

class RobustSVMWithUnknown:
    """SVM classifier dengan unknown detection"""
    
    def __init__(self, svm_model, decision_threshold, enable_unknown=True):
        self.svm = svm_model
        self.decision_threshold = decision_threshold
        self.enable_unknown = enable_unknown
    
    def predict(self, X):
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
        
        confidence = 1 / (1 + np.exp(-max_scores))
        
        return predictions, confidence
    
    def predict_proba(self, X):
        return self.svm.predict_proba(X)
    
    def decision_function(self, X):
        return self.svm.decision_function(X)

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

def interpret_svm_algorithm(best_svm, X_train, y_train, X_test, y_test, class_names, scaler):
    """🧠 INTERPRETASI ALGORITMA SVM"""
    print("\n" + "="*70)
    print("🧠 SUPPORT VECTOR MACHINE (SVM) - ALGORITHM INTERPRETATION")
    print("="*70)
    
    kernel = best_svm.kernel
    C = best_svm.C
    gamma = best_svm.gamma
    
    print("\n1️⃣  ALGORITHM OVERVIEW:")
    print(f"""
   SVM finds optimal hyperplane using '{kernel}' kernel
   Effective in {X_train.shape[1]}-dimensional space
   Memory efficient with support vectors only
    """)
    
    print("\n2️⃣  MODEL PARAMETERS:")
    print(f"   C = {C} (Regularization)")
    print(f"   Kernel = '{kernel}'")
    print(f"   Gamma = {gamma}")
    
    print("\n3️⃣  SUPPORT VECTORS:")
    n_support = best_svm.n_support_
    total_support = np.sum(n_support)
    print(f"   Total: {total_support}/{len(X_train)} ({total_support/len(X_train)*100:.1f}%)")
    
    for i, (class_name, count) in enumerate(zip(class_names, n_support)):
        class_size = np.sum(y_train == i)
        print(f"   • {class_name}: {count}/{class_size} ({count/class_size*100:.1f}%)")

def plot_separated_visualizations(best_svm, X_train, y_train, X_test, y_test, 
                                  label_encoder, best_params, decision_threshold, scaler):
    """📊 CREATE ALL 12 SEPARATED VISUALIZATIONS"""
    print("\n=== CREATING 12 SEPARATED VISUALIZATIONS ===")
    
    y_pred = best_svm.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    decision_train = best_svm.decision_function(X_train)
    decision_test = best_svm.decision_function(X_test)
    
    if decision_train.ndim > 1:
        max_decision_train = np.max(decision_train, axis=1)
        max_decision_test = np.max(decision_test, axis=1)
    else:
        max_decision_train = decision_train
        max_decision_test = decision_test
    
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
    
    # 2. C Parameter Optimization
    print("  [2/12] C Parameter Optimization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    train_accs = []
    test_accs = []
    cv_accs = []
    
    optimal_params = best_params.copy()
    for c in c_values:
        optimal_params['C'] = c
        svm_temp = SVC(**optimal_params, probability=True, random_state=RANDOM_STATE)
        svm_temp.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, svm_temp.predict(X_train)))
        test_accs.append(accuracy_score(y_test, svm_temp.predict(X_test)))
        cv_accs.append(cross_val_score(svm_temp, X_train, y_train, cv=5).mean())
    
    ax.semilogx(c_values, train_accs, 'bo-', label='Train', linewidth=2, markersize=8)
    ax.semilogx(c_values, test_accs, 'ro-', label='Test', linewidth=2, markersize=8)
    ax.semilogx(c_values, cv_accs, 'go-', label='CV', linewidth=2, markersize=8)
    ax.axvline(x=best_params['C'], color='purple', linestyle='--', linewidth=2,
                label=f"Optimal C={best_params['C']}")
    ax.set_title('C Parameter Optimization (Regularization)', fontsize=14, fontweight='bold')
    ax.set_xlabel('C (Regularization)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_c_parameter_optimization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Decision Function Distribution
    print("  [3/12] Decision Function Distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(max_decision_train, bins=40, alpha=0.6, label='Train', color='blue', edgecolor='black')
    ax.hist(max_decision_test, bins=40, alpha=0.6, label='Test', color='red', edgecolor='black')
    ax.axvline(x=decision_threshold, color='green', linestyle='--', linewidth=3,
                label=f'Threshold={decision_threshold:.3f}')
    ax.axvline(x=np.mean(max_decision_train), color='blue', linestyle='-', linewidth=2,
                label=f'Mean Train={np.mean(max_decision_train):.3f}')
    ax.axvline(x=np.mean(max_decision_test), color='red', linestyle='-', linewidth=2,
                label=f'Mean Test={np.mean(max_decision_test):.3f}')
    if ENABLE_UNKNOWN_DETECTION:
        ax.axvspan(ax.get_xlim()[0], decision_threshold, alpha=0.3, color='orange', 
                   label='Unknown Detection Zone')
    ax.set_title('Decision Function Distribution (Prediction Confidence)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Decision Function Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_decision_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Support Vectors Analysis
    print("  [4/12] Support Vectors Analysis...")
    fig, ax = plt.subplots(figsize=(10, 6))
    n_support = best_svm.n_support_
    
    bars = ax.bar(classes, n_support, color='coral', alpha=0.7, edgecolor='black')
    for i, (bar, count) in enumerate(zip(bars, n_support)):
        class_size = np.sum(y_train == i)
        percentage = (count / class_size) * 100 if class_size > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., count,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    total_sv = np.sum(n_support)
    ax.set_title(f'Support Vectors per Class\nTotal: {total_sv}/{len(X_train)} ({total_sv/len(X_train)*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Support Vectors', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_support_vectors.png'), dpi=300, bbox_inches='tight')
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
    correct_scores = max_decision_test[correct_mask]
    incorrect_scores = max_decision_test[~correct_mask] if np.sum(~correct_mask) > 0 else []
    
    box_data = [correct_scores]
    labels_box = ['Correct\nPredictions']
    colors = ['green']
    
    if len(incorrect_scores) > 0:
        box_data.append(incorrect_scores)
        labels_box.append('Incorrect\nPredictions')
        colors.append('red')
    
    bp = ax.boxplot(box_data, labels=labels_box, patch_artist=True, 
                     widths=0.6, showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(y=decision_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Threshold={decision_threshold:.3f}')
    ax.set_title('Prediction Confidence Analysis (Decision Scores)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decision Function Score', fontsize=12)
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
        if n_samples < len(classes):
            continue
        svm_temp = SVC(**best_params, probability=True, random_state=RANDOM_STATE)
        svm_temp.fit(X_train[:n_samples], y_train[:n_samples])
        train_scores.append(accuracy_score(y_train[:n_samples], svm_temp.predict(X_train[:n_samples])))
        test_scores.append(accuracy_score(y_test, svm_temp.predict(X_test)))
    
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
    
    # 9. Probability Distribution
    print("  [9/12] Probability Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    probabilities = best_svm.predict_proba(X_test)
    max_probs = np.max(probabilities, axis=1)
    
    ax.hist(max_probs, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=np.mean(max_probs), color='red', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(max_probs):.3f}')
    ax.axvline(x=np.median(max_probs), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(max_probs):.3f}')
    ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Max Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_probability_distribution.png'), dpi=300, bbox_inches='tight')
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
    cv_score = cross_val_score(best_svm, X_train, y_train, cv=5).mean()
    cv_std = cross_val_score(best_svm, X_train, y_train, cv=5).std()
    total_sv = np.sum(best_svm.n_support_)
    unknown_status = "ENABLED ✅" if ENABLE_UNKNOWN_DETECTION else "DISABLED ❌"
    
    summary_text = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           ENHANCED SVM MODEL SUMMARY - DUAL SOURCE v3.0             ║
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
  • Algorithm: Support Vector Machine (SVM)
  • Kernel: {best_params['kernel']}
  • C (Regularization): {best_params['C']}
  • Gamma: {best_params['gamma']}
  • Support Vectors: {total_sv}/{len(X_train)} ({total_sv/len(X_train)*100:.1f}%)

📏 DECISION STATISTICS:
═══════════════════════════════════════════════════════════════════════
  • Train Mean Score: {np.mean(max_decision_train):.4f}
  • Test Mean Score: {np.mean(max_decision_test):.4f}
  • Train Std Dev: {np.std(max_decision_train):.4f}
  • Test Std Dev: {np.std(max_decision_test):.4f}
  • Score Ratio (Test/Train): {np.mean(max_decision_test)/np.mean(max_decision_train):.2f}x

🎓 SUPPORT VECTORS:
═══════════════════════════════════════════════════════════════════════
  • Total Support Vectors: {total_sv}/{len(X_train)} ({total_sv/len(X_train)*100:.1f}%)
  • Per-Class Distribution:
{chr(10).join([f'    • {name}: {count}' for name, count in zip(classes[:10], n_support[:10])])}
{'    ...' if len(classes) > 10 else ''}

🔧 PREPROCESSING PIPELINE:
═══════════════════════════════════════════════════════════════════════
  • Original Features: 1442 (face landmarks + geometric features)
  • Final Features: {X_train.shape[1]}
  • Scaler: RobustScaler (10-90 percentile)
  • Feature Selection: Auto (via SVM kernel)

🔒 UNKNOWN DETECTION:
═══════════════════════════════════════════════════════════════════════
  • Status: {unknown_status}
  • Decision Threshold: {decision_threshold:.4f}
  • Percentile: {UNKNOWN_DECISION_THRESHOLD_PERCENTILE}th
  • Method: Decision function-based threshold

📂 OUTPUT FILES:
═══════════════════════════════════════════════════════════════════════
  01. Confusion Matrix
  02. C Parameter Optimization
  03. Decision Function Distribution
  04. Support Vectors Analysis
  05. Per-Class Performance
  06. Prediction Confidence
  07. Class Balance Analysis
  08. Learning Curve
  09. Probability Distribution
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
    print("   02_c_parameter_optimization.png")
    print("   03_decision_distribution.png")
    print("   04_support_vectors.png")
    print("   05_per_class_performance.png")
    print("   06_prediction_confidence.png")
    print("   07_class_balance.png")
    print("   08_learning_curve.png")
    print("   09_probability_distribution.png")
    print("   10_misclassification_matrix.png")
    print("   11_normalized_confusion_matrix.png")
    print("   12_model_summary.txt")

def optimized_svm_grid_search(X_train, y_train):
    """Grid search for optimal SVM"""
    print("\n=== SVM GRID SEARCH ===")
    
    svm = SVC(probability=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        svm, SVM_PARAM_GRID,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def calculate_decision_threshold(best_svm, X_train, y_train, X_test):
    """Calculate adaptive decision threshold"""
    print("\n=== CALCULATING DECISION THRESHOLD ===")
    
    decision_train = best_svm.decision_function(X_train)
    decision_test = best_svm.decision_function(X_test)
    
    if decision_train.ndim > 1:
        max_decision_train = np.max(decision_train, axis=1)
        max_decision_test = np.max(decision_test, axis=1)
    else:
        max_decision_train = decision_train
        max_decision_test = decision_test
    
    threshold_percentile = UNKNOWN_DECISION_THRESHOLD_PERCENTILE
    decision_threshold = np.percentile(
        np.concatenate([max_decision_train, max_decision_test]),
        threshold_percentile
    )
    
    print(f"Train decision - Mean: {np.mean(max_decision_train):.4f}, Std: {np.std(max_decision_train):.4f}")
    print(f"Test decision - Mean: {np.mean(max_decision_test):.4f}, Std: {np.std(max_decision_test):.4f}")
    print(f"Unknown threshold ({threshold_percentile}th percentile): {decision_threshold:.4f}")
    
    return decision_threshold

def train_enhanced_svm(X_train, X_test, y_train, y_test, label_encoder):
    """Train enhanced SVM with full analysis"""
    print("\n=== ENHANCED SVM TRAINING ===")
    start_time = time.time()
    
    scaler = RobustScaler(quantile_range=(10, 90))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_svm, best_params = optimized_svm_grid_search(X_train_scaled, y_train)
    decision_threshold = calculate_decision_threshold(best_svm, X_train_scaled, y_train, X_test_scaled)
    
    y_pred = best_svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
    
    training_time = time.time() - start_time
    
    print(f"\n=== TRAINING RESULTS ===")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    analyze_confusion_matrix_detailed(cm, label_encoder.classes_)
    interpret_svm_algorithm(best_svm, X_train_scaled, y_train, X_test_scaled, y_test,
                           label_encoder.classes_, scaler)
    plot_separated_visualizations(best_svm, X_train_scaled, y_train, X_test_scaled, y_test,
                                  label_encoder, best_params, decision_threshold, scaler)
    
    return best_svm, scaler, accuracy, decision_threshold, best_params

def save_enhanced_model(model, label_encoder, scaler, threshold, best_params, filename):
    """Save SVM model with Unknown Detection"""
    
    robust_model = RobustSVMWithUnknown(
        svm_model=model,
        decision_threshold=threshold,
        enable_unknown=ENABLE_UNKNOWN_DETECTION
    )
    
    model_data = {
        'model': robust_model,
        'svm_base': model,
        'encoder': label_encoder,
        'scaler': scaler,
        'model_type': 'EnhancedSVM_DualSource',
        'classes': list(label_encoder.classes_),
        'decision_threshold': threshold,
        'best_params': best_params,
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
    print(" 🎯 ENHANCED SVM TRAINER v3.0 - DUAL SOURCE")
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
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_encoded
    )
    
    print(f"\n📂 Split (70/30):")
    print(f"  • Training: {len(X_train)}")
    print(f"  • Testing: {len(X_test)}")
    
    model, scaler, accuracy, threshold, best_params = train_enhanced_svm(
        X_train, X_test, y_train, y_test, label_encoder
    )
    
    save_enhanced_model(model, label_encoder, scaler, threshold, best_params, SVM_MODEL_FILENAME)
    
    print("\n" + "=" * 70)
    print(" ✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model Type: Enhanced SVM v3.0 with Full Analysis")
    print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Decision Threshold: {threshold:.4f}")
    print(f"Unknown Detection: {'ENABLED ✅' if ENABLE_UNKNOWN_DETECTION else 'DISABLED ❌'}")
    print(f"\n📊 Outputs Generated:")
    print(f"  • Model: {SVM_MODEL_FILENAME}")
    print(f"  • Visualizations: {OUTPUT_DIR}/ (12 files)")
    print(f"  • Summary: {OUTPUT_DIR}/12_model_summary.txt")
    print(f"\n🚀 Model ready for deployment!")

if __name__ == "__main__":
    plt.style.use('default')
    main()