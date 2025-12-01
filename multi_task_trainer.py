import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# --- KONFIGURASI ROBUST (SAMA DENGAN ASLI) ---
DATA_ROOT = "data_mp_wajah"
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_PCA = True
PCA_COMPONENTS = 0.99
USE_FEATURE_SELECTION = True
FEATURE_SELECTION_K = 'all'
USE_ROBUST_SCALING = True
USE_DATA_AUGMENTATION = True
AUGMENTATION_NOISE_LEVEL = 0.02
ENABLE_UNKNOWN_DETECTION = True
UNKNOWN_THRESHOLD_PERCENTILE = 85

# --- KONFIGURASI MULTI-TASK (BARU) ---
GENDER_LABELS = ['Pria', 'Wanita']
# Sesuaikan label objek ini dengan nama folder data Anda
OBJECT_LABELS = ['Kacamata', 'Topi', 'TidakAda', 'Masker'] 

MODEL_FILENAME_ID = "model_knn_face_id.pkl"
MODEL_FILENAME_GENDER = "model_knn_gender.pkl"
MODEL_FILENAME_OBJECT = "model_knn_object.pkl"


# ----------------------------------------------------
# Kelas-kelas Pembantu (RobustFeatureProcessor & RobustKNNWithUnknown)
# ----------------------------------------------------
# (Kelas-kelas ini TIDAK DIUBAH dan SAMA persis dengan kode yang Anda berikan)
# ----------------------------------------------------

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


# ----------------------------------------------------
# FUNGSI UTAMA UNTUK MULTI-TASK
# ----------------------------------------------------

def load_and_preprocess_data():
    """Memuat data fitur wajah dan mengurai label untuk 3 tugas (ID, Gender, Object)."""
    features = []
    labels_raw = []
    
    print("Memuat data fitur wajah...")
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Direktori '{DATA_ROOT}' tidak ditemukan!")
        return None
    
    # Bagian ini sama: load fitur dan label mentah
    total_files = 0
    feature_lengths = []
    
    for name in os.listdir(DATA_ROOT):
        person_path = os.path.join(DATA_ROOT, name)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                if filename.endswith(".npy"):
                    file_path = os.path.join(person_path, filename)
                    try:
                        feature = np.load(file_path)
                        if np.any(np.isnan(feature)) or np.any(np.isinf(feature)): continue
                        feature_lengths.append(len(feature))
                        features.append(feature)
                        labels_raw.append(name)
                        total_files += 1
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    if not features:
        print("Tidak ada fitur yang berhasil dimuat!")
        return None
    
    # Filter fitur berdasarkan panjang paling umum
    if len(set(feature_lengths)) > 1:
        most_common_length = max(set(feature_lengths), key=feature_lengths.count)
        filtered_features = []
        filtered_labels = []
        for feat, label in zip(features, labels_raw):
            if len(feat) == most_common_length:
                filtered_features.append(feat)
                filtered_labels.append(label)
        features = filtered_features
        labels_raw = filtered_labels
    
    features = np.array(features)
    labels_raw = np.array(labels_raw)
    
    print(f"Total sampel dimuat: {len(features)}. Fitur shape: {features.shape}")
    
    # ----------------------------------------------------
    # ✨ MULTI-TASK LABEL PARSING
    # ----------------------------------------------------
    labels_id = []
    labels_gender = []
    labels_object = []

    print("\n=== PARSING MULTI-TASK LABELS ===")
    
    for full_label in labels_raw:
        # Asumsi format: [Gender]_[ID]_[Object] atau [ID]
        parts = full_label.split('_')
        
        # 1. Label ID (misal: Andy)
        id_label = parts[1] if len(parts) > 1 and parts[1] else full_label
        labels_id.append(id_label)
        
        # 2. Label Gender (misal: Pria)
        gender_label = parts[0] if parts[0] in GENDER_LABELS else 'Gender_Unclassified'
        labels_gender.append(gender_label)
        
        # 3. Label Object (misal: Kacamata)
        object_label = parts[2] if len(parts) > 2 and parts[2] in OBJECT_LABELS else 'TidakAda'
        labels_object.append(object_label)
        
    
    # ----------------------------------------------------
    # 4. PREPROCESSING & ENCODING
    # ----------------------------------------------------
    feature_processor = RobustFeatureProcessor()
    
    # Gunakan ID untuk fitting Feature Processor (karena ID adalah klasifikasi utama)
    label_encoder_id = LabelEncoder()
    labels_encoded_id = label_encoder_id.fit_transform(labels_id)
    
    # Fit & Transform fitur HANYA SEKALI
    processed_features, processed_labels_id_encoded = feature_processor.fit_transform(features, labels_encoded_id)
    
    # Encode label lain (menggunakan fit_transform agar semua label terwakili)
    label_encoder_gender = LabelEncoder()
    labels_encoded_gender = label_encoder_gender.fit_transform(labels_gender)
    
    label_encoder_object = LabelEncoder()
    labels_encoded_object = label_encoder_object.fit_transform(labels_object)
    
    return {
        'features': processed_features,
        'labels_id': processed_labels_id_encoded,
        'labels_gender': labels_encoded_gender,
        'labels_object': labels_encoded_object,
        'processor': feature_processor,
        'encoder_id': label_encoder_id,
        'encoder_gender': label_encoder_gender,
        'encoder_object': label_encoder_object
    }

# (fungsi robust_knn_grid_search, calculate_adaptive_threshold, dan plot_comprehensive_analysis SAMA)
# (fungsi train_robust_knn SAMA)
# (fungsi save_robust_model SAMA)

# Menyalin fungsi-fungsi pendukung yang SAMA persis dari kode Anda sebelumnya:

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
    print("\n=== CALCULATING ADAPTIVE THRESHOLD FOR UNKNOWN DETECTION ===")
    
    distances_train, _ = best_knn.kneighbors(X_train)
    avg_distances_train = np.mean(distances_train, axis=1)
    
    distances_test, _ = best_knn.kneighbors(X_test)
    avg_distances_test = np.mean(distances_test, axis=1)
    
    threshold_percentile = UNKNOWN_THRESHOLD_PERCENTILE
    adaptive_threshold = np.percentile(
        np.concatenate([avg_distances_train, avg_distances_test]), 
        threshold_percentile
    )
    
    print(f"Unknown threshold ({threshold_percentile}th percentile): {adaptive_threshold:.4f}")
    
    return adaptive_threshold

def plot_comprehensive_analysis(best_knn, X_train, y_train, X_test, y_test, 
                                 label_encoder, best_params, adaptive_threshold, 
                                 feature_processor):
    """Plot analisis komprehensif (Disederhanakan untuk Multi-task)"""
    # Catatan: Plot ini akan menampilkan analisis untuk model yang sedang di-train (ID, Gender, atau Object)
    print("\n=== CREATING COMPREHENSIVE VISUALIZATION ===")
    
    fig = plt.figure(figsize=(20, 12))
    
    y_pred = best_knn.predict(X_test)
    
    distances_train, _ = best_knn.kneighbors(X_train)
    distances_test, _ = best_knn.kneighbors(X_test)
    avg_dist_train = np.mean(distances_train, axis=1)
    avg_dist_test = np.mean(distances_test, axis=1)
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    # Handle prediksi Unknown (-1) jika ada
    y_test_clean = y_test[y_pred != -1]
    y_pred_clean = y_pred[y_pred != -1]
    cm = confusion_matrix(y_test_clean, y_pred_clean)
    
    try:
        # Filter classes agar hanya yang muncul di y_test_clean
        unique_classes = np.unique(y_test_clean)
        class_names = [label_encoder.classes_[c] for c in unique_classes]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names, ax=ax1)
    except:
         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1) # Fallback

    ax1.set_title(f'Confusion Matrix\nAccuracy: {accuracy_score(y_test_clean, y_pred_clean):.3f}', 
                  fontsize=14, fontweight='bold')
    
    # 2. K-Neighbors Performance (Sama)
    ax2 = plt.subplot(2, 3, 2)
    k_values = [3, 5, 7, 9, 11, 15, 21, 25, 31]
    train_accs = []
    test_accs = []
    cv_accs = []
    
    optimal_params = best_params.copy()
    for k in k_values:
        if k > len(X_train) // 2: break
        optimal_params['n_neighbors'] = k
        knn_temp = KNeighborsClassifier(**optimal_params)
        knn_temp.fit(X_train, y_train)
        
        train_accs.append(accuracy_score(y_train, knn_temp.predict(X_train)))
        test_accs.append(accuracy_score(y_test, knn_temp.predict(X_test)))
        cv_accs.append(cross_val_score(knn_temp, X_train, y_train, cv=5).mean())
    
    ax2.plot(k_values[:len(train_accs)], train_accs, 'bo-', label='Train', linewidth=2)
    ax2.plot(k_values[:len(test_accs)], test_accs, 'ro-', label='Test', linewidth=2)
    ax2.plot(k_values[:len(cv_accs)], cv_accs, 'go-', label='CV', linewidth=2)
    ax2.axvline(x=best_params['n_neighbors'], color='purple', linestyle='--', label=f"Optimal K={best_params['n_neighbors']}")
    ax2.set_title('K-Neighbors vs Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('K (Number of Neighbors)')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Distance Distribution + Unknown Zone (Sama)
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(avg_dist_train, bins=30, alpha=0.6, label='Train', color='blue', edgecolor='black')
    ax3.hist(avg_dist_test, bins=30, alpha=0.6, label='Test', color='red', edgecolor='black')
    ax3.axvline(x=adaptive_threshold, color='green', linestyle='--', linewidth=2, label=f'Unknown Threshold={adaptive_threshold:.3f}')
    if ENABLE_UNKNOWN_DETECTION:
        ax3.axvspan(adaptive_threshold, ax3.get_xlim()[1], alpha=0.2, color='orange', label='Unknown Zone')
    
    ax3.set_title('Distance Distribution + Unknown Detection', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. PCA Variance (Sama)
    ax4 = plt.subplot(2, 3, 4)
    if feature_processor.pca is not None:
        explained_var = feature_processor.pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        n_components = len(explained_var)
        ax4.bar(range(1, min(51, n_components+1)), explained_var[:50], alpha=0.6, color='skyblue', label='Individual')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(range(1, min(51, n_components+1)), cumsum_var[:50], 'ro-', linewidth=2, label='Cumulative')
        ax4_twin.axhline(y=0.99, color='green', linestyle='--', label='99% Variance')
        ax4.set_title(f'PCA Variance (Total: {n_components} components)', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'PCA not used', ha='center', va='center', fontsize=14)
        ax4.set_title('PCA Analysis', fontsize=14, fontweight='bold')
    
    # 5. Per-Class Performance (Sama)
    ax5 = plt.subplot(2, 3, 5)
    report = classification_report(y_test, y_pred, 
                                   labels=range(len(label_encoder.classes_)),
                                   target_names=label_encoder.classes_, 
                                   output_dict=True, zero_division=0)
    
    classes = label_encoder.classes_
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    ax5.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax5.bar(x, recall, width, label='Recall', alpha=0.8)
    ax5.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    ax5.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(classes, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Model Summary (Sama)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    cv_score = cross_val_score(best_knn, X_train, y_train, cv=5).mean()
    test_accuracy = accuracy_score(y_test, y_pred)
    
    variance_explained = feature_processor.pca.explained_variance_ratio_.sum() if feature_processor.pca else 0.0
    
    summary_text = f"""
ROBUST KNN MODEL SUMMARY
========================================

Model Configuration:
• K Neighbors: {best_params['n_neighbors']}
• Metric: {best_params['metric']}

Dataset Info:
• Train Samples: {len(X_train)}
• Test Samples: {len(X_test)}
• Features: {X_train.shape[1]}
• Classes: {len(label_encoder.classes_)}

Performance Metrics:
• Test Accuracy: {test_accuracy:.4f}
• CV Score: {cv_score:.4f}
• Unknown Threshold: {adaptive_threshold:.4f}

Preprocessing:
• PCA Components: {X_train.shape[1]}
• Variance Explained: {variance_explained:.4f}
• Augmentation: {USE_DATA_AUGMENTATION}
"""
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    # Simpan plot dengan nama file yang spesifik untuk task-nya
    task_name = label_encoder.classes_[0].split('_')[0] if label_encoder.classes_ else 'Unknown'
    plt.savefig(f'robust_knn_analysis_{task_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig) # Tutup figure untuk menghemat memori
    print(f"✓ Comprehensive analysis saved: robust_knn_analysis_{task_name}.png")

def train_robust_knn(X_train, X_test, y_train, y_test, label_encoder, feature_processor):
    """Train KNN yang robust"""
    print("\n=== ROBUST KNN TRAINING ===")
    start_time = time.time()
    
    best_knn, best_params = robust_knn_grid_search(X_train, y_train)
    adaptive_threshold = calculate_adaptive_threshold(best_knn, X_train, y_train, X_test)
    
    y_pred = best_knn.predict(X_test)
    # Gunakan y_test_clean untuk akurasi test
    y_test_clean = y_test[y_pred != -1]
    y_pred_clean = y_pred[y_pred != -1]

    accuracy = accuracy_score(y_test_clean, y_pred_clean)
    cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)
    
    training_time = time.time() - start_time
    
    print("\n=== TRAINING RESULTS ===")
    print("Test Accuracy (excluding Unknown): {:.4f}".format(accuracy))
    
    plot_comprehensive_analysis(best_knn, X_train, y_train, X_test, y_test,
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
        'model_type': 'RobustKNN_with_Unknown',
        'classes': list(label_encoder.classes_),
        'distance_threshold': threshold,
        'best_params': best_params,
        # Simpan hanya parameter konfigurasi processor, bukan objek yang sudah fit
        'preprocessing_config': feature_processor.config, 
        'unknown_detection_enabled': ENABLE_UNKNOWN_DETECTION,
        'version': '3.0_multi_task'
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✓ Robust model for {os.path.basename(filename)} saved: {filename}")


def main():
    print("=== ROBUST KNN MULTI-TASK TRAINING ===")
    
    data = load_and_preprocess_data()
    if data is None or data['features'] is None:
        return
    
    # 1. SPLIT DATA (Dilakukan sekali, dijamin stratifikasi berdasarkan ID)
    X_train, X_test, y_id_train, y_id_test, y_gender_train, y_gender_test, y_object_train, y_object_test = train_test_split(
        data['features'], data['labels_id'], data['labels_gender'], data['labels_object'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data['labels_id'] # Stratify berdasarkan ID
    )
    
    print("\n Split dataset:")
    print("   Training: {} sampel".format(len(X_train)))
    print("   Testing: {} sampel".format(len(X_test)))
    
    # Inisialisasi processor dummy untuk tugas selain ID (agar tidak fit ulang)
    dummy_processor = RobustFeatureProcessor(data['processor'].config) 
    
    # ----------------------------------------------------
    # 2. TRAIN & SAVE MODEL ID (Face Recognition)
    # ----------------------------------------------------
    print("\n\n" + "#" * 50)
    print("## START TRAINING TASK 1: FACE ID (RECOGNITION) ##")
    print("#" * 50)
    
    model_id, acc_id, thresh_id, params_id = train_robust_knn(
        X_train, X_test, y_id_train, y_id_test, data['encoder_id'], data['processor']
    )
    save_robust_model(model_id, data['encoder_id'], thresh_id, params_id, data['processor'], MODEL_FILENAME_ID)


    # ----------------------------------------------------
    # 3. TRAIN & SAVE MODEL GENDER
    # ----------------------------------------------------
    print("\n\n" + "#" * 50)
    print("## START TRAINING TASK 2: GENDER CLASSIFICATION ##")
    print("#" * 50)
    
    model_gender, acc_gender, thresh_gender, params_gender = train_robust_knn(
        X_train, X_test, y_gender_train, y_gender_test, data['encoder_gender'], dummy_processor
    )
    save_robust_model(model_gender, data['encoder_gender'], thresh_gender, params_gender, dummy_processor, MODEL_FILENAME_GENDER)

    # ----------------------------------------------------
    # 4. TRAIN & SAVE MODEL OBJECT (Kacamata/Topi)
    # ----------------------------------------------------
    print("\n\n" + "#" * 50)
    print("## START TRAINING TASK 3: OBJECT RECOGNITION (Kacamata/Topi) ##")
    print("#" * 50)

    model_object, acc_object, thresh_object, params_object = train_robust_knn(
        X_train, X_test, y_object_train, y_object_test, data['encoder_object'], dummy_processor
    )
    save_robust_model(model_object, data['encoder_object'], thresh_object, params_object, dummy_processor, MODEL_FILENAME_OBJECT)

    print("\n=== ROBUST MULTI-TASK TRAINING COMPLETE ===")
    print(f"Final Accuracies (excluding Unknown): ID={acc_id:.4f}, Gender={acc_gender:.4f}, Object={acc_object:.4f}")

if __name__ == "__main__":
    main()