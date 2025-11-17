import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# --- KONFIGURASI OPTIMIZED ---
DATA_ROOT = "data_mp_wajah"
MODEL_FILENAME = "face_knn_optimized_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing options
USE_PCA = True
PCA_COMPONENTS = 0.95  # Keep 95% of variance
USE_FEATURE_SELECTION = True
FEATURE_SELECTION_K = 500  # Select top 500 features
USE_ROBUST_SCALING = True  # Better for outliers than StandardScaler

class FeatureProcessor:
    """Class untuk preprocessing yang bisa disimpan dan digunakan kembali"""
    
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
                'use_robust_scaling': USE_ROBUST_SCALING
            }
        else:
            self.config = preprocessing_config
    
    def fit_transform(self, features, labels=None):
        """Fit dan transform features (untuk training)"""
        print("\n=== ADVANCED PREPROCESSING WITH FEATURE PROCESSOR ===")
        print(f"Original features shape: {features.shape}")
        
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
        self.correlation_features_mask = np.array(remaining_features)
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
            self.pca = PCA(n_components=self.config['pca_components'], random_state=RANDOM_STATE)
            features = self.pca.fit_transform(features)
            print(f"After PCA: {features.shape}")
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        print(f"Final processed features shape: {features.shape}")
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

def load_and_preprocess_data():
    """Memuat dan preprocessing data fitur wajah"""
    features = []
    labels_raw = []
    
    print("Memuat data fitur wajah...")
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Direktori '{DATA_ROOT}' tidak ditemukan!")
        return None, None, None, None
    
    total_files = 0
    class_counts = {}
    
    for name in os.listdir(DATA_ROOT):
        person_path = os.path.join(DATA_ROOT, name)
        if os.path.isdir(person_path):
            person_files = 0
            for filename in os.listdir(person_path):
                if filename.endswith(".npy"):
                    file_path = os.path.join(person_path, filename)
                    try:
                        feature = np.load(file_path)
                        
                        # Data validation
                        if np.any(np.isnan(feature)) or np.any(np.isinf(feature)):
                            print(f"Warning: Invalid data in {file_path}, skipping...")
                            continue
                        
                        features.append(feature)
                        labels_raw.append(name)
                        person_files += 1
                        total_files += 1
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
            print(f"  {name}: {person_files} sampel")
            class_counts[name] = person_files
    
    if not features:
        print("Tidak ada fitur yang berhasil dimuat!")
        return None, None, None, None
    
    print(f"Total sampel dimuat: {total_files}")
    
    # Convert to numpy arrays
    features = np.array(features)
    labels_raw = np.array(labels_raw)
    
    # Data analysis before preprocessing
    print(f"\n=== DATA ANALYSIS ===")
    print(f"Feature shape: {features.shape}")
    print(f"Feature range: [{np.min(features):.6f}, {np.max(features):.6f}]")
    print(f"Feature mean: {np.mean(features):.6f}")
    print(f"Feature std: {np.std(features):.6f}")
    print(f"NaN values: {np.sum(np.isnan(features))}")
    print(f"Inf values: {np.sum(np.isinf(features))}")
    
    # Create and fit feature processor
    feature_processor = FeatureProcessor()
    
    # Encode labels for feature selection
    label_encoder_temp = LabelEncoder()
    labels_encoded_temp = label_encoder_temp.fit_transform(labels_raw)
    
    # Advanced preprocessing dengan feature processor
    processed_features = feature_processor.fit_transform(features, labels_encoded_temp)
    
    # Visualisasi distribusi data
    plot_data_distribution(class_counts)
    
    return processed_features, labels_raw, class_counts, feature_processor

def plot_data_distribution(class_counts):
    """Plot distribusi data per kelas"""
    if not class_counts:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Class distribution
    plt.subplot(2, 2, 1)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    bars = plt.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Distribusi Sampel per Kelas', fontsize=14, fontweight='bold')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Sampel')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Class balance analysis
    plt.subplot(2, 2, 2)
    plt.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Proporsi Kelas')
    
    # Subplot 3: Sample distribution statistics
    plt.subplot(2, 2, 3)
    stats = ['Min', 'Max', 'Mean', 'Std']
    values = [min(counts), max(counts), np.mean(counts), np.std(counts)]
    plt.bar(stats, values, color='lightblue', alpha=0.7, edgecolor='black')
    plt.title('Statistik Distribusi Sampel')
    plt.ylabel('Jumlah Sampel')
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Data quality metrics
    plt.subplot(2, 2, 4)
    total_samples = sum(counts)
    min_samples = min(counts)
    balance_ratio = min_samples / max(counts)
    
    metrics = ['Total\nSamples', 'Min\nSamples', 'Balance\nRatio']
    metric_values = [total_samples, min_samples, balance_ratio]
    colors_metrics = ['green' if balance_ratio > 0.7 else 'orange' if balance_ratio > 0.5 else 'red']
    
    bars = plt.bar(metrics[:2], metric_values[:2], color='lightgreen', alpha=0.7)
    plt.bar(metrics[2], metric_values[2], color=colors_metrics[0], alpha=0.7)
    
    plt.title('Metrik Kualitas Data')
    plt.ylabel('Nilai')
    
    for bar, value in zip(plt.gca().patches, metric_values):
        if bar == plt.gca().patches[2]:  # Balance ratio bar
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('optimized_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Optimized data distribution plot saved as 'optimized_data_distribution.png'")

def optimized_knn_grid_search(X_train, y_train):
    """Grid search untuk mencari hyperparameter optimal KNN"""
    print("Melakukan grid search untuk hyperparameter optimal...")
    
    # Parameter grid yang lebih komprehensif
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'cosine'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    }
    
    # GridSearchCV dengan cross-validation
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

def plot_optimized_performance_analysis(X_train, y_train, X_test, y_test, best_knn, best_params):
    """Analisis performa komprehensif"""
    print("Melakukan analisis performa komprehensif...")
    
    # Test range K values dengan parameter optimal lainnya
    k_values = range(1, min(31, len(X_train) // 3))
    train_accuracies = []
    test_accuracies = []
    cv_scores_mean = []
    
    optimal_params = best_params.copy()
    
    for k in k_values:
        optimal_params['n_neighbors'] = k
        knn = KNeighborsClassifier(**optimal_params)
        knn.fit(X_train, y_train)
        
        # Training accuracy
        train_pred = knn.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_accuracies.append(train_acc)
        
        # Test accuracy
        test_pred = knn.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_accuracies.append(test_acc)
        
        # Cross validation
        cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores_mean.append(cv_scores.mean())
    
    # Plotting
    plt.figure(figsize=(18, 6))
    
    # Subplot 1: K vs Accuracy with optimized parameters
    plt.subplot(1, 3, 1)
    plt.plot(k_values, train_accuracies, 'bo-', label='Training Accuracy', linewidth=2)
    plt.plot(k_values, test_accuracies, 'ro-', label='Test Accuracy', linewidth=2)
    plt.plot(k_values, cv_scores_mean, 'go-', label='CV Accuracy', linewidth=2)
    
    best_k = best_params['n_neighbors']
    plt.axvline(x=best_k, color='purple', linestyle='--', alpha=0.7, 
                label=f'Optimal K={best_k}')
    
    plt.title('Optimized KNN Performance vs K Value', fontweight='bold')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 2: Distance distribution analysis
    plt.subplot(1, 3, 2)
    
    # Get distances for test samples
    distances, indices = best_knn.kneighbors(X_test)
    avg_distances = np.mean(distances, axis=1)
    
    plt.hist(avg_distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(avg_distances), color='red', linestyle='-', 
                label=f'Mean: {np.mean(avg_distances):.3f}')
    plt.axvline(x=np.percentile(avg_distances, 75), color='orange', linestyle='--', 
                label=f'75th percentile: {np.percentile(avg_distances, 75):.3f}')
    
    plt.title('Average Distance Distribution', fontweight='bold')
    plt.xlabel('Average Distance to K Neighbors')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 3: Performance metrics comparison
    plt.subplot(1, 3, 3)
    
    # Calculate various metrics
    y_pred = best_knn.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, best_knn.predict(X_train))
    cv_acc = np.mean(cross_val_score(best_knn, X_train, y_train, cv=5))
    
    metrics = ['Train Acc', 'Test Acc', 'CV Acc']
    values = [train_acc, test_acc, cv_acc]
    colors = ['blue', 'red', 'green']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Optimized Performance Metrics', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimized_knn_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Optimized KNN performance analysis saved as 'optimized_knn_performance.png'")
    
    # Calculate and return adaptive threshold
    threshold_percentile = 85  # Use 85th percentile as threshold
    adaptive_threshold = np.percentile(avg_distances, threshold_percentile)
    
    return adaptive_threshold

def train_optimized_knn(X_train, X_test, y_train, y_test, label_encoder):
    """Melatih KNN yang dioptimasi"""
    print(f"\n=== OPTIMIZED KNN TRAINING ===")
    start_time = time.time()
    
    # Grid search untuk parameter optimal
    best_knn, best_params = optimized_knn_grid_search(X_train, y_train)
    
    # Analisis performa
    adaptive_threshold = plot_optimized_performance_analysis(
        X_train, y_train, X_test, y_test, best_knn, best_params
    )
    
    # Final evaluation
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)
    
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} detik")
    print(f"Best parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Score (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Adaptive threshold: {adaptive_threshold:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Optimized KNN Confusion Matrix\nAccuracy: {accuracy:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('optimized_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_knn, accuracy, adaptive_threshold, best_params

def save_optimized_model(model, label_encoder, threshold, best_params, feature_processor, filename):
    """Simpan model beserta preprocessing pipeline yang lengkap"""
    model_data = {
        'model': model,
        'encoder': label_encoder,
        'model_type': 'OptimizedKNN',
        'classes': list(label_encoder.classes_),
        'distance_threshold': threshold,
        'best_params': best_params,
        'feature_processor': feature_processor,  # Simpan feature processor yang sudah di-fit
        'preprocessing_config': {
            'use_pca': USE_PCA,
            'pca_components': PCA_COMPONENTS,
            'use_feature_selection': USE_FEATURE_SELECTION,
            'feature_selection_k': FEATURE_SELECTION_K,
            'use_robust_scaling': USE_ROBUST_SCALING
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Optimized model dengan feature processor saved: {filename}")

def main():
    print("=== OPTIMIZED KNN FACE RECOGNITION TRAINING ===")
    print("Optimizations included:")
    print("- Advanced preprocessing pipeline")
    print("- Feature selection and PCA")
    print("- Robust scaling")
    print("- Grid search hyperparameter tuning")
    print("- Adaptive thresholding")
    print("- Correlation removal")
    print("- Feature processor yang bisa disimpan")
    print()
    
    # Load and preprocess data DENGAN feature processor
    features, labels_raw, class_counts, feature_processor = load_and_preprocess_data()
    if features is None:
        return
    
    # Check classes
    unique_classes = len(set(labels_raw))
    if unique_classes < 2:
        print(f"Error: Hanya {unique_classes} kelas ditemukan. KNN memerlukan minimal 2 kelas.")
        return
    
    print(f"\n{unique_classes} kelas ditemukan: {list(set(labels_raw))}")
    
    # Label encoding
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_raw)
    
    print(f"\nOptimized Dataset Info:")
    print(f"  Jumlah sampel: {len(features)}")
    print(f"  Jumlah kelas: {len(label_encoder.classes_)}")
    print(f"  Kelas: {list(label_encoder.classes_)}")
    print(f"  Dimensi fitur (setelah preprocessing): {features.shape[1]}")
    
    # Check data balance
    min_samples = min(class_counts.values())
    max_samples = max(class_counts.values())
    balance_ratio = min_samples / max_samples
    
    print(f"  Balance ratio: {balance_ratio:.2f}")
    if balance_ratio < 0.5:
        print("  WARNING: Data tidak seimbang. Pertimbangkan untuk menambah data.")
    
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
    
    # Train optimized KNN
    model, accuracy, threshold, best_params = train_optimized_knn(
        X_train, X_test, y_train, y_test, label_encoder
    )
    
    # Save model DENGAN feature processor
    save_optimized_model(model, label_encoder, threshold, best_params, feature_processor, MODEL_FILENAME)
    
    # Summary
    print(f"\n=== OPTIMIZED TRAINING SUMMARY ===")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Adaptive Threshold: {threshold:.4f}")
    print(f"Model Type: Optimized KNN")
    print("✓ Feature processor disimpan dalam model!")
    print("\nOptimizations applied successfully!")
    print("Model siap digunakan dengan performa yang ditingkatkan.")

if __name__ == "__main__":
    plt.style.use('default')
    main()