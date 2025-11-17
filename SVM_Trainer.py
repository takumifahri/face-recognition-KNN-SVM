import numpy as np
import os
import pickle
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# --- KONFIGURASI OPTIMIZED ---
DATA_ROOT = "data_mp_wajah"
SVM_MODEL_FILENAME = "face_svm_optimized_model.pkl"
ONECLASS_MODEL_FILENAME = "oneclass_svm_optimized_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# SVM Parameters for Grid Search
SVM_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

def load_and_analyze_data():
    """Memuat dan menganalisis data fitur wajah"""
    features = []
    labels_raw = []
    
    print("Memuat data fitur wajah...")
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Direktori '{DATA_ROOT}' tidak ditemukan!")
        print("Jalankan 'python data_collect.py' terlebih dahulu.")
        return None, None, None
    
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
        return None, None, None
    
    print(f"Total sampel dimuat: {total_files}")
    
    # Convert to numpy arrays
    features = np.array(features)
    labels_raw = np.array(labels_raw)
    
    # Data analysis
    print(f"\n=== DATA ANALYSIS ===")
    print(f"Feature shape: {features.shape}")
    print(f"Feature range: [{np.min(features):.6f}, {np.max(features):.6f}]")
    print(f"Feature mean: {np.mean(features):.6f}")
    print(f"Feature std: {np.std(features):.6f}")
    print(f"NaN values: {np.sum(np.isnan(features))}")
    print(f"Inf values: {np.sum(np.isinf(features))}")
    
    # Visualisasi distribusi data
    plot_data_distribution(class_counts)
    
    return features, labels_raw, class_counts

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
    plt.savefig('optimized_svm_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Optimized SVM data distribution plot saved as 'optimized_svm_data_distribution.png'")

def optimized_svm_grid_search(X_train, y_train):
    """Grid search untuk mencari hyperparameter optimal SVM"""
    print("Melakukan grid search untuk hyperparameter optimal SVM...")
    
    # GridSearchCV dengan cross-validation
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

def plot_optimized_svm_analysis(X_train, y_train, X_test, y_test, best_svm, best_params, scaler):
    """Analisis performa komprehensif untuk SVM"""
    print("Melakukan analisis performa komprehensif SVM...")
    
    # Test different C values dengan parameter optimal lainnya
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    train_accuracies = []
    test_accuracies = []
    cv_scores_mean = []
    
    optimal_params = best_params.copy()
    
    for c in c_values:
        optimal_params['C'] = c
        svm = SVC(**optimal_params, probability=True, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        
        # Training accuracy
        train_pred = svm.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_accuracies.append(train_acc)
        
        # Test accuracy
        test_pred = svm.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_accuracies.append(test_acc)
        
        # Cross validation
        cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores_mean.append(cv_scores.mean())
    
    # Plotting
    plt.figure(figsize=(18, 6))
    
    # Subplot 1: C vs Accuracy with optimized parameters
    plt.subplot(1, 3, 1)
    plt.semilogx(c_values, train_accuracies, 'bo-', label='Training Accuracy', linewidth=2)
    plt.semilogx(c_values, test_accuracies, 'ro-', label='Test Accuracy', linewidth=2)
    plt.semilogx(c_values, cv_scores_mean, 'go-', label='CV Accuracy', linewidth=2)
    
    best_c = best_params['C']
    plt.axvline(x=best_c, color='purple', linestyle='--', alpha=0.7, 
                label=f'Optimal C={best_c}')
    
    plt.title('Optimized SVM Performance vs C Value', fontweight='bold')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 2: Decision function analysis
    plt.subplot(1, 3, 2)
    
    # Get decision function scores for test samples
    decision_scores = best_svm.decision_function(X_test)
    
    # For multi-class, take max decision score for each sample
    if decision_scores.ndim > 1:
        max_decision_scores = np.max(decision_scores, axis=1)
    else:
        max_decision_scores = decision_scores
    
    plt.hist(max_decision_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(max_decision_scores), color='red', linestyle='-', 
                label=f'Mean: {np.mean(max_decision_scores):.3f}')
    plt.axvline(x=np.percentile(max_decision_scores, 25), color='orange', linestyle='--', 
                label=f'25th percentile: {np.percentile(max_decision_scores, 25):.3f}')
    
    plt.title('Decision Function Scores Distribution', fontweight='bold')
    plt.xlabel('Decision Function Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 3: Performance metrics comparison
    plt.subplot(1, 3, 3)
    
    # Calculate various metrics
    y_pred = best_svm.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, best_svm.predict(X_train))
    cv_acc = np.mean(cross_val_score(best_svm, X_train, y_train, cv=5))
    
    # Get probability scores for confidence analysis
    prob_scores = best_svm.predict_proba(X_test)
    max_prob_scores = np.max(prob_scores, axis=1)
    avg_confidence = np.mean(max_prob_scores)
    
    metrics = ['Train Acc', 'Test Acc', 'CV Acc', 'Avg Conf']
    values = [train_acc, test_acc, cv_acc, avg_confidence]
    colors = ['blue', 'red', 'green', 'orange']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Optimized SVM Performance Metrics', fontweight='bold')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimized_svm_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Optimized SVM performance analysis saved as 'optimized_svm_performance.png'")
    
    # Calculate and return adaptive threshold
    confidence_threshold = np.percentile(max_prob_scores, 15)  # Use 15th percentile as threshold
    
    return confidence_threshold, avg_confidence

def train_optimized_svm(X_train, X_test, y_train, y_test, label_encoder):
    """Melatih SVM yang dioptimasi"""
    print(f"\n=== OPTIMIZED SVM TRAINING ===")
    start_time = time.time()
    
    # Normalisasi data untuk SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search untuk parameter optimal
    best_svm, best_params = optimized_svm_grid_search(X_train_scaled, y_train)
    
    # Analisis performa
    confidence_threshold, avg_confidence = plot_optimized_svm_analysis(
        X_train_scaled, y_train, X_test_scaled, y_test, best_svm, best_params, scaler
    )
    
    # Final evaluation
    y_pred = best_svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=5)
    
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} detik")
    print(f"Best parameters: {best_params}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Score (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"Confidence threshold: {confidence_threshold:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Optimized SVM Confusion Matrix\nAccuracy: {accuracy:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('optimized_svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_svm, scaler, accuracy, confidence_threshold, best_params

def train_optimized_oneclass_svm(features, labels_raw):
    """Melatih One-Class SVM yang dioptimasi"""
    print(f"\n=== TRAINING OPTIMIZED ONE-CLASS SVM ===")
    print("Note: Hanya satu kelas terdeteksi, menggunakan One-Class SVM untuk anomaly detection")
    
    start_time = time.time()
    
    # Normalisasi data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Grid search for One-Class SVM
    param_grid = {
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'nu': [0.01, 0.05, 0.1, 0.2, 0.5],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    print("Grid search untuk One-Class SVM...")
    best_score = -np.inf
    best_params = {}
    best_model = None
    
    # Manual grid search for One-Class SVM (since it doesn't have scoring)
    for gamma in param_grid['gamma']:
        for nu in param_grid['nu']:
            for kernel in param_grid['kernel']:
                try:
                    model = OneClassSVM(gamma=gamma, nu=nu, kernel=kernel)
                    model.fit(features_scaled)
                    predictions = model.predict(features_scaled)
                    
                    # Score based on number of inliers (should be high but not 100%)
                    inlier_ratio = np.sum(predictions == 1) / len(predictions)
                    score = inlier_ratio if 0.7 <= inlier_ratio <= 0.95 else 0
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'gamma': gamma, 'nu': nu, 'kernel': kernel}
                        best_model = model
                except:
                    continue
    
    print(f"Best One-Class SVM parameters: {best_params}")
    
    # Final model with best parameters
    model = OneClassSVM(**best_params)
    model.fit(features_scaled)
    
    # Prediksi pada training data untuk evaluasi
    predictions = model.predict(features_scaled)
    decision_scores = model.decision_function(features_scaled)
    outliers = np.sum(predictions == -1)
    
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f} detik")
    print(f"Total sampel: {len(features)}")
    print(f"Outliers detected: {outliers} ({outliers/len(features)*100:.1f}%)")
    print(f"Average decision score: {np.mean(decision_scores):.4f}")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels_raw)
    
    # Plot One-Class SVM results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(decision_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='-', label='Decision Boundary')
    plt.axvline(x=np.mean(decision_scores), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(decision_scores):.3f}')
    plt.title('One-Class SVM Decision Scores')
    plt.xlabel('Decision Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    labels = ['Inliers', 'Outliers']
    sizes = [len(features) - outliers, outliers]
    colors = ['lightgreen', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Outlier Detection Results')
    
    plt.tight_layout()
    plt.savefig('optimized_oneclass_svm_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, label_encoder, scaler, best_params

def save_optimized_model(model, label_encoder, scaler, filename, model_type, **kwargs):
    """Simpan model dan komponen pendukung yang dioptimasi"""
    model_data = {
        'model': model,
        'encoder': label_encoder,
        'scaler': scaler,
        'model_type': model_type,
        'classes': list(label_encoder.classes_),
        **kwargs  # Additional parameters
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Optimized model {model_type} saved: {filename}")

def main():
    print("=== OPTIMIZED SVM FACE RECOGNITION TRAINING ===")
    print("Optimizations included:")
    print("- Advanced data analysis and visualization")
    print("- Grid search hyperparameter tuning")
    print("- Comprehensive performance analysis")
    print("- Adaptive confidence thresholding")
    print("- Statistical metrics and plots")
    print()
    
    # Load and analyze data
    features, labels_raw, class_counts = load_and_analyze_data()
    if features is None:
        return
    
    # Check number of unique classes
    unique_classes = len(set(labels_raw))
    
    if unique_classes == 1:
        print(f"\nHanya {unique_classes} kelas ditemukan: {list(set(labels_raw))}")
        print("Pilihan:")
        print("1. Kumpulkan data untuk orang lain")
        print("2. Lanjutkan dengan Optimized One-Class SVM")
        
        choice = input("\nPilih (1/2): ")
        if choice == "1":
            print("Kumpulkan data untuk orang lain menggunakan data_collect.py")
            return
        elif choice == "2":
            model, label_encoder, scaler, best_params = train_optimized_oneclass_svm(features, labels_raw)
            save_optimized_model(model, label_encoder, scaler, ONECLASS_MODEL_FILENAME, 
                               "OptimizedOneClassSVM", best_params=best_params)
            print(f"\nTraining Optimized OneClass SVM selesai!")
            print(f"Untuk testing: python test_recog.py")
        else:
            print("Pilihan tidak valid.")
        return
    
    # Multi-class SVM training
    print(f"\n{unique_classes} kelas ditemukan: {list(set(labels_raw))}")
    
    # Label encoding
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_raw)
    
    print(f"\nOptimized Dataset Info:")
    print(f"  Jumlah sampel: {len(features)}")
    print(f"  Jumlah kelas: {len(label_encoder.classes_)}")
    print(f"  Kelas: {list(label_encoder.classes_)}")
    print(f"  Dimensi fitur: {features.shape[1]}")
    
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
    
    # Train optimized SVM
    svm_model, scaler, svm_accuracy, confidence_threshold, best_params = train_optimized_svm(
        X_train, X_test, y_train, y_test, label_encoder
    )
    
    # Save model
    save_optimized_model(svm_model, label_encoder, scaler, SVM_MODEL_FILENAME, 
                        "OptimizedSVM", 
                        confidence_threshold=confidence_threshold,
                        best_params=best_params)
    
    # Summary
    print(f"\n=== OPTIMIZED TRAINING SUMMARY ===")
    print(f"Final Accuracy: {svm_accuracy:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Confidence Threshold: {confidence_threshold:.4f}")
    print(f"Model Type: Optimized SVM")
    print("\nOptimizations applied successfully!")
    print("Model siap digunakan dengan performa yang ditingkatkan.")

if __name__ == "__main__":
    plt.style.use('default')
    main()