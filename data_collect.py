import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- KONFIGURASI ---
NAMA_ORANG = input("Masukkan nama Anda: ")  # Ganti dengan nama Anda
DATA_ROOT = "data_mp_wajah"
NUM_SAMPLES = 500  # Jumlah sampel yang akan dikumpulkan
DELAY = 0.1  # Delay antar capture (detik)

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Buat direktori jika belum ada
person_dir = os.path.join(DATA_ROOT, NAMA_ORANG)
os.makedirs(person_dir, exist_ok=True)

print(f"=== PENGUMPULAN DATA WAJAH ===")
print(f"Nama: {NAMA_ORANG}")
print(f"Target sampel: {NUM_SAMPLES}")
print(f"Direktori: {person_dir}")
print("\nTekan 'q' untuk berhenti, 's' untuk mulai capture")

cap = cv2.VideoCapture(0)
sample_count = 0
capturing = False
last_capture_time = 0

while cap.isOpened() and sample_count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi wajah
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Gambar landmark untuk feedback visual
        mp_drawing.draw_landmarks(
            frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS
        )
        
        # Ekstraksi fitur
        feature_vector = []
        for lm in landmarks:
            feature_vector.extend([lm.x, lm.y, lm.z])
        
        # Capture otomatis jika sedang capturing
        current_time = time.time()
        if capturing and (current_time - last_capture_time) > DELAY:
            # Simpan fitur
            filename = f"sample_{sample_count:04d}.npy"
            filepath = os.path.join(person_dir, filename)
            np.save(filepath, np.array(feature_vector))
            
            sample_count += 1
            last_capture_time = current_time
            print(f"Sampel {sample_count}/{NUM_SAMPLES} tersimpan: {filename}")
        
        # Status display
        status_text = f"Sampel: {sample_count}/{NUM_SAMPLES}"
        if capturing:
            status_text += " [CAPTURING]"
        else:
            status_text += " [READY - Tekan 's']"
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tidak ada wajah terdeteksi", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Data Collection - MediaPipe Face', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        capturing = not capturing
        print(f"Capturing: {'ON' if capturing else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print(f"\nPengumpulan data selesai! Total sampel: {sample_count}")