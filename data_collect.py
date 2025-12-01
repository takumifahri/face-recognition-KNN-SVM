import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- KONFIGURASI ---
NAMA_ORANG = input("Masukkan nama Anda: ")
DATA_ROOT = "data_mp_wajah"
NUM_SAMPLES = 200
DELAY = 0.1

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def extract_enhanced_features(landmarks):
    """
    ✅ FIXED: Consistent 1442 features
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
    
    # Ratios
    features.append(eye_distance / face_width if face_width > 0 else 0)
    features.append(mouth_width / face_width if face_width > 0 else 0)
    features.append(face_width / face_height if face_height > 0 else 0)
    
    # Symmetry
    left_eye_nose = np.sqrt((left_eye.x - nose_tip.x)**2 + (left_eye.y - nose_tip.y)**2)
    right_eye_nose = np.sqrt((right_eye.x - nose_tip.x)**2 + (right_eye.y - nose_tip.y)**2)
    symmetry_ratio = left_eye_nose / right_eye_nose if right_eye_nose > 0 else 1.0
    features.append(symmetry_ratio)
    
    return np.array(features)

# Buat direktori
person_dir = os.path.join(DATA_ROOT, NAMA_ORANG)
os.makedirs(person_dir, exist_ok=True)

print(f"=== ENHANCED DATA COLLECTION v2.0 ===")
print(f"Nama: {NAMA_ORANG}")
print(f"Target: {NUM_SAMPLES} samples")
print(f"Features: 1442 (478 landmarks × 3 + 8 geometric)")
print("\n[s] Start/Stop | [q] Quit")

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
    
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Draw
        mp_drawing.draw_landmarks(
            frame, 
            results.multi_face_landmarks[0], 
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
        
        # Extract
        feature_vector = extract_enhanced_features(landmarks)
        
        # Capture
        current_time = time.time()
        if capturing and (current_time - last_capture_time) > DELAY:
            filename = f"sample_{sample_count:04d}.npy"
            filepath = os.path.join(person_dir, filename)
            np.save(filepath, feature_vector)
            
            sample_count += 1
            last_capture_time = current_time
            print(f"✓ {sample_count}/{NUM_SAMPLES} | Features: {len(feature_vector)}")
        
        # Display
        status = f"{sample_count}/{NUM_SAMPLES}"
        if capturing:
            status += " [CAPTURING]"
            color = (0, 255, 0)
        else:
            status += " [READY]"
            color = (0, 165, 255)
        
        cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 70), color, 3)
        cv2.putText(frame, status, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 255), 2)
        cv2.putText(frame, "No face detected", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Data Collection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        capturing = not capturing
        print(f"Capturing: {'ON' if capturing else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ DONE! Total: {sample_count} samples | Features: 1442 each")