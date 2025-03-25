import cv2
import mediapipe as mp
import csv
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
recording = False
frame_count = 0
data_file = "gesture_data.csv"

# Open CSV file once, avoid reopening every frame
csv_file = open(data_file, "w", newline="")
csv_writer = csv.writer(csv_file)

def extract_features(hand_landmarks):
    """Compute normalized distances between landmarks"""
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    base = landmarks[0]  # Use wrist as reference
    normalized_landmarks = landmarks - base  

    distances = np.linalg.norm(normalized_landmarks[:, None] - normalized_landmarks, axis=2)
    return distances[np.triu_indices(21, k=1)].tolist()

print("📢 Press 'C' to start/stop recording. Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Detect hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show status
        status_text = "Recording..." if recording else "Press 'C' to start recording"
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0) if recording else (255, 255, 255), 2, cv2.LINE_AA)

        if recording:
            frame_count += 1
            if frame_count % 20 == 0:  # Record every 20 frames
                for hand_landmarks in results.multi_hand_landmarks:
                    distances = extract_features(hand_landmarks)
                    csv_writer.writerow(distances)

    else:
        cv2.putText(frame, "No Hand Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recording", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        recording = not recording  # Toggle recording mode
        print("✅ Recording Started" if recording else "❌ Recording Stopped")
    
    if key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
