import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Handsape Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

# Load saved gestures
recorded_gestures = np.load("gestures.npy", allow_pickle=True).item()
DISTANCE_THRESHOLD = 0.7

# Coordinate functions
def extract_coordinates(landmarks):
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)

def normalize_coordinates(coords):
    coords = coords.reshape(-1, 3)
    centroid = np.mean(coords, axis=0)
    coords -= centroid
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()

def calculate_distance(current_coords, recorded_coords):
    return np.linalg.norm(current_coords - recorded_coords)

# Main loop
print("Press 'Q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Webcam read failed")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    status_text = "No hand detected"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_coords = extract_coordinates(hand_landmarks.landmark)
            current_coords = normalize_coordinates(current_coords)

            best_match = None
            min_distance = float('inf')
            for gesture_name, recorded_frames in recorded_gestures.items():
                for recorded_coords in recorded_frames:
                    norm_recorded = normalize_coordinates(recorded_coords)
                    distance = calculate_distance(current_coords, norm_recorded)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = gesture_name
            if min_distance < DISTANCE_THRESHOLD and best_match:
                status_text = f"Detected: {best_match} ({min_distance:.2f})"
            else:
                status_text = f"No Match ({min_distance:.2f})"

    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Recognize Gestures", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()