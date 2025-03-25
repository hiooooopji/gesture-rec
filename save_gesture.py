import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

# Storage for recorded gestures
recorded_gestures = {}

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

# Main loop
gesture_name = "Thumbs"  # Default
print("Press 'N' to change gesture name, 'S' to save coordinates, 'Q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Webcam read failed")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display gesture name
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save coordinates on 'S'
    if cv2.waitKey(1) & 0xFF == ord('s') and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            current_coords = extract_coordinates(hand_landmarks.landmark)
            current_coords = normalize_coordinates(current_coords)
            if gesture_name not in recorded_gestures:
                recorded_gestures[gesture_name] = []
            recorded_gestures[gesture_name].append(current_coords)
            print(f"Saved to '{gesture_name}' - Total frames: {len(recorded_gestures[gesture_name])}")

    # Change gesture name on 'N'
    if cv2.waitKey(1) & 0xFF == ord('n'):
        gesture_name = input("Enter new gesture name: ").strip() or "Thumbs"

    cv2.imshow("Save Gestures", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save to file
np.save("gestures.npy", recorded_gestures)
print("Gestures saved to 'gestures.npy'")

cap.release()
cv2.destroyAllWindows()
hands.close()