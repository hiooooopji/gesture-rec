import cv2
import mediapipe as mp
import numpy as np
import keyboard

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
recorded_gestures = {}

def extract_coordinates(landmarks):
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)

print("Hold 'R' to record a gesture. Enter its name when prompted. Press 'Q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if keyboard.is_pressed('r'):
                coords = extract_coordinates(hand_landmarks.landmark)
                if 'temp_gesture' not in locals():
                    temp_gesture = []
                    print("Recording gesture... Release 'R' to finish.")
                temp_gesture.append(coords)
            elif 'temp_gesture' in locals():
                gesture_name = input("Enter gesture name: ")
                recorded_gestures[gesture_name] = temp_gesture
                print(f"Recorded '{gesture_name}' with {len(temp_gesture)} frames.")
                del temp_gesture

    cv2.imshow("Record Gestures", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

np.save('recorded_gestures.npy', recorded_gestures)
print("Gestures saved to 'recorded_gestures.npy'")

cap.release()
cv2.destroyAllWindows()
hands.close()