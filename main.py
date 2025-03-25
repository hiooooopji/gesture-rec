import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import threading

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Webcam not accessible")

# Storage for recorded gestures: {gesture_name: [frame1_coords, frame2_coords, ...]}
recorded_gestures = {}
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

# Video processing thread
def video_loop():
    while cap.isOpened() and root.winfo_exists():
        ret, frame = cap.read()
        if not ret:
            status_label.config(text="Error: Webcam failed")
            print("Webcam read failed")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        status_text = "No hand detected"
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                status_text = "Hand detected"

                current_coords = extract_coordinates(hand_landmarks.landmark)
                current_coords = normalize_coordinates(current_coords)

                if recognize_var.get():
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

        # Update status label
        status_label.config(text=status_text)

        # Convert frame to Tkinter format
        try:
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
        except Exception as e:
            print(f"Image conversion error: {e}")
            status_label.config(text="Error: Image display failed")

    print("Video loop ended")
    cap.release()

# Save coordinates function
def save_coordinates():
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture frame")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            current_coords = extract_coordinates(hand_landmarks.landmark)
            current_coords = normalize_coordinates(current_coords)

            gesture_name = name_entry.get().strip()
            if not gesture_name:
                gesture_name = "Thumbs"
            if gesture_name not in recorded_gestures:
                recorded_gestures[gesture_name] = []
            recorded_gestures[gesture_name].append(current_coords)
            status_label.config(text=f"Saved to '{gesture_name}' (#{len(recorded_gestures[gesture_name])})")
            print(f"Saved frame to '{gesture_name}' - Total frames: {len(recorded_gestures[gesture_name])})")
    else:
        messagebox.showwarning("Warning", "No hand detected to save!")

# UI Setup
root = tk.Tk()
root.title("Gesture Recognition")
root.geometry("800x600")
root.configure(bg="#2C3E50")

# Title
title_label = tk.Label(root, text="Hand Gesture Control", font=("Arial", 20, "bold"), 
                       fg="white", bg="#2C3E50")
title_label.pack(pady=10)

# Video feed
video_label = tk.Label(root, bg="#34495E")
video_label.pack(pady=10)

# Control Frame
control_frame = tk.Frame(root, bg="#2C3E50")
control_frame.pack(pady=10)

# Gesture Name Entry
tk.Label(control_frame, text="Gesture Name:", font=("Arial", 12), fg="white", bg="#2C3E50").grid(row=0, column=0, padx=5)
name_entry = tk.Entry(control_frame, font=("Arial", 12), width=15)
name_entry.insert(0, "Thumbs")
name_entry.grid(row=0, column=1, padx=5)

# Buttons
save_button = tk.Button(control_frame, text="Save Coordinates", font=("Arial", 12), 
                        command=save_coordinates, bg="#3498DB", fg="white", width=15)
save_button.grid(row=0, column=2, padx=5)

recognize_var = tk.BooleanVar()
recognize_button = tk.Checkbutton(control_frame, text="Recognize", font=("Arial", 12), 
                                  variable=recognize_var, bg="#2C3E50", fg="white", 
                                  selectcolor="#34495E", activebackground="#2C3E50", activeforeground="white")
recognize_button.grid(row=0, column=3, padx=5)

quit_button = tk.Button(control_frame, text="Quit", font=("Arial", 12), 
                        command=root.quit, bg="#95A5A6", fg="white", width=15)
quit_button.grid(row=0, column=4, padx=5)

# Status Label
status_label = tk.Label(root, text="Initializing...", font=("Arial", 12), fg="white", bg="#2C3E50")
status_label.pack(pady=10)

# Start video thread
threading.Thread(target=video_loop, daemon=True).start()

# Run the application
root.mainloop()

# Cleanup
cap.release()
hands.close()