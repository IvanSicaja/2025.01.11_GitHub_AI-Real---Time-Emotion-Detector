import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from tensorflow.keras.models import load_model

# Load the trained TensorFlow model
model = load_model("emotion_model_99.83.h5")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Open video capture
cap = cv2.VideoCapture("1.0.1 Videos for testing/emotionMix_2.mp4")  # Change to filename if using a video file

# Emotion labels
label_map = {0: "Happy", 1: "Neutral", 2: "Sad"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for processing speed
    frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            landmark_data = []

            # Collect only the first 468 landmarks
            for i, lm in enumerate(landmarks.landmark):
                if i >= 468:  # Ignore extra landmarks
                    break
                landmark_data.append([lm.x, lm.y])  # Collect x, y coordinates

            # Ensure we have exactly 468 landmarks
            if len(landmark_data) == 468:
                # Convert landmarks to numpy array and reshape
                features = np.array(landmark_data)  # Shape (468, 2)
                features = np.expand_dims(features, axis=0)  # Add batch dimension (1, 468, 2)
                features = np.expand_dims(features, axis=2)  # Reshape to (1, 468, 1, 2)

                # Predict emotion
                output = model.predict(features)
                predicted_emotion = np.argmax(output, axis=1)[0]
                emotion_label = label_map.get(predicted_emotion, "Unknown")

                # Display result
                cv2.putText(frame_resized, f'Emotion: {emotion_label}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Emotion Recognition", frame_resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
