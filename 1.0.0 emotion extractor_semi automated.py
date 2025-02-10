import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime

# Function to display frames using OpenCV with a fixed resolution
def show_frame(frame):
    # Resize the frame to 1920x1080
    resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imshow('Video with Landmarks', resized_frame)

# Function to start the process, prompting user for the emotion label
def start_face_landmark_extraction(video_path, emotion):
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils  # For drawing the landmarks
    drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return  # Exit the function if video cannot be opened

    # Define columns for the Excel file, including emotion and landmarks
    columns = ['label'] + [f"Landmark_{i}_x" for i in range(468)] + [f"Landmark_{i}_y" for i in range(468)]

    # Try loading existing Excel file or create a new one
    try:
        landmark_df = pd.read_excel("landmarks.xlsx")
    except FileNotFoundError:
        landmark_df = pd.DataFrame(columns=columns)

    # Initialize frame counter
    frame_counter = 0

    # Loop through the video frames
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from video.")
            break  # Exit if frames are not being read

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        results = face_mesh.process(rgb_frame)

        # If landmarks are detected, save them along with the emotion
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Collect the emotion and landmarks
                landmark_data = [emotion]  # Start with the emotion in the first column

                for i, landmark in enumerate(landmarks.landmark):
                    landmark_data.append(landmark.x)  # X coordinate
                    landmark_data.append(landmark.y)  # Y coordinate

                print(f"Landmark data for frame {frame_counter}: {landmark_data[:10]}...")  # Debug print

                # Append the data for this frame to the dataframe
                landmark_df.loc[len(landmark_df)] = landmark_data

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # Optional: Show connections between landmarks
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

        # Show the frame with landmarks in OpenCV
        show_frame(frame)

        # Save the dataframe to Excel after every 10 frames to avoid excessive writes
        frame_counter += 1
        if frame_counter % 1000 == 0:
            landmark_df.to_excel("landmarks.xlsx", index=False)

        # Press 'q' to exit video playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save any remaining data to Excel
    landmark_df.to_excel("landmarks.xlsx", index=False)

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    print("Landmark extraction complete and saved to 'landmarks.xlsx'")

# Path to the video file
video_path = "1.0.0 Videos for training/happy_10.mp4"  # Replace with the path to your video file

# Specify the emotion
# Replace with:
# "happy"
# "sad"
# "neutral"
# etc.
emotion = "happy"

# Start the landmark extraction process
start_face_landmark_extraction(video_path, emotion)

