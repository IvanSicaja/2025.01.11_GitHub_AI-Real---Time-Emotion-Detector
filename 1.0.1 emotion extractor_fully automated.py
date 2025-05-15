import cv2
import mediapipe as mp
import pandas as pd
import os

# Function to display frames using OpenCV with a fixed resolution
def show_frame(frame):
    resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.imshow('Video with Landmarks', resized_frame)

# Function to extract landmarks from a video
def process_video(video_path, emotion, landmark_df, face_mesh, drawing_spec):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return landmark_df

    frame_counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"Finished processing {video_path}.")
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                landmark_data = [emotion]

                for lm in landmarks.landmark:
                    landmark_data.append(lm.x)
                    landmark_data.append(lm.y)

                # Print row data with video name and frame number
                print(f"Video: {os.path.basename(video_path)} | Frame: {frame_counter} | Row Data: {landmark_data}")

                # Append the data to the DataFrame
                landmark_df.loc[len(landmark_df)] = landmark_data

                # Draw landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

        # Show the frame
        show_frame(frame)

        frame_counter += 1

        # Save every 1000 frames to avoid data loss
        if frame_counter % 1000 == 0:
            landmark_df.to_excel("landmarks_demo.xlsx", index=False)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    return landmark_df

# Main function to process all videos in the folder
def extract_landmarks_from_videos(folder_path):
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Drawing specification
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=3)

    # Define columns for the landmarks DataFrame
    columns = ['label'] + [f"Landmark_{i}_x" for i in range(468)] + [f"Landmark_{i}_y" for i in range(468)]

    # Try to load existing landmarks file, or create a new DataFrame
    try:
        landmark_df = pd.read_excel("landmarks.xlsx")
    except FileNotFoundError:
        landmark_df = pd.DataFrame(columns=columns)

    # Iterate over all video files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):  # Process only MP4 files
            video_path = os.path.join(folder_path, filename)
            emotion = filename.split("_")[0]  # Extract emotion from the filename
            print(f"Processing {video_path} with emotion '{emotion}'...")

            # Process the video and update the DataFrame
            landmark_df = process_video(video_path, emotion, landmark_df, face_mesh, drawing_spec)

    # Save the final DataFrame to an Excel file
    landmark_df.to_excel("landmarks_demo.xlsx", index=False)
    print("Landmark extraction complete. Data saved to 'landmarks.xlsx'.")

    # Close OpenCV windows
    cv2.destroyAllWindows()

# Path to the folder containing videos
videos_folder = "1.0.0 Videos for training"

# Start the process
extract_landmarks_from_videos(videos_folder)
