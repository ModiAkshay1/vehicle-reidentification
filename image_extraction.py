import cv2
import pandas as pd
import os
from tqdm import tqdm
import json

def extract_images(config_path, csv_folder, output_base_folder):
    # Read the config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create the base output folder
    os.makedirs(output_base_folder, exist_ok=True)


    for cam_id, video_path in tqdm(config.items(), desc="Processing videos"):
        # Create a folder for this video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_folder = os.path.join(output_base_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Create folders for each class (0 to 6)
        for i in range(7):
            os.makedirs(os.path.join(video_output_folder, str(i)), exist_ok=True)

        # Find the corresponding CSV file
        csv_file = os.path.join(csv_folder, f"{video_name}.csv")
        print(f'csv_file : {csv_file}')
        if not os.path.exists(csv_file):
            print(f"CSV file not found for {video_name}, skipping.")
            continue

        df = pd.read_csv(csv_file)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            continue

        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting images from {video_name}"):
            frame_num = int(row['frame'])
            class_id = int(float(row['class_id']))
            x, y = int(row['x']), int(row['y'])
            width, height = int(row['width']), int(row['height'])
            tracklet_id = row['tracklet_id']

            # Set the video to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                print(f"Failed to extract frame {frame_num}")
                continue

            # Ensure coordinates are within frame boundaries
            h, w, _ = frame.shape
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            width = min(width, w - x)
            height = min(height, h - y)

            # Extract the region of interest (bounding box)
            bbox = frame[y:y + height, x:x + width]
            resize_dim = (224, 224)
            bbox = cv2.resize(bbox, resize_dim)

            # Check if the cropped image is valid
            if bbox.size == 0:
                print(f"Warning: Empty crop for frame {frame_num}, tracklet {tracklet_id}, skipping.")
                continue

            # Save the image in the respective class folder
            output_file = f"{tracklet_id}-{class_id}-{frame_num}-{video_name}.jpg"
            output_path = os.path.join(video_output_folder, str(class_id), output_file)
            cv2.imwrite(output_path, bbox)

        # Release the video capture
        cap.release()

    print("Image extraction completed.")

if __name__ == "__main__":
    # This block is for testing the module independently
    config_path = 'config.json'
    csv_folder = 'query_csv'
    output_base_folder = 'datasets/images_query'
    extract_images(config_path, csv_folder, output_base_folder)