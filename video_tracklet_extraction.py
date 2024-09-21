# import cv2
# import numpy as np
# from ultralytics import YOLO
# from sort import Sort
# import csv
# import os
# from tqdm import tqdm
# import json 
# import re 

# names = {0: 'Bicycle', 1: 'Bus', 2: 'Cars', 3: 'LCV', 4: 'Three-Wheeler', 5: 'Two-Wheeler', 6: 'Truck'}

# def process_video(video_path, output_folder):
#     # Load your fine-tuned YOLO model
#     model = YOLO('yolo5s_finetuned.pt')

#     # Initialize SORT tracker
#     mot_tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Define the output paths
#     video_name = os.path.basename(video_path)
#     csv_path = os.path.join(output_folder, f'{video_name}.csv')

#     # Open CSV file for writing
#     with open(csv_path, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(['tracklet_id', 'frame', 'class_id', 'class_name', 'x', 'y', 'width', 'height', 'confidence'])

#         # Initialize progress bar for this video
#         pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames")

#         frame_count = 0

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Run YOLO detection
#             results = model(frame, verbose=False)

#             # Extract bounding boxes, scores, and class IDs
#             boxes = results[0].boxes.xyxy.cpu().numpy()
#             scores = results[0].boxes.conf.cpu().numpy()
#             class_ids = results[0].boxes.cls.cpu().numpy()

#             # Prepare detections for SORT (x1, y1, x2, y2, score, class)
#             detections = np.column_stack((boxes, scores, class_ids))

#             # Update SORT tracker
#             tracked_objects = mot_tracker.update(detections)

#             # Write to CSV
#             for track in tracked_objects:
#                 x1, y1, x2, y2, track_id, _, _, _, obj_id = track
#                 x, y = int(x1), int(y1)
#                 w, h = int(x2 - x1), int(y2 - y1)
#                 class_id = int(track_id)
                
#                 # Find the corresponding detection for confidence score
#                 detection_idx = np.argmin(np.sum((boxes[:, :2] - [x1, y1])**2, axis=1))
#                 confidence = scores[detection_idx]

#                 csv_writer.writerow([int(obj_id), frame_count, class_id, names[class_id], x, y, w, h, confidence])

#             frame_count += 1
#             pbar.update(1)  # Update progress bar

#         pbar.close()  # Close the progress bar for this video

#     # Release everything
#     cap.release()

#     print(f"Processing complete for {video_path}")
#     print(f"Tracklet information saved in '{csv_path}'")

# def process_videos(config_path):
#     # Read config file
#     with open(config_path, 'r') as f:
#         config = json.load(f)
    
#     # Create output folder if it doesn't exist
#     output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_tracklet")
#     os.makedirs(output_folder, exist_ok=True)

#     # Initialize overall progress bar
#     overall_pbar = tqdm(total=len(config), desc="Overall Progress", unit="cameras")

#     # Process each video in the config
#     for cam_id, video_path in config.items():
#         video_name = os.path.basename(video_path)
#         video_name_only = re.split('[/.]',video_name)[-2]
#         csv_path = os.path.join(output_folder, f'{video_name}.csv')
        
#         # Check if CSV file already exists
#         if os.path.exists(csv_path):
#             print(f"CSV file for {video_name} already exists. Skipping processing.")
#             overall_pbar.update(1)
#             continue
        
#         process_video(video_path, output_folder)
#         overall_pbar.update(1)  # Update overall progress bar

#     overall_pbar.close()  # Close the overall progress bar

# if __name__ == "__main__":
#     # This block is for testing the module independently
#     config_path = 'config.json'
#     process_videos(config_path)

import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import pandas as pd  # Import pandas
import os
from tqdm import tqdm
import json
import re

names = {0: 'Bicycle', 1: 'Bus', 2: 'Cars', 3: 'LCV', 4: 'Three-Wheeler', 5: 'Two-Wheeler', 6: 'Truck'}

def process_video(video_path, output_folder):
    # Load your fine-tuned YOLO model
    yolo_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'yolo5s_finetuned.pt')
    model = YOLO(yolo_path)

    # Initialize SORT tracker
    mot_tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the output paths
    video_name = os.path.basename(video_path)
    video_name_only = re.split('[/.]',video_name)[-2]
    csv_path = os.path.join(output_folder, f'{video_name_only}.csv')

    # Initialize a list to collect rows for the DataFrame
    results_list = []

    # Initialize progress bar for this video
    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, verbose=False)

        # Extract bounding boxes, scores, and class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        # Prepare detections for SORT (x1, y1, x2, y2, score, class)
        detections = np.column_stack((boxes, scores, class_ids))

        # Update SORT tracker
        tracked_objects = mot_tracker.update(detections)

        # Collect results in the list
        for track in tracked_objects:
            x1, y1, x2, y2, track_id, _, _, _, obj_id = track
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            class_id = int(track_id)
            
            # Find the corresponding detection for confidence score
            detection_idx = np.argmin(np.sum((boxes[:, :2] - [x1, y1])**2, axis=1))
            confidence = scores[detection_idx]

            # Append the results to the list
            results_list.append([
                int(obj_id), frame_count, class_id, names[class_id], x, y, w, h, confidence
            ])

        frame_count += 1
        pbar.update(1)  # Update progress bar

    pbar.close()  # Close the progress bar for this video

    # Convert the list of results to a DataFrame
    df = pd.DataFrame(results_list, columns=['tracklet_id', 'frame', 'class_id', 'class_name', 'x', 'y', 'width', 'height', 'confidence'])

    # Write the DataFrame to CSV
    df.to_csv(csv_path, index=False)

    # Release everything
    cap.release()

    print(f"Processing complete for {video_path}")
    print(f"Tracklet information saved in '{csv_path}'")

def process_videos(config_path):
    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create output folder if it doesn't exist
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "original_tracklet")
    os.makedirs(output_folder, exist_ok=True)

    # Initialize overall progress bar
    overall_pbar = tqdm(total=len(config), desc="Overall Progress", unit="cameras")

    # Process each video in the config
    for cam_id, video_path in config.items():
        video_name = os.path.basename(video_path)
        video_name_only = re.split('[/.]',video_name)[-2]
        print(f'video_name_only : {video_name_only}')
        csv_path = os.path.join(output_folder, f'{video_name_only}.csv')
        print(f'csv_path : {csv_path}')
        
        # Check if CSV file already exists
        if os.path.exists(csv_path):
            print(f"CSV file for {video_name} already exists. Skipping processing.")
            overall_pbar.update(1)
            continue
        
        process_video(video_path, output_folder)
        overall_pbar.update(1)  # Update overall progress bar

    overall_pbar.close()  # Close the overall progress bar

if __name__ == "__main__":
    # This block is for testing the module independently
    config_path = 'config.json'
    process_videos(config_path)
