import os
import json
import argparse
from tqdm import tqdm 
from video_tracklet_extraction import process_videos
from process_tracklets import process_tracklets_query, process_tracklets_gallery
from image_extraction import extract_images
from match_query_gallery import query_gallery_matching
import subprocess
import numpy as np 
import pandas as pd 
import cv2 
import json 
import re 

def extract_frame_from_video(id,video_path, tracklet_id, frame_number, class_name, x, y, width, height, cam_number, output_filename):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Failed to extract frame {frame_number} from {video_path}")
        return

    # Draw the bounding box
    label = f"{class_name}_{cam_number}_{frame_number}_{tracklet_id}"
    bbox_color = (0, 255, 0)  # Green bounding box
    cv2.rectangle(frame, (x, y), (x + width, y + height), bbox_color, 2)

    # Set a larger font size and thickness for the label
    font_scale = 0.8
    font_thickness = 2

    # Get the text size to create a background for the label
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_width, text_height = text_size

    # Set the text background position
    label_x1 = x
    label_y1 = y - text_height - 10
    label_x2 = x + text_width
    label_y2 = y

    # Draw a filled rectangle for the text background
    text_bg_color = (0, 0, 0)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), text_bg_color, -1)

    # Add the label text on top of the rectangle
    text_color = (255, 255, 255)
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # Save the frame with the bounding box and label
    cv2.imwrite(output_filename, frame)
    cap.release()
    print(f"Saved {output_filename}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process videos for tracklet extraction, post-processing, and image extraction.')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file')
    parser.add_argument('output_path', type=str, help='Team name')
    args = parser.parse_args()
    file_dir=os.path.realpath(__file__)

    # Get the config file path from command line argument
    config_path = args.config

    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return

    print(f'args : {args}')

    # Step 1: Video Tracklet Extraction
    print(" ---------- Step 1: Starting Video Tracklet Extraction ---------- ")
    process_videos(config_path)
    print(" ---------- Step 1: Video Tracklet Extraction Completed ---------- ")

    # Step 2: Process Tracklets for Query
    print(" ---------- Step 2: Starting Tracklet Processing for Query ---------- ")
    input_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "original_tracklet")
    query_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "query_csv")
    gallery_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gallery_csv")

    process_tracklets_query(input_folder, query_folder)
    print(" ---------- Step 2: Tracklet Processing for Query Completed ---------- ")

    # Step 3: Process Tracklets for Gallery
    print(" ---------- Step 3: Starting Tracklet Processing for Gallery ---------- ")
    # gallery_folder = "gallery_csv"
    process_tracklets_gallery(input_folder, gallery_folder)
    print(" ---------- Step 3: Tracklet Processing for Gallery Completed ---------- ")

    # Step 4a: Image Extraction for Query
    print(" ---------- Step 4a: Starting Image Extraction for Query ---------- ")
    output_base_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/images_query')
    extract_images(config_path, query_folder, output_base_folder)
    print(" Step 4a: Image Extraction for Query Completed ---------- ")

    # Step 4b: Image Extraction for Gallery
    print(" ---------- Step 4b: Starting Image Extraction for Gallery ---------- ")
    output_base_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/images_test')
    extract_images(config_path, gallery_folder, output_base_folder)
    print(" Step 4b: Image Extraction for Query Completed ---------- ")

    # Step 5: Image Extraction for Gallery
    print(" ---------- Step 5: Starting Query and Gallery Matching ---------- ")
    class_list = ['0','1','2','3','4','5','6']
    query_img_dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/images_query')
    gallery_img_dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets/images_test')
    result_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results_csv')
    model_paths = ['models/class0136/checkpoint.pth.tar','models/class2/checkpoint.pth.tar','models/class4/checkpoint.pth.tar', 'models/class5/checkpoint.pth.tar' ]
    for i in range(len(model_paths)):
        model_paths[i]=os.path.join(os.path.dirname(os.path.realpath(__file__)), model_paths[i])

    for query in tqdm(os.listdir(query_img_dataset)) : 
        for gallery in os.listdir(gallery_img_dataset) : 
            if gallery == query : 
                continue 
            for class_id in class_list : 
                csv_sub_dir = os.path.join(result_dir, query)
                os.makedirs(csv_sub_dir, exist_ok=True)
                csv_save_path = os.path.join(csv_sub_dir,class_id)
                os.makedirs(csv_save_path, exist_ok=True)
                print(f'###### csv_save_path : {csv_save_path}')
                
                csv_save_path_final = f'{csv_save_path}/{gallery}'
                print(f' ******* csv_save_path : {csv_save_path_final}')

                if class_id == '0' or class_id == '1' or class_id == '3' or class_id == '6' :
                    model_weight = model_paths[0]
                elif class_id == '2' : 
                    model_weight = model_paths[1]
                elif class_id == '4' : 
                    model_weight = model_paths[2]
                else : 
                    model_weight = model_paths[3]

                gallery_dir = os.path.join(gallery_img_dataset,gallery)
                query_dir = os.path.join(query_img_dataset,query)
                test_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"test.py")

                # Construct the command
                command = [
                    "python3", test_file,
                    "-d", "veri",
                    "--resume", model_weight,
                    "--path_query", os.path.join(query_img_dataset, query_dir),
                    "--path_gallery", os.path.join(gallery_img_dataset, gallery_dir),
                    "--class_query", class_id,
                    "--class_save_path", csv_save_path_final
                ]

                print(command)

                # Execute the command
                try:
                    subprocess.run(command, check=True)
                    print(f"Completed matching for query: {query}, gallery: {gallery}, class: {class_id}")
                except subprocess.CalledProcessError as e:
                    print(f"Error running command for query: {query}, gallery: {gallery}, class: {class_id}")
                    print(f"Error details: {e}")

    
    print(" ---------- Step 5: Starting Query and Gallery Matching Completed ---------- ")

    print(" ---------- Step 6: Count Matrix generation and saving  ---------- ")
  
    class_list_names = {'0': "Bicycle", '1': "Bus", '2': "Car", '3': "LCV", '4': "Three-Wheeler", '5': "Two-Wheeler", '6': "Truck"}
    with open(config_path, 'r') as f:
        config = json.load(f)

    config2 = dict()

    for key in config.keys():
        name = config[key]
        namef = re.split('[/.]', name)[-2]
        config2[key] = namef

    print(config)
    print()
    print(config2)
    

    cam_names = list(sorted(config2.keys()))
    cam_names_dict = {cam_name: index for index, cam_name in enumerate(cam_names)}
    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results_csv')
    # save_path = os.path.join(os.getcwd(), 'data/[blank]')
    save_path = os.path.join('app/data',args.output_path)
    # save_path = os.path.join('data',args.output_path)
    # save_path=args.output_path
    os.makedirs(save_path, exist_ok=True)
    save_path_matrix = os.path.join(save_path, "Matrices")
    save_path_images = os.path.join(save_path, "Images")
    query_paths = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'query_csv')
    gallery_paths = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gallery_csv')
    os.makedirs(save_path_matrix, exist_ok=True)
    os.makedirs(save_path_images, exist_ok=True)

    for class_number in class_list_names.keys():
        n = len(cam_names)
        
        # Initialize the array with integers
        array = np.zeros((n, n), dtype=int)
        
        class_name = class_list_names[class_number]
        class_name_file = class_name + ".json"
        class_name_file = os.path.join(save_path_matrix, class_name_file) 

        for query in cam_names:
            for gallery in cam_names:
                if query != gallery:
                    csv_file = config2[query] + "/" + class_number + "/" + config2[gallery] + ".csv"
                    csv_file_path = os.path.join(base_dir, csv_file)

                    print(f"csv_file_path : {csv_file_path}")
                    if not os.path.exists(csv_file_path):
                        print('-----------------')
                        continue

                    df = pd.read_csv(csv_file_path)
                    # print(f'df : {df}')
                    total_rows = len(df)
                    
                    # Assign integer value
                    array[cam_names_dict[query]][cam_names_dict[gallery]] = total_rows

        print(f'array : {array}')
        
        # Convert the integer array to a list and save it in JSON format
        array_list = array.tolist()
        with open(class_name_file, 'w') as json_file:
            json.dump(array_list, json_file)

    print(" ---------- Step 6: Count Matrix generation and saving completed ---------- ")


    print(' ---------- Step 7 : Image Extraction ----------  ')
    for class_number in class_list:
        class_name=class_list_names[class_number]
        curr_save=os.path.join(save_path_images,class_name)
        os.makedirs(curr_save, exist_ok=True)
        total_rows=0
        for query in cam_names:
            for gallery in cam_names:
                if query not in gallery:
                    csv_file=config2[query]+"/"+class_number+"/"+config2[gallery]+".csv"
                    print(f'csv_file : {csv_file}')
                    csv_file_path = os.path.join(base_dir, csv_file)
                    if not os.path.exists(csv_file_path):
                        print('------------')
                        continue
                    results_df = pd.read_csv(csv_file_path)
                    results_csv = csv_file_path 
                    query_csv_folder = query_paths
                    gallery_csv_folder = gallery_paths
                    output_dir = curr_save
                    for index, row in results_df.iterrows():
                        query_image_path = row['Query Image']
                        gallery_image_path = row['Gallery Image']

                        # Extract video name, tracklet ID, class ID, and frame number from query image path
                        query_image_parts = query_image_path.split('/')[-1].split('-')
                        # query_tracklet_id = total_rows+index  # Tracklet ID
                        query_tracklet_id = int(query_image_parts[0])  # Tracklet ID
                        # query_class_id = int(query_image_parts[1])  # Class ID
                        query_id = total_rows+index  # Class ID
                        query_frame_number = int(query_image_parts[2])  # Frame number
                        query_video_name = '-'.join(query_image_parts[3:]).replace('.jpg', '')  # Remove '.jpg' from the video name

                        # Extract video name, tracklet ID, class ID, and frame number from gallery image path
                        gallery_image_parts = gallery_image_path.split('/')[-1].split('-')
                        gallery_tracklet_id = int(gallery_image_parts[0])  # Tracklet ID
                        # gallery_class_id = total_rows+index  # Class ID
                        gallery_id = total_rows+index  # Class ID
                        gallery_frame_number = int(gallery_image_parts[2])  # Frame number
                        gallery_video_name = '-'.join(gallery_image_parts[3:]).replace('.jpg', '')  # Remove '.jpg' from the video name

                        # Load the corresponding CSV files
                        query_csv_path = os.path.join(query_csv_folder, f"{query_video_name}.csv")
                        gallery_csv_path = os.path.join(gallery_csv_folder, f"{gallery_video_name}.csv")

                        # Ensure that the CSV file paths are correct
                        if not os.path.exists(query_csv_path):
                            print(f"Query CSV file not found: {query_csv_path}")
                            continue
                        if not os.path.exists(gallery_csv_path):
                            print(f"Gallery CSV file not found: {gallery_csv_path}")
                            continue

                        query_df = pd.read_csv(query_csv_path)
                        gallery_df = pd.read_csv(gallery_csv_path)

                        # Find the matching rows in query_csv and gallery_csv
                        query_row = query_df[query_df['tracklet_id'] == query_tracklet_id].iloc[0]
                        gallery_row = gallery_df[gallery_df['tracklet_id'] == gallery_tracklet_id].iloc[0]

                        # Extract bounding box info for query and gallery
                        query_x, query_y, query_width, query_height = query_row['x'], query_row['y'], query_row['width'], query_row['height']
                        gallery_x, gallery_y, gallery_width, gallery_height = gallery_row['x'], gallery_row['y'], gallery_row['width'], gallery_row['height']

                        # Define camera numbers (modify this based on how you extract cam number)
                        query_cam_number = query  # You can extract this from query_video_name or set it as needed
                        gallery_cam_number = gallery

                        # Extract query frame and save
                        # query_video_path = f"/path/to/query_videos/{query_video_name}.mp4"  # Adjust this path
                        print(f'class name : {class_name}')
                        query_video_path = config[query]  
                        query_output_filename = f"{class_name}_{query_cam_number}_{query_frame_number}_{query_id}.jpg"
                        extract_frame_from_video(query_id,query_video_path, query_tracklet_id, query_frame_number, class_name, query_x, query_y, query_width, query_height, query_cam_number, os.path.join(output_dir, query_output_filename))

                        # Extract gallery frame and save
                        # gallery_video_path = f"/path/to/gallery_videos/{gallery_video_name}.mp4"  # Adjust this path
                        gallery_video_path = config[gallery]  
                        gallery_output_filename = f"{class_name}_{gallery_cam_number}_{gallery_frame_number}_{gallery_id}.jpg"
                        extract_frame_from_video(gallery_id,gallery_video_path, gallery_tracklet_id, gallery_frame_number, class_name, gallery_x, gallery_y, gallery_width, gallery_height, gallery_cam_number, os.path.join(output_dir, gallery_output_filename))
 

    print("All processes completed successfully!")

if __name__ == "__main__":
    main()