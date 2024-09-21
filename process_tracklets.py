import pandas as pd
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

def calculate_bbox_area(row):
    """Convert (x, y, width, height) to (x_min, y_min, x_max, y_max) and calculate the area."""
    x_min = row['x']
    y_min = row['y']
    x_max = x_min + row['width']
    y_max = y_min + row['height']
    
    bbox_area = (x_max - x_min) * (y_max - y_min)
    return bbox_area

def process_tracklet(group, tracklet_id, n=1):
    # Step 1: Check if group has fewer than 3 rows
    if len(group) < 3:
        return pd.DataFrame()  # Return an empty DataFrame if fewer than 3 rows

    # Step 2: Group by class_name and calculate sample sizes for each class
    class_counts = group['class_name'].value_counts()
    
    # Step 3: Filter the group to only keep rows from the class with the max number of samples
    best_class_name = class_counts.idxmax()
    group = group[group['class_name'] == best_class_name]

    # Step 4: Calculate area for each row
    group['area'] = group.apply(calculate_bbox_area, axis=1)

    # Step 5: Sort the group by confidence and area in descending order
    sorted_group = group.sort_values(by=['confidence', 'area'], ascending=[False, False])

    # Step 6: Select the top 'n' rows
    selected_rows = sorted_group.head(n)

    # Step 7: Determine the best class name (highest confidence sum)
    best_class_name = selected_rows.groupby('class_name')['confidence'].sum().idxmax()

    # Step 8: Add the best class name to each row in the selected rows
    selected_rows['best_class_name'] = best_class_name
    selected_rows = selected_rows[selected_rows['confidence'] > 0.6]

    return selected_rows

def process_tracklets_query(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for input_file in tqdm(input_files, desc="Processing query tracklet files"):
        input_path = os.path.join(input_folder, input_file)
        df = pd.read_csv(input_path)

        # Process each tracklet_id
        processed_dfs = []
        for tracklet_id, group in df.groupby('tracklet_id'):
            processed_group = process_tracklet(group, tracklet_id, n=1)
            processed_dfs.append(processed_group)

        # Combine all processed data
        processed_df = pd.concat(processed_dfs)
        
        output_file = input_file.replace("_tracklet_info.csv", ".csv")
        output_path = os.path.join(output_folder, output_file)

        # Save the processed data to a new CSV file
        processed_df.to_csv(output_path, index=False)
        print(f"Processed query data saved to {output_path}")

def process_tracklets_gallery(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for input_file in tqdm(input_files, desc="Processing gallery tracklet files"):
        input_path = os.path.join(input_folder, input_file)
        df = pd.read_csv(input_path)

        # Process each tracklet_id
        processed_dfs = []
        for tracklet_id, group in df.groupby('tracklet_id'):
            processed_group = process_tracklet(group, tracklet_id, n=6)
            processed_dfs.append(processed_group)

        # Combine all processed data
        processed_df = pd.concat(processed_dfs)
        
        output_file = input_file.replace("_tracklet_info.csv", ".csv")
        output_path = os.path.join(output_folder, output_file)

        # Save the processed data to a new CSV file
        processed_df.to_csv(output_path, index=False)
        print(f"Processed gallery data saved to {output_path}")

if __name__ == "__main__":
    # This block is for testing the module independently
    input_folder = "original_tracklet"
    query_folder = "query_csv"
    gallery_folder = "gallery_csv"
    process_tracklets_query(input_folder, query_folder)
    process_tracklets_gallery(input_folder, gallery_folder)