import os
import subprocess
from tqdm import tqdm

def query_gallery_matching(query_img_dataset, gallery_img_dataset, result_dir, model_paths):
    class_list = ['0', '1', '2', '3', '4', '5', '6']

    for query in tqdm(os.listdir(query_img_dataset), desc="Processing query folders"):
        for gallery in os.listdir(gallery_img_dataset):
            if gallery == query:
                continue

            for class_id in class_list:
                csv_sub_dir = os.path.join(result_dir, query)
                os.makedirs(csv_sub_dir, exist_ok=True)
                csv_save_path = os.path.join(csv_sub_dir, class_id)

                if class_id in ['0', '1', '3', '6']:
                    model_weight = model_paths[0]
                elif class_id == '2':
                    model_weight = model_paths[1]
                elif class_id == '4':
                    model_weight = model_paths[2]
                else:
                    model_weight = model_paths[3]

                # Construct the command
                command = [
                    "python", "test.py",
                    "-d", "veri",
                    "--resume", model_weight,
                    "--path_query", os.path.join(query_img_dataset, query),
                    "--path_gallery", os.path.join(gallery_img_dataset, gallery),
                    "--class_query", class_id,
                    "--class_save_path", csv_save_path
                ]

                # Execute the command
                try:
                    subprocess.run(command, check=True)
                    print(f"Completed matching for query: {query}, gallery: {gallery}, class: {class_id}")
                except subprocess.CalledProcessError as e:
                    print(f"Error running command for query: {query}, gallery: {gallery}, class: {class_id}")
                    print(f"Error details: {e}")

if __name__ == "__main__":
    # This block is for testing the module independently
    query_img_dataset = '/home/ashhar21137/submission_pipeline/datasets/images_query'
    gallery_img_dataset = '/home/ashhar21137/submission_pipeline/datasets/images_test'
    result_dir = '/home/ashhar21137/submission_pipeline/results_csv'
    model_paths = [
        '/models/class0136/checkpoint.pth.tar',
        '/models/class2/checkpoint.pth.tar',
        '/models/class4/checkpoint.pth.tar',
        '/models/class5/checkpoint.pth.tar'
    ]
    query_gallery_matching(query_img_dataset, gallery_img_dataset, result_dir, model_paths)