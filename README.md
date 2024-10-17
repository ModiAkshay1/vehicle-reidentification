

# Video Tracklet Matching Pipeline

This repository contains a pipeline for video tracklet extraction and vehicle re-identification. The pipeline processes video data to match query images with gallery images and stores the results in a structured format.

## Setup

1. **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate <environment_name>
    ```

2. **Run the main script:**
    ```bash
    python main.py videos.json
    ```

## Pipeline Steps

### 1. Video Tracklet Extraction

- **Description:** Extracts tracklets from the provided videos.
- **Script:** `video_tracklet_extraction.py`

### 2. Process Tracklets

- **Description:** Processes the extracted tracklets to generate query and gallery CSV files.
- **Output:**
  - `query_csv/` - Contains query CSV files.
  - `gallery_csv/` - Contains gallery CSV files.
- **Script:** `process_tracklet.py`

### 3. Image Extraction

- **Description:** Extracts images from the query and gallery CSV files.
- **Output:**
  - `dataset/image_queries/` - Contains query images.
  - `datasets/images_test/` - Contains gallery images.
- **Script:** `image_extraction.py`

### 4. Matching Query Images with Gallery Images

- **Description:** Matches query images with gallery images for each vehicle class and generates a CSV with the matching results.
- **Script:** `main.py`

### 5. Count Vehicles Matched Across Cameras

- **Description:** Counts the number of vehicles matched across different cameras using the previously generated CSV.
- **Output:** Count matrix stored in `app/data/blank/matrix`
- **Script:** `main.py`

### 6. Extract and Store Matched Images

- **Description:** Extracts matched images based on the CSV results and stores them in a designated folder.
- **Output:** Matched images stored in `app/ data/blank/images`
- **Script:** `main.py`

## File Structure

- `environment.yml` - Conda environment setup file
- `main.py` - Main script to run the pipeline
- `video_tracklet_extraction.py` - Script for video tracklet extraction
- `process_tracklet.py` - Script for processing tracklets
- `image_extraction.py` - Script for image extraction
- `dataset/image_queries/` - Directory for query images
- `datasets/images_test/` - Directory for gallery images
- `app/data/blank/` - Directory for output data including matrix and images

## Notes

- Ensure that all scripts and dependencies are correctly configured before running the pipeline.
- Verify the output directories and files to ensure the pipeline has executed as expected.
- you can download pretrained models from this link https://drive.google.com/drive/folders/1Y3TNH2bFjCs15LTdCziIqvcTTaAIFLdv?usp=sharing , where it contains checkpoints for 7 different types of vehicle class (‘Cars’, ‘Bus’,
‘Truck’, ‘Three-Wheeler’, ‘Two-Wheeler’, ‘LCV’, and ‘Bicycle’).
