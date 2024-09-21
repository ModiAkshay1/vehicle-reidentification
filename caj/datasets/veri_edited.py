from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp
import os

from ..utils.data import BaseImageDataset

def extract_info_from_filename(filename):
    # Remove the file extension (.jpg)
    filename_wo_ext = os.path.splitext(filename)[0]
    
    # Split by underscore (_), assuming the format: video_name_trackletID_classID_frame_number.jpg
    parts = filename_wo_ext.split('-')
    
    # Extract the respective components
    video_name = parts[3]            # Video name
    tracklet_id = os.path.basename(parts[0])           # Tracklet ID
    class_id = parts[1]              # Class ID
    frame_number = parts[2]          # Frame number
    
    return video_name, int(tracklet_id), int(class_id), int(frame_number)

class VeRi(BaseImageDataset):
    """
    VeRi
    Reference:
    Liu, X., Liu, W., Ma, H., Fu, H.: Large-scale vehicle re-identification in urban surveillance videos. In: IEEE   %
    International Conference on Multimedia and Expo. (2016) accepted.
    Dataset statistics:
    # identities: 776 vehicles(576 for training and 200 for testing)
    # images: 37778 (train) + 11579 (query)
    """
    dataset_dir = 'VeRi'

    def __init__(self, root, verbose=True, **kwargs):
        super(VeRi, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.train_dir="/raid/home/akshay21166/bengaluru/all_dataset_2/"
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self.check_before_run()

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir2(self.query_dir, relabel=False)
        gallery = self.process_dir2(self.gallery_dir, relabel=False)

        if verbose:
            print('=> VeRi loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('"{}" is not available'.format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError('"{}" is not available'.format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError('"{}" is not available'.format(self.gallery_dir))

    def process_dir(self, dir_path, relabel=False):
        img_paths = []
        classes=["4"]
        for subfolder in os.listdir(dir_path):
            if subfolder not in classes:
                continue
            subfolder_path = osp.join(dir_path, subfolder)
            print(subfolder_path)
            if osp.isdir(subfolder_path):
                img_paths.extend(glob.glob(osp.join(subfolder_path, '*.jpg')))
        
        # Updated pattern to extract information from new filename format
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-.*\.jpg')

        # pid_container = set()
        # for img_path in img_paths:
        #     match = pattern.search(img_path)
        #     if match:
        #         tracklet_id, class_id, frame_number = map(int, match.groups())
        #         pid = class_id  # Assume class_id is the pid
        #         if pid == -1:
        #             continue  # Junk images are just ignored
        #         pid_container.add(pid)
        
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}
        video_container = set()
        for img_path in img_paths:
            video_name, tracklet_id, class_id, frame_number = extract_info_from_filename(img_path)
            video_container.add(video_name)
        video_container={value: index for index, value in enumerate(video_container, start=1)}
        dataset = []
        for img_path in img_paths:
            video_name, tracklet_id, class_id, frame_number = extract_info_from_filename(img_path)
            
            # if match:
            # tracklet_id, class_id, frame_number = map(int, match.groups())
            pid = tracklet_id  # Assume class_id is the pid
                # if pid == -1:
                #     continue  # Junk images are just ignored
                # camid = int(osp.basename(osp.dirname(img_path)))  # Extract camid from folder name
                # assert 0 <= pid <= 776  # pid == 0 means background
                # assert 0 <= camid <= 20  # Adjust if necessary
                # if relabel:
                #     pid = pid2label[pid]
            dataset.append((img_path, pid, video_container[video_name]))

        return dataset

    def process_dir2(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
