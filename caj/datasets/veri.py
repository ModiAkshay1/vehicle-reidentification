# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import glob
# import re
# import os.path as osp
# import os

# from ..utils.data import BaseImageDataset

# def extract_info_from_filename(filename):
#     # Remove the file extension (.jpg)
#     filename_wo_ext = os.path.splitext(filename)[0]
    
#     # Split by underscore (_), assuming the format: video_name_trackletID_classID_frame_number.jpg
#     parts = filename_wo_ext.split('-')
    
#     # Extract the respective components
#     video_name = parts[3]            # Video name
#     tracklet_id = os.path.basename(parts[0])           # Tracklet ID
#     class_id = parts[1]              # Class ID
#     frame_number = parts[2]          # Frame number
    
#     return video_name, int(tracklet_id), int(class_id), int(frame_number)

# class VeRi(BaseImageDataset):
#     """
#     VeRi
#     Reference:
#     Liu, X., Liu, W., Ma, H., Fu, H.: Large-scale vehicle re-identification in urban surveillance videos. In: IEEE   %
#     International Conference on Multimedia and Expo. (2016) accepted.
#     Dataset statistics:
#     # identities: 776 vehicles(576 for training and 200 for testing)
#     # images: 37778 (train) + 11579 (query)
#     """
#     dataset_dir = 'VeRi'

#     def __init__(self, root, verbose=True, **kwargs):
#         super(VeRi, self).__init__()
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.train_dir = osp.join(self.dataset_dir, 'image_train')
#         self.train_dir="/home/ashhar21137/all_new_dataset"
#         self.query_dir = osp.join(self.dataset_dir, 'image_query')
#         self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

#         self.check_before_run()

#         train = self.process_dir(self.train_dir, relabel=True)
#         query = self.process_dir2(self.query_dir, relabel=False)
#         gallery = self.process_dir2(self.gallery_dir, relabel=False)

#         if verbose:
#             print('=> VeRi loaded')
#             self.print_dataset_statistics(train, query, gallery)

#         self.train = train
#         self.query = query
#         self.gallery = gallery

#         self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
#         self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
#         self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

#     def check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
#         if not osp.exists(self.train_dir):
#             raise RuntimeError('"{}" is not available'.format(self.train_dir))
#         if not osp.exists(self.query_dir):
#             raise RuntimeError('"{}" is not available'.format(self.query_dir))
#         if not osp.exists(self.gallery_dir):
#             raise RuntimeError('"{}" is not available'.format(self.gallery_dir))

#     def process_dir(self, dir_path, relabel=False):
#         img_paths = []
#         classes=["0","1","3","6"]
#         for subfolder in os.listdir(dir_path):
#             if subfolder not in classes:
#                 continue
#             subfolder_path = osp.join(dir_path, subfolder)
#             if osp.isdir(subfolder_path):
#                 img_paths.extend(glob.glob(osp.join(subfolder_path, '*.jpg')))
        
#         # Updated pattern to extract information from new filename format
#         pattern = re.compile(r'(\d+)-(\d+)-(\d+)-.*\.jpg')

#         # pid_container = set()
#         # for img_path in img_paths:
#         #     match = pattern.search(img_path)
#         #     if match:
#         #         tracklet_id, class_id, frame_number = map(int, match.groups())
#         #         pid = class_id  # Assume class_id is the pid
#         #         if pid == -1:
#         #             continue  # Junk images are just ignored
#         #         pid_container.add(pid)
        
#         # pid2label = {pid: label for label, pid in enumerate(pid_container)}
#         video_container = set()
#         for img_path in img_paths:
#             video_name, tracklet_id, class_id, frame_number = extract_info_from_filename(img_path)
#             video_container.add(video_name)
#         video_container={value: index for index, value in enumerate(video_container, start=1)}
#         dataset = []
#         for img_path in img_paths:
#             video_name, tracklet_id, class_id, frame_number = extract_info_from_filename(img_path)
#             # if match:
#             # tracklet_id, class_id, frame_number = map(int, match.groups())
#             pid = tracklet_id  # Assume class_id is the pid
#                 # if pid == -1:
#                 #     continue  # Junk images are just ignored
#                 # camid = int(osp.basename(osp.dirname(img_path)))  # Extract camid from folder name
#                 # assert 0 <= pid <= 776  # pid == 0 means background
#                 # assert 0 <= camid <= 20  # Adjust if necessary
#                 # if relabel:
#                 #     pid = pid2label[pid]
#             dataset.append((img_path, pid, video_container[video_name]))

#         return dataset

#     def process_dir2(self, dir_path, relabel=False):
#         img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
#         pattern = re.compile(r'([-\d]+)_c([-\d]+)')

#         pid_container = set()
#         for img_path in img_paths:
#             pid, _ = map(int, pattern.search(img_path).groups())
#             if pid == -1:
#                 continue  # junk images are just ignored
#             pid_container.add(pid)
#         pid2label = {pid: label for label, pid in enumerate(pid_container)}

#         dataset = []
#         for img_path in img_paths:
#             pid, camid = map(int, pattern.search(img_path).groups())
#             if pid == -1:
#                 continue  # junk images are just ignored
#             assert 0 <= pid <= 776  # pid == 0 means background
#             assert 1 <= camid <= 20
#             camid -= 1  # index starts from 0
#             if relabel:
#                 pid = pid2label[pid]
#             dataset.append((img_path, pid, camid))

#         return dataset




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp
import os

from ..utils.data import BaseImageDataset

# def extract_info_from_filename(filename):
#     # Remove the file extension (.jpg)
#     filename_wo_ext = os.path.splitext(filename)[0]
#     print('filename : ', filename)
    
#     # Split by underscore (_), assuming the format: video_name_trackletID_classID_frame_number.jpg
#     parts = filename_wo_ext.split('-')
    
#     # Extract the respective components
#     video_name = parts[3]            # Video name
#     tracklet_id = os.path.basename(parts[0])           # Tracklet ID
#     class_id = parts[1]              # Class ID
#     frame_number = parts[2]          # Frame number
    
#     # print('INSIDE extract_info_from_filename')
#     print('video_name : ',video_name)
#     print('frame number : ',frame_number)
#     print(parts)
#     # print(f'video_name : {video_name} | tracklet_id : {tracklet_id} | class_id :  {int(class_id)} | frame number : {int(frame_number)}')
#     return video_name, int(tracklet_id), int(class_id), int(frame_number)

def extract_info_from_filename(filename):
    # Get just the filename without the full path
    base_filename = os.path.basename(filename)
    
    # Remove the file extension (.jpg)
    filename_wo_ext = os.path.splitext(base_filename)[0]
    # print('filename : ', filename)
    
    # Split by hyphen (-), assuming the format: trackletID-classID-frameNumber-videoName.jpg
    parts = filename_wo_ext.split('-')
    
    # Extract the respective components
    tracklet_id = parts[0]           # Tracklet ID
    class_id = parts[1]              # Class ID
    frame_number = parts[2]          # Frame number
    video_name = '-'.join(parts[3:]) # Video name (everything after the third hyphen)
    
    # print('video_name : ', video_name)
    # print('frame number : ', frame_number)
    # print(parts)
    
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
    dataset_dir = 'veri'

    def __init__(self, root, path_query, path_gallery, class_query, verbose=True, **kwargs):
        super(VeRi, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.dataset_dir = '/home/ashhar21137/bengaluru/custom_31'

        # print(f'self.dataset_dir : {self.dataset_dir}')

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        # print(f'self.train_dir : {self.train_dir}')
        
        self.train_dir="/home/ashhar21137/bengaluru/all_new_dataset"
        
        # print(f'self.train_dir : {self.train_dir}')

        # self.query_dir = '/home/ashhar21137/submission_pipeline/datasets/images_query/HP_Ptrl_Bnk_BEL_Rd_FIX_2_time_2024-05-31T07:30:02_000'
        self.query_dir = path_query


        # self.query_dir = osp.join(self.dataset_dir, 'image_query')
        # print(f' self.query_dir : { self.query_dir}')
        
        # self.gallery_dir = osp.join('/home/ashhar21137/submission_pipeline/datasets/images_test')

        # self.gallery_dir = '/home/ashhar21137/submission_pipeline/datasets/images_test/Sty_Wll_Ldge_FIX_3_time_2024-05-31T07:30:02_000'

        self.gallery_dir = path_gallery

        self.class_query = class_query 


        # self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        # print(f' self.gallery_dir : { self.gallery_dir}')


        # self.check_before_run()

        # train = self.process_dir(self.train_dir, relabel=True)

        query = self.process_dir(self.query_dir, relabel=False)

        gallery = self.process_dir(self.gallery_dir, relabel=False)

        # query = self.process_dir2(self.query_dir, relabel=False)
        # gallery = self.process_dir2(self.gallery_dir, relabel=False)

        # if verbose:
        #     print('=> VeRi loaded')
        #     self.print_dataset_statistics(train, query, gallery)
        if verbose:
            print('=> VeRi loaded')
            self.print_dataset_statistics_testonly(query, gallery)

        


        # self.train = train
        self.query = query
        self.gallery = gallery

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
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
        # classes=["4","2"]
        classes = self.class_query 
        for subfolder in os.listdir(dir_path):
            if subfolder not in classes:
                continue
            subfolder_path = osp.join(dir_path, subfolder)
            # print(subfolder_path)
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
            # print(f'img_path : {img_path}')
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
