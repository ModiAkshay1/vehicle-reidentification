from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import os
import csv
import copy
from collections import Counter
from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    # mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    # print('Mean AP: {:4.1%}'.format(mAP))

    # if (not cmc_flag):
    #     return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    # return cmc_scores['market1501'], mAP
    return cmc_scores['market1501']



import os
import csv
import numpy as np


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
        

    def extract_tracklet_class_frame_id(self, filename):
        """
        Extract tracklet ID, class ID, and frame number from filenames like:
        '13-4-257-HP_Ptrl_Bnk_BEL_Rd_FIX_2_time_2024-05-31T07:30:02_008.jpg'
        - 13 is the tracklet ID (first part)
        - 4 is the class ID (second part)
        - 257 is the frame number (third part)
        """
        base_filename = os.path.basename(filename)
        parts = base_filename.split('-')
        tracklet_id = parts[0]  # Extract tracklet ID (first part)
        class_id = parts[1]      # Extract class ID (second part)
        frame_id = int(parts[2])  # Extract frame number (third part)
        return tracklet_id, class_id, frame_id

    def extract_cnn_feature(self, model, inputs):
        inputs = to_torch(inputs).cuda()
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs

    def extract_features(self, model, data_loader, print_freq=50):
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        labels = OrderedDict()

        end = time.time()
        with torch.no_grad():
            for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
                data_time.update(time.time() - end)

                outputs = self.extract_cnn_feature(model, imgs)
                for fname, output, pid in zip(fnames, outputs, pids):
                    features[fname] = output
                    labels[fname] = pid

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print('Extract Features: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          .format(i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

        return features, labels

    def pairwise_distance(self, features, query=None, gallery=None):
        if query is None and gallery is None:
            n = len(features)
            x = torch.cat(list(features.values()))
            x = x.view(n, -1)
            dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
            dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
            return dist_m

        x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_m.addmm_(1, -2, x, y.t())
        return dist_m, x.numpy(), y.numpy()

    def evaluate_all(self, query_features, gallery_features, distmat, query=None, gallery=None,
                     query_ids=None, gallery_ids=None,
                     query_cams=None, gallery_cams=None,
                     cmc_topk=(1, 5, 10), cmc_flag=False):
        if query is not None and gallery is not None:
            query_ids = [pid for _, pid, _ in query]
            gallery_ids = [pid for _, pid, _ in gallery]
            query_cams = [cam for _, _, cam in query]
            gallery_cams = [cam for _, _, cam in gallery]
        else:
            assert (query_ids is not None and gallery_ids is not None
                    and query_cams is not None and gallery_cams is not None)

        cmc_configs = {
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True),}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        print('CMC Scores:')
        for k in cmc_topk:
            print(f'top-{k:<4}{cmc_scores["market1501"][k - 1]:12.1%}')
        return cmc_scores['market1501']

    def evaluate(self, data_loader, query, gallery, args=None, cmc_flag=False, rerank=False, output_csv='results.csv'):
        # Extract features
        features, _ = self.extract_features(self.model, data_loader)
        # class_query = args.class_query
        csv_save_path = args.class_save_path
        # print(f' in evaluate class_query : {class_query}')
        # print(f' in evaluate csv_save_path : {csv_save_path}')

        # Compute pairwise distance matrix
        distmat, query_features, gallery_features = self.pairwise_distance(features, query, gallery)

        # Create dictionaries to store tracklet IDs, class IDs, and frame numbers for query and gallery
        query_tracklets = {}
        query_classes = {}
        query_frames = {}
        gallery_tracklets = {}
        gallery_classes = {}
        gallery_frames = {}

        # Populate tracklet, class IDs, and frame numbers for the query images
        for idx, (fname, pid, camid) in enumerate(query):
            try:
                tracklet_id, class_id, frame_id = self.extract_tracklet_class_frame_id(fname)
                query_tracklets[fname] = tracklet_id
                query_classes[fname] = class_id
                query_frames[fname] = frame_id
            except Exception as e:
                print(f"Error extracting tracklet/class/frame ID for query {fname}: {e}")
                continue

        # Populate tracklet, class IDs, and frame numbers for the gallery images
        for idx, (fname, pid, camid) in enumerate(gallery):
            try:
                tracklet_id, class_id, frame_id = self.extract_tracklet_class_frame_id(fname)
                gallery_tracklets[fname] = tracklet_id
                gallery_classes[fname] = class_id
                gallery_frames[fname] = frame_id
            except Exception as e:
                print(f"Error extracting tracklet/class/frame ID for gallery {fname}: {e}")
                continue

        # Sort each row of the distance matrix in ascending order and get indices
        sorted_indices = np.argsort(distmat, axis=1)

        # Create and open CSV file for writing
        # csv_file_path = os.path.abspath(output_csv)  # Get the absolute path of the CSV file
        with open(f'{csv_save_path}.csv', mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write header
            writer.writerow(['Query Tracklet ID', 'Query Class ID', 'Query Image',
                             'Selected Gallery Tracklet ID', 'Gallery Class ID', 
                             'Gallery Image', 'Distance'])

            # Loop over each query
            for i in range(len(query)):
                query_fname = query[i][0]
                query_tracklet_id = query_tracklets.get(query_fname, None)
                query_class_id = query_classes.get(query_fname, None)
                query_frame_id = query_frames.get(query_fname, None)

                if query_tracklet_id is None or query_class_id is None or query_frame_id is None:
                    print(f"Warning: Tracklet/Class/Frame ID not found for query {query_fname}")
                    continue  # Skip to the next query

                # Get the top 10 gallery indices for this query
                top_10_indices = sorted_indices[i][:10]

                # Filter the top 10 gallery matches by class and frame number
                gallery_id_counts = Counter()
                gallery_id_distances = {}

                for rank_idx in top_10_indices:
                    gallery_fname = gallery[rank_idx][0]
                    gallery_tracklet_id = gallery_tracklets.get(gallery_fname, None)
                    gallery_class_id = gallery_classes.get(gallery_fname, None)
                    gallery_frame_id = gallery_frames.get(gallery_fname, None)

                    # Ensure the gallery image is from the same class as the query
                    # and that the gallery frame number is greater than the query frame number
                    if (gallery_tracklet_id is None or gallery_class_id is None or 
                            gallery_class_id != query_class_id or gallery_frame_id <= query_frame_id):
                        continue  # Skip if class ID does not match or frame number is not greater

                    # Update count of how many times this gallery tracklet ID appears in the top 10
                    gallery_id_counts[gallery_tracklet_id] += 1

                    # Store the minimum distance for each gallery tracklet ID
                    distance = distmat[i, rank_idx].item()
                    if gallery_tracklet_id not in gallery_id_distances:
                        gallery_id_distances[gallery_tracklet_id] = distance
                    else:
                        gallery_id_distances[gallery_tracklet_id] = min(gallery_id_distances[gallery_tracklet_id], distance)

                # Find gallery tracklet IDs that appear 3 or more times
                candidates = [gid for gid, count in gallery_id_counts.items() if count >= 3]

                if candidates:
                    # If multiple gallery tracklet IDs have count >= 3, choose the one with the smallest distance
                    selected_gallery_id = min(candidates, key=lambda gid: gallery_id_distances[gid])

                    # Find the gallery image corresponding to the selected gallery tracklet ID with the smallest distance
                    for rank_idx in top_10_indices:
                        gallery_fname = gallery[rank_idx][0]
                        gallery_tracklet_id = gallery_tracklets.get(gallery_fname, None)

                        if gallery_tracklet_id == selected_gallery_id:
                            selected_gallery_image = gallery_fname
                            selected_distance = gallery_id_distances[selected_gallery_id]
                            break

                    # Write the selected match to the CSV
                    writer.writerow([query_tracklet_id, query_class_id, query_fname,
                                     selected_gallery_id, gallery_class_id, selected_gallery_image, selected_distance])
                else:
                    print(f"No gallery tracklet with count >= 3 found for query {query_fname}")

        print(f"CSV file generated successfully at: {csv_save_path}")

        # if not rerank:
        #     eval_results = self.evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
        # else:
        #     print('Applying person re-ranking ...')
        #     cids = np.append(np.array([cid for _, _, cid in query]), np.array([cid for _, _, cid in gallery]))
        #     distmat_qq, _, _ = self.pairwise_distance(features, query, query)
        #     distmat_gg, _, _ = self.pairwise_distance(features, gallery, gallery)
        #     distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(), cids, args)
        #     eval_results = self.evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        # return eval_results, query_tracklets, gallery_tracklets


# class Evaluator(object):
#     def __init__(self, model):
#         super(Evaluator, self).__init__()
#         self.model = model

#     def evaluate(self, data_loader, query, gallery, args=None, cmc_flag=False, rerank=False):
#         features, _ = extract_features(self.model, data_loader)
#         distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
#         print(query_features.shape)
#         print(gallery_features.shape)
#         # print(query)
#         # print("gallery: ",gallery)
        
#         # Sort each row of the distance matrix in ascending order
#         sorted_distmat = np.sort(distmat, axis=1)
        
#         print("Sorted distance matrix (each row in ascending order):")
#         print(sorted_distmat)

#         if not rerank:
#             return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

#         else:
#             print('Applying person re-ranking ...')
#             cids = np.append(np.array([cid for _, _, cid in query]), np.array([cid for _, _, cid in gallery]))
#             distmat_qq, _, _ = pairwise_distance(features, query, query)
#             distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
#             distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(), cids, args)
#             return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
