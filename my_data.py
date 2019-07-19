from __future__ import print_function

import os
import sys
import cv2
import torch
import torch.utils.data

import pickle, gzip
import numpy as np
import scipy.io as sio
import cv2
import json
from pycocotools.cocoeval import COCOeval
from collections import defaultdict, Sequence
import torch.multiprocessing as multiprocessing

from upsnet.config.config import config
from upsnet.rpn.assign_anchor import add_rpn_blobs
from upsnet.bbox import bbox_transform
from upsnet.bbox.sample_rois import sample_rois
import networkx as nx
from lib.utils.logging import logger

import pycocotools.mask as mask_util


from upsnet.config.config import config
from upsnet.dataset.json_dataset import JsonDataset, extend_with_flipped_entries, filter_for_training, add_bbox_regression_targets
from upsnet.dataset.base_dataset import BaseDataset
from upsnet.rpn.assign_anchor import add_rpn_blobs
from PIL import Image, ImageDraw
from lib.utils.logging import logger
from upsnet.dataset.myjson import myJson

import pycocotools.mask as mask_util

class mydata(BaseDataset):

    def __init__(self, image_sets, flip=False, proposal_files=None, phase='train', result_path=''):

        super(mydata, self).__init__()

        image_dirs = {
            'test_my': os.path.join(config.dataset.dataset_path, 'my_test_image'),  ##put your image path  to 
        }
        assert len(image_sets) == 1
        self.dataset =myJson(image_dirs,anno_file='./UPSNet-master/data/coco/annotations/instances_val2017.json')
        roidb = self.dataset.read_my_roidb(image_dirs['test_my'])
        self.panoptic_json_file = './UPSNet-master/data/coco/annotations/panoptic_val2017_stff.json'

        self.roidb = roidb
        self.phase = phase
        self.flip = flip
        self.result_path = result_path
        self.num_classes = 81

    def __len__(self):
        return len(self.roidb)

    def __getitem__(self, index):
        blob = defaultdict(list)
        im_blob, im_scales = self.get_image_blob([self.roidb[index]])
        if config.network.has_rpn:
            if self.phase != 'test':
                add_rpn_blobs(blob, im_scales, [self.roidb[index]])
                data = {'data': im_blob,
                        'im_info': blob['im_info']}
                label = {'roidb': blob['roidb'][0]}
                for stride in config.network.rpn_feat_stride:
                    label.update({
                        'rpn_labels_fpn{}'.format(stride): blob['rpn_labels_int32_wide_fpn{}'.format(stride)].astype(
                            np.int64),
                        'rpn_bbox_targets_fpn{}'.format(stride): blob['rpn_bbox_targets_wide_fpn{}'.format(stride)],
                        'rpn_bbox_inside_weights_fpn{}'.format(stride): blob[
                            'rpn_bbox_inside_weights_wide_fpn{}'.format(stride)],
                        'rpn_bbox_outside_weights_fpn{}'.format(stride): blob[
                            'rpn_bbox_outside_weights_wide_fpn{}'.format(stride)]
                    })
            else:
                data = {'data': im_blob,
                        'im_info': np.array([[im_blob.shape[-2],
                                              im_blob.shape[-1],
                                             im_scales[0]]], np.float32)}
                label = None
        else:
            raise NotImplementedError
        if config.network.has_fcn_head:
            if self.phase != 'test':
                seg_gt = np.array(Image.open(self.roidb[index]['image'].replace('images', 'annotations').replace('train2017', 'panoptic_train2017_semantic_trainid_stff').replace('val2017', 'panoptic_val2017_semantic_trainid_stff').replace('jpg', 'png')))
                if self.roidb[index]['flipped']:
                    seg_gt = np.fliplr(seg_gt)
                seg_gt = cv2.resize(seg_gt, None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                label.update({'seg_gt': seg_gt})
                label.update({'gt_classes': label['roidb']['gt_classes'][label['roidb']['is_crowd'] == 0]})
                label.update({'mask_gt': np.zeros((len(label['gt_classes']), im_blob.shape[-2], im_blob.shape[-1]))})
                idx = 0
                for i in range(len(label['roidb']['gt_classes'])):
                    if label['roidb']['is_crowd'][i] != 0:
                        continue
                    if type(label['roidb']['segms'][i]) is list and type(label['roidb']['segms'][i][0]) is list:
                        img = Image.new('L', (int(np.round(im_blob.shape[-1] / im_scales[0])), int(np.round(im_blob.shape[-2] / im_scales[0]))), 0)
                        for j in range(len(label['roidb']['segms'][i])):
                            ImageDraw.Draw(img).polygon(tuple(label['roidb']['segms'][i][j]), outline=1, fill=1)
                        label['mask_gt'][idx] = cv2.resize(np.array(img), None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                    else:
                        assert type(label['roidb']['segms'][i]) is dict or type(label['roidb']['segms'][i][0]) is dict
                        if type(label['roidb']['segms'][i]) is dict:
                            label['mask_gt'][idx] = cv2.resize(mask_util.decode(mask_util.frPyObjects([label['roidb']['segms'][i]], label['roidb']['segms'][i]['size'][0], label['roidb']['segms'][i]['size'][1]))[:, :, 0], None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                        else:
                            assert len(label['roidb']['segms'][i]) == 1
                            output = mask_util.decode(label['roidb']['segms'][i])
                            label['mask_gt'][idx] = cv2.resize(output[:, :, 0], None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                    idx += 1
                if config.train.fcn_with_roi_loss:
                    gt_boxes = label['roidb']['boxes'][np.where((label['roidb']['gt_classes'] > 0) & (label['roidb']['is_crowd'] == 0))[0]]
                    gt_boxes = np.around(gt_boxes * im_scales[0]).astype(np.int32)
                    label.update({'seg_roi_gt': np.zeros((len(gt_boxes), config.network.mask_size, config.network.mask_size), dtype=np.int64)})
                    for i in range(len(gt_boxes)):
                        if gt_boxes[i][3] == gt_boxes[i][1]:
                            gt_boxes[i][3] += 1
                        if gt_boxes[i][2] == gt_boxes[i][0]:
                            gt_boxes[i][2] += 1
                        label['seg_roi_gt'][i] = cv2.resize(seg_gt[gt_boxes[i][1]:gt_boxes[i][3], gt_boxes[i][0]:gt_boxes[i][2]], (config.network.mask_size, config.network.mask_size), interpolation=cv2.INTER_NEAREST)
            else:
                pass

        return data, label, index      


    def vis_all_mask(self, all_boxes, all_masks, save_path=None):
        """
        visualize all detections in one image
        :param im_array: [b=1 c h w] in rgb
        :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        :param class_names: list of names in imdb
        :param scale: visualize the scaled image
        :return:
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import random
        import cv2
        from lib.utils.colormap import colormap

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        color_list = colormap(rgb=True) / 255
        mask_color_id = 0

        for i in range(len(self.roidb)):
            im = np.array(Image.open(self.roidb[i]['image']))
            fig = plt.figure(frameon=False)

            fig.set_size_inches(im.shape[1] / 200, im.shape[0] / 200)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(im)
            for j, name in enumerate(self.dataset.classes):
                #breakpoint()
                if name == '__background__':
                    continue
                boxes = all_boxes[j][i]
                segms = all_masks[j][i]
                if segms == []:
                    continue
                masks = mask_util.decode(segms)
                for k in range(boxes.shape[0]):
                    score = boxes[k, -1]
                    mask = masks[:, :, k]
                    if score < 0.5:
                        continue
                    bbox = boxes[k, :]
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                      fill=False, edgecolor='g', linewidth=1, alpha=0.5)
                    )
                    ax.text(bbox[0], bbox[1] - 2, name + '{:0.2f}'.format(score).lstrip('0'), fontsize=5, family='serif',
                            bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'), color='white')
                    color_mask = color_list[mask_color_id % len(color_list), 0:3]
                    mask_color_id += 1
                    w_ratio = .4
                    for c in range(3):
                        color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio

                    contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    for c in contour:
                        ax.add_patch(
                            Polygon(
                                c.reshape((-1, 2)),
                                fill=True, facecolor=color_mask, edgecolor='w', linewidth=0.8, alpha=0.5
                            )
                        )
            if save_path is None:
                plt.show()
            else:
                #breakpoint()
                fig.savefig(os.path.join(save_path, '{}.png'.format(self.roidb[i]['file_name'])), dpi=200)
            plt.close('all')
    def evaluate_panoptic(self, pred_pans_2ch, output_dir):

        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))

        from panopticapi.utils import IdGenerator

        # def get_gt(pan_gt_json_file=None, pan_gt_folder=None):
            # if pan_gt_json_file is None:
                # pan_gt_json_file = self.panoptic_json_file
            # if pan_gt_folder is None:
                # pan_gt_folder = self.panoptic_gt_folder
            # with open(pan_gt_json_file, 'r') as f:
                # pan_gt_json = json.load(f)
            # files = [item['file_name'] for item in pan_gt_json['images']]
            # cpu_num = multiprocessing.cpu_count()
            # files_split = np.array_split(files, cpu_num)
            # workers = multiprocessing.Pool(processes=cpu_num)
            # processes = []
            # for proc_id, files_set in enumerate(files_split):
                # p = workers.apply_async(BaseDataset._load_image_single_core, (proc_id, files_set, pan_gt_folder))
                # processes.append(p)
            # workers.close()
            # workers.join()
            # pan_gt_all = []
            # for p in processes:
                # pan_gt_all.extend(p.get())

        def get_gt(pan_gt_json_file=None):
            if pan_gt_json_file is None:
                pan_gt_json_file = self.panoptic_json_file
            with open(pan_gt_json_file, 'r') as f:
                pan_gt_json = json.load(f)
            files = [item['file_name'] for item in pan_gt_json['images']]
            categories = pan_gt_json['categories']
            categories = {el['id']: el for el in categories}
            #breakpoint()
            color_gererator = IdGenerator(categories)


            return pan_gt_json, categories, color_gererator

        def get_pred(pan_2ch_all, color_gererator, cpu_num=None):
            if cpu_num is None:
                cpu_num = multiprocessing.cpu_count()
            pan_2ch_split = np.array_split(pan_2ch_all, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
                p = workers.apply_async(BaseDataset._converter_2ch_single_core, (proc_id, pan_2ch_set, color_gererator))
                processes.append(p)
            workers.close()
            workers.join()
            annotations, pan_all = [], []
            for p in processes:
                p = p.get()
                annotations.extend(p[0])
                pan_all.extend(p[1])
            pan_json = {'annotations': annotations}
            return pan_all, pan_json

        def save_image(images, save_folder, gt_json, colors=None):
            #breakpoint()
            os.makedirs(save_folder, exist_ok=True)
            #names = [os.path.join(save_folder, item['file_name'].replace('_leftImg8bit', '').replace('jpg', 'png').replace('jpeg', 'png')) for item in gt_json['images']]
            names = []
            for i in range(len(self.roidb)):
                names.append(os.path.join(save_folder,self.roidb[i]['file_name']))
            #breakpoint()
            #name = [os.path.join(save_folder,)]
            cpu_num = multiprocessing.cpu_count()
            images_split = np.array_split(images, cpu_num)
            names_split = np.array_split(names, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            for proc_id, (images_set, names_set) in enumerate(zip(images_split, names_split)):
                workers.apply_async(BaseDataset._save_image_single_core, (proc_id, images_set, names_set, colors))
            workers.close()
            workers.join()

        # def pq_compute(gt_jsons, pred_jsons, gt_pans, pred_pans, categories):
            # start_time = time.time()
            # #from json and from numpy
            # gt_image_jsons = gt_jsons['images']
            # gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
            # cpu_num = multiprocessing.cpu_count()
            # gt_jsons_split, pred_jsons_split = np.array_split(gt_jsons, cpu_num), np.array_split(pred_jsons, cpu_num)
            # gt_pans_split, pred_pans_split = np.array_split(gt_pans, cpu_num), np.array_split(pred_pans, cpu_num)
            # gt_image_jsons_split = np.array_split(gt_image_jsons, cpu_num)

            # workers = multiprocessing.Pool(processes=cpu_num)
            # processes = []
            # for proc_id, (gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set) in enumerate(zip(gt_jsons_split, pred_jsons_split, gt_pans_split, pred_pans_split, gt_image_jsons_split)):
                # p = workers.apply_async(BaseDataset._pq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories))
                # processes.append(p)
            # workers.close()
            # workers.join()
            # pq_stat = PQStat()
            # for p in processes:
                # pq_stat += p.get()
            # metrics = [("All", None), ("Things", True), ("Stuff", False)]
            # results = {}
            # for name, isthing in metrics:
                # results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
                # if name == 'All':
                    # results['per_class'] = per_class_results

            # if logger:
                # logger.info("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
                # logger.info("-" * (10 + 7 * 4))
                # for name, _isthing in metrics:
                    # logger.info("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']))

                # logger.info("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"))
                # for idx, result in results['per_class'].items():
                    # logger.info("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}".format(idx, 100 * result['pq'], 100 * result['sq'], 100 * result['rq'], result['iou'], result['tp'],
                                                                                             # result['fp'], result['fn']))

            # t_delta = time.time() - start_time
            # print("Time elapsed: {:0.2f} seconds".format(t_delta))
            # return results


        # if eval for test-dev, since there is no gt we simply retrieve image names from image_info json files
        # with open(self.panoptic_json_file, 'r') as f:
        #     gt_json = json.load(f)
        #     gt_json['images'] = sorted(gt_json['images'], key=lambda x: x['id'])
        # other wise:
        gt_json, categories, color_gererator = get_gt()

        pred_pans, pred_json = get_pred(pred_pans_2ch, color_gererator)
        save_image(pred_pans_2ch, os.path.join(output_dir, 'pan_2ch'), gt_json)
        save_image(pred_pans, os.path.join(output_dir, 'pan'), gt_json)
        json.dump(gt_json, open(os.path.join(output_dir, 'gt.json'), 'w'))
        json.dump(pred_json, open(os.path.join(output_dir, 'pred.json'), 'w'))
        #results = pq_compute(gt_json, pred_json, gt_pans, pred_pans, categories)

        #return results

    def get_unified_pan_result(self, segs, pans, cls_inds, stuff_area_limit=4 * 64 * 64):
        pred_pans_2ch = []

        for (seg, pan, cls_ind) in zip(segs, pans, cls_inds):
            pan_seg = pan.copy()
            pan_ins = pan.copy()
            id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes
            ids = np.unique(pan)
            ids_ins = ids[ids > id_last_stuff]
            pan_ins[pan_ins <= id_last_stuff] = 0
            for idx, id in enumerate(ids_ins):
                region = (pan_ins == id)
                if id == 255:
                    pan_seg[region] = 255
                    pan_ins[region] = 0
                    continue
                cls, cnt = np.unique(seg[region], return_counts=True)
                if cls[np.argmax(cnt)] == cls_ind[id - id_last_stuff - 1] + id_last_stuff:
                    pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                    pan_ins[region] = idx + 1
                else:
                    if np.max(cnt) / np.sum(cnt) >= 0.5 and cls[np.argmax(cnt)] <= id_last_stuff:
                        pan_seg[region] = cls[np.argmax(cnt)]
                        pan_ins[region] = 0
                    else:
                        pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                        pan_ins[region] = idx + 1

            idx_sem = np.unique(pan_seg)
            for i in range(idx_sem.shape[0]):
                if idx_sem[i] <= id_last_stuff:
                    area = pan_seg == idx_sem[i]
                    if (area).sum() < stuff_area_limit:
                        pan_seg[area] = 255

            pan_2ch = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)
            pan_2ch[:, :, 0] = pan_seg
            pan_2ch[:, :, 1] = pan_ins
            pred_pans_2ch.append(pan_2ch)
        return pred_pans_2ch

    def get_combined_pan_result(self, segs, boxes, masks, score_threshold=0.6, fraction_threshold=0.7, stuff_area_limit=4*64*64):
        # suppose ins masks are already sorted in descending order by scores
        boxes_all, masks_all, cls_idxs_all = [], [], []
        boxes_all = []
        import itertools
        import time
        for i in range(len(segs)):
            boxes_i = np.vstack([boxes[j][i] for j in range(1, len(boxes))])
            masks_i = np.array(list(itertools.chain(*[masks[j][i] for j in range(1, len(masks))])))
            cls_idxs_i = np.hstack([np.array([j for _ in boxes[j][i]]).astype(np.int32) for j in range(1, len(boxes))])
            sorted_idxs = np.argsort(boxes_i[:, 4])[::-1]
            boxes_all.append(boxes_i[sorted_idxs])
            masks_all.append(masks_i[sorted_idxs])
            cls_idxs_all.append(cls_idxs_i[sorted_idxs])

        cpu_num = multiprocessing.cpu_count()
        boxes_split = np.array_split(boxes_all, cpu_num)
        cls_idxs_split = np.array_split(cls_idxs_all, cpu_num)
        masks_split = np.array_split(masks_all, cpu_num)
        segs_split = np.array_split(segs, cpu_num)
        workers = multiprocessing.Pool(processes=cpu_num)
        processes = []
        for proc_id, (boxes_set, cls_idxs_set, masks_set, sems_set) in enumerate(zip(boxes_split, cls_idxs_split, masks_split, segs_split)):
            p = workers.apply_async(BaseDataset._merge_pred_single_core, (proc_id, boxes_set, cls_idxs_set, masks_set, sems_set, score_threshold, fraction_threshold, stuff_area_limit))
            processes.append(p)
        workers.close()
        workers.join()
        pan_2ch_all = []
        for p in processes:
            pan_2ch_all.extend(p.get())
        return pan_2ch_all

    @staticmethod
    def _merge_pred_single_core(proc_id, boxes_set, cls_idxs_set, masks_set, sems_set, score_threshold, fraction_threshold, stuff_area_limit):
        from pycocotools.mask import decode as mask_decode
        pan_2ch_all = []
        id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes

        for idx_outer in range(len(boxes_set)):
            boxes, scores, cls_idxs, masks = boxes_set[idx_outer][:, :4], boxes_set[idx_outer][:, 4], cls_idxs_set[idx_outer], masks_set[idx_outer]
            sem = sems_set[idx_outer]
            h, w = sem.shape
            ins_mask = np.zeros((h, w), dtype=np.uint8)
            ins_sem = np.zeros((h, w), dtype=np.uint8)
            idx_ins_array = np.zeros(config.dataset.num_classes - 1, dtype=np.uint32)
            for idx_inner in range(len(scores)):
                score, cls_idx, mask = scores[idx_inner], cls_idxs[idx_inner], masks[idx_inner]
                if score < score_threshold:
                    continue
                mask = mask_decode(masks[idx_inner])
                ins_remain = (mask == 1) & (ins_mask == 0)
                if (mask.astype(np.float32).sum() == 0) or (ins_remain.astype(np.float32).sum() / mask.astype(np.float32).sum() < fraction_threshold):
                    continue
                idx_ins_array[cls_idx - 1] += 1
                ins_mask[ins_remain] = idx_ins_array[cls_idx - 1]
                ins_sem[ins_remain] = cls_idx

            idx_sem = np.unique(sem)
            for i in range(idx_sem.shape[0]):
                if idx_sem[i] <= id_last_stuff:
                    area = sem == idx_sem[i]
                    if (area).sum() < stuff_area_limit:
                        sem[area] = 255

            # merge sem and ins, leave conflict region as 255
            pan_2ch = np.zeros((h, w, 3), dtype=np.uint8)
            pan_2ch_c0 = sem.copy()
            pan_2ch_c1 = ins_mask.copy()
            conflict = (sem > id_last_stuff) & (ins_mask == 0)  # sem treat as thing while ins treat as stuff
            pan_2ch_c0[conflict] = 255
            insistence = (ins_mask != 0)  # change sem region to ins thing region
            pan_2ch_c0[insistence] = ins_sem[insistence] + id_last_stuff
            pan_2ch[:, :, 0] = pan_2ch_c0
            pan_2ch[:, :, 1] = pan_2ch_c1
            pan_2ch_all.append(pan_2ch)

        return pan_2ch_all

    @staticmethod
    def _load_image_single_core(proc_id, files_set, folder):
        images = []
        for working_idx, file in enumerate(files_set):
            try:
                image = np.array(Image.open(os.path.join(folder, file)))
                images.append(image)
            except Exception:
                pass
        return images

    @staticmethod
    def _converter_2ch_single_core(proc_id, pan_2ch_set, color_gererator):
        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))
        from panopticapi.utils import rgb2id
        OFFSET = 1000
        VOID = 255
        annotations, pan_all = [], []
        for idx in range(len(pan_2ch_set)):
            pan_2ch = np.uint32(pan_2ch_set[idx])
            pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 1]
            pan_format = np.zeros((pan_2ch.shape[0], pan_2ch.shape[1], 3), dtype=np.uint8)

            l = np.unique(pan)
            segm_info = []
            for el in l:
                sem = el // OFFSET
                if sem == VOID:
                    continue
                mask = pan == el
                if vis_panoptic:
                    color = color_gererator.categories[sem]['color']
                else:
                    color = color_gererator.get_color(sem)
                pan_format[mask] = color
                index = np.where(mask)
                x = index[1].min()
                y = index[0].min()
                width = index[1].max() - x
                height = index[0].max() - y
                segm_info.append({"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)), "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()})
            annotations.append({"segments_info": segm_info})
            if vis_panoptic:
                pan_format = Image.fromarray(pan_format)
                draw = ImageDraw.Draw(pan_format)
                for el in l:
                    sem = el // OFFSET
                    if sem == VOID:
                        continue
                    if color_gererator.categories[sem]['isthing'] and el % OFFSET != 0:
                        mask = ((pan == el) * 255).astype(np.uint8)
                        _, contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for c in contour:
                            c = c.reshape(-1).tolist()
                            if len(c) < 4:
                                print('warning: invalid contour')
                                continue
                            draw.line(c, fill='white', width=2)
                pan_format = np.array(pan_format)
            pan_all.append(pan_format)
        return annotations, pan_all

    @staticmethod
    def _pq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories):
        OFFSET = 256 * 256 * 256
        VOID = 0
        pq_stat = PQStat()
        for idx, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(zip(gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set)):
            # if idx % 100 == 0:
            #     logger.info('Compute pq -> Core: {}, {} from {} images processed'.format(proc_id, idx, len(gt_jsons_set)))
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256

            gt_segms = {el['id']: el for el in gt_json['segments_info']}
            pred_segms = {el['id']: el for el in pred_json['segments_info']}

            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

            # confusion matrix calculation
            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()
            tp = 0
            fp = 0
            fn = 0

            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                if gt_label not in gt_segms:
                    continue
                if pred_label not in pred_segms:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    continue

                union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                    (VOID, pred_label), 0)
                iou = intersection / union
                if iou > 0.5:
                    pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    tp += 1

            # count false positives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                pq_stat[gt_info['category_id']].fn += 1
                fn += 1

            # count false positives
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                pq_stat[pred_info['category_id']].fp += 1
                fp += 1
        # logger.info('Compute pq -> Core: {}, all {} images processed'.format(proc_id, len(gt_jsons_set)))
        return pq_stat
















