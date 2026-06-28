# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
import time
from pycocotools.cocoeval import COCOeval
import cv2
from pycocotools.coco import COCO

from detectron2 import _C

logger = logging.getLogger(__name__)


# class COCOeval_opt(COCOeval):
#     """
#     This is a slightly modified version of the original COCO API, where the functions evaluateImg()
#     and accumulate() are implemented in C++ to speedup evaluation
#     """
#
#     def evaluate(self):
#         """
#         Run per image evaluation on given images and store results in self.evalImgs_cpp, a
#         datastructure that isn't readable from Python but is used by a c++ implementation of
#         accumulate().  Unlike the original COCO PythonAPI, we don't populate the datastructure
#         self.evalImgs because this datastructure is a computational bottleneck.
#         :return: None
#         """
#         tic = time.time()
#
#         p = self.params
#         # add backward compatibility if useSegm is specified in params
#         if p.useSegm is not None:
#             p.iouType = "segm" if p.useSegm == 1 else "bbox"
#         logger.info("Evaluate annotation type *{}*".format(p.iouType))
#         p.imgIds = list(np.unique(p.imgIds))
#         if p.useCats:
#             p.catIds = list(np.unique(p.catIds))
#         p.maxDets = sorted(p.maxDets)
#         self.params = p
#
#         self._prepare()  # bottleneck
#
#         # loop through images, area range, max detection number
#         catIds = p.catIds if p.useCats else [-1]
#
#         if p.iouType == "segm" or p.iouType == "bbox":
#             computeIoU = self.computeIoU
#         elif p.iouType == "keypoints":
#             computeIoU = self.computeOks
#         self.ious = {
#             (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
#         }  # bottleneck
#
#         maxDet = p.maxDets[-1]
#
#         # <<<< Beginning of code differences with original COCO API
#         def convert_instances_to_cpp(instances, is_det=False):
#             # Convert annotations for a list of instances in an image to a format that's fast
#             # to access in C++
#             instances_cpp = []
#             for instance in instances:
#                 instance_cpp = _C.InstanceAnnotation(
#                     int(instance["id"]),
#                     instance["score"] if is_det else instance.get("score", 0.0),
#                     instance["area"],
#                     bool(instance.get("iscrowd", 0)),
#                     bool(instance.get("ignore", 0)),
#                 )
#                 instances_cpp.append(instance_cpp)
#             return instances_cpp
#
#         # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++
#         ground_truth_instances = [
#             [convert_instances_to_cpp(self._gts[imgId, catId]) for catId in p.catIds]
#             for imgId in p.imgIds
#         ]
#         detected_instances = [
#             [convert_instances_to_cpp(self._dts[imgId, catId], is_det=True) for catId in p.catIds]
#             for imgId in p.imgIds
#         ]
#         ious = [[self.ious[imgId, catId] for catId in catIds] for imgId in p.imgIds]
#
#         if not p.useCats:
#             # For each image, flatten per-category lists into a single list
#             ground_truth_instances = [[[o for c in i for o in c]] for i in ground_truth_instances]
#             detected_instances = [[[o for c in i for o in c]] for i in detected_instances]
#
#         # Call C++ implementation of self.evaluateImgs()
#         self._evalImgs_cpp = _C.COCOevalEvaluateImages(
#             p.areaRng, maxDet, p.iouThrs, ious, ground_truth_instances, detected_instances
#         )
#         self._evalImgs = None
#
#         self._paramsEval = copy.deepcopy(self.params)
#         toc = time.time()
#         logger.info("COCOeval_opt.evaluate() finished in {:0.2f} seconds.".format(toc - tic))
#         # >>>> End of code differences with original COCO API
#
#     def accumulate(self):
#         """
#         Accumulate per image evaluation results and store the result in self.eval.  Does not
#         support changing parameter settings from those used by self.evaluate()
#         """
#         logger.info("Accumulating evaluation results...")
#         tic = time.time()
#         assert hasattr(
#             self, "_evalImgs_cpp"
#         ), "evaluate() must be called before accmulate() is called."
#
#         self.eval = _C.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)
#
#         # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections
#         self.eval["recall"] = np.array(self.eval["recall"]).reshape(
#             self.eval["counts"][:1] + self.eval["counts"][2:]
#         )
#
#         # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X
#         # num_area_ranges X num_max_detections
#         self.eval["precision"] = np.array(self.eval["precision"]).reshape(self.eval["counts"])
#         self.eval["scores"] = np.array(self.eval["scores"]).reshape(self.eval["counts"])
#         toc = time.time()
#         logger.info("COCOeval_opt.accumulate() finished in {:0.2f} seconds.".format(toc - tic))

from collections import defaultdict
import pycocotools.mask as maskUtils
import pdb
import datetime
class COCOeval_opt(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        # per-image per-category evaluation results [KxAxI] elements
        self.evalImgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = AmodalParams(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.seg_stats = []
        self.ious = {}  # ious between all gts and dts
        self.ious_for_AJI = {}  # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def computeIoU_for_AJI(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return [], [], []


        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [0] * len(g)
        if len(g) != 0:
            gt_area = maskUtils.area(g)
        else:
            gt_area = None

        ious = maskUtils.iou(d, g, iscrowd)
        nd_ious=np.asarray(ious)

        if len(d) == 0 or len(g) == 0:

            ious = []
            dsc = []
            intersection = []
            merge_area=[]
            if len(d) > 0:
                merge_area = copy.deepcopy(d)
            if len(g) > 0:
                merge_area = copy.deepcopy(g)
            merge_area = maskUtils.merge(merge_area, intersect=False)
            union = [maskUtils.area(merge_area)]

        else:
            intersection = np.zeros_like(nd_ious, dtype=np.double)
            union = np.zeros_like(nd_ious, dtype=np.double)
            m,n=nd_ious.shape
            for i in range(m):
                intersection_area=[]
                j_indices=[]
                union_area=[]
                for j in range(n):
                    if nd_ious[i][j] != 0:
                        mask1 = d[i]
                        mask2 = g[j]
                        mask1= maskUtils.decode(mask1)
                        mask2=maskUtils.decode(mask2)
                        intersection_area.append(np.sum(np.logical_and(mask1, mask2)))
                        j_indices.append(j)
                        union_area.append(np.sum(np.logical_or(mask1, mask2)))
                if intersection_area:
                    max_intersection_area=max(intersection_area)
                    max_intersection_index=intersection_area.index(max_intersection_area)
                    max_union_area=union_area[max_intersection_index]
                    j_index=j_indices[max_intersection_index]
                    intersection[i][j_index] = max_intersection_area
                    union[i][j_index] = max_union_area

            # [2 * i/(u + i) for i,u in zip(intersection, union)]
            dsc = 2 * intersection/(union + intersection + 1e-10)

            if dsc.max() > 1:
                pdb.set_trace()

        return ious,intersection,union,gt_area, dsc

    def _prepare(self,):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params

        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(
                imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(
                imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)

        for dt in dts:
            # if dt['score']>0.90:
                self._dts[dt['image_id'], dt['category_id']].append(dt)
        # per-image per-category evaluation results
        self.evalImgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results

    def compute_F1(self, gt_area, iou, UseIOU=True):
        TP = 0
        FP = 0
        FN = 0

        PR_thread = [i for i in np.linspace(0.5, 0.9, 10)]
        TPLIST = [0 for i in range(10)]
        FPLIST = [0 for i in range(10)]
        # PLIST = [0 for i in range(28)]
        # RLIST =[0 for i in range(28)]
        F1LIST = [0 for i in range(10)]
        iou_copy = copy.deepcopy(iou)
        gt_num = iou.shape[1]
        # gt_map_seg = np.zeros((gt_num,2)) # 0 for map idx, 1 for iou value
        # pdb.set_trace()
        iou_list = iou_copy.T.tolist()
        inter_index_list = list(map(lambda x: x.index(
            max(x)) if max(x) > 0 else -1, iou_list))
        inter_value_list = list(map(lambda x: max(x), iou_list))
        # gt_map_seg[:,0] = np.asarray(inter_index_list,)
        # gt_map_seg[:,1] = np.asarray(inter_value_list)
        inter_index_set = set(inter_index_list)
        inter_index_set.discard(-1)

        while (len(inter_index_list) - inter_value_list.count(0)) != len(inter_index_set):
            # find the duplicate index and set another segmented result to ground truth base on criterion

            duplicate_indices = []

            for v in inter_index_set:
                if inter_index_list.count(v) > 1:
                    duplicate_indices = [i for i, x in enumerate(
                        inter_index_list) if x == v]
                    break
            # then get the max iou index in duplicate indices
            if len(duplicate_indices) == 0:
                # pdb.set_trace()
                print('bug')
            iou_for_duplicate = list(
                map(inter_value_list.__getitem__, duplicate_indices))
            # delete the index with max iou
            del duplicate_indices[(
                iou_for_duplicate.index(max(iou_for_duplicate)))]
            # search for best iou match again
            for i in duplicate_indices:
                iou_list[i][v] = 0
                inter_index_list[i] = iou_list[i].index(
                    max(iou_list[i])) if max(iou_list[i]) > 0 else -1
                inter_value_list[i] = max(iou_list[i])
            inter_index_set = set(inter_index_list)
            inter_index_set.discard(-1)
        # so far, for each gt, we map a seg result.
        # Then computer ratio = intersect/union
        for gtidx, segidx in enumerate(inter_index_list):
            if segidx != -1:

                if UseIOU:
                    value = iou_list[gtidx][segidx]

                if value > 0.5:
                    TP += 1

                # LIST
                for k, thread in enumerate(PR_thread):
                    if value > thread:
                        TPLIST[k] += 1

        # add unmatched segmented result to FP
        seg_num = iou.shape[0]
        FNLIST = [len(gt_area) - f for f in TPLIST]

        FPLIST = [iou.shape[0] - t for t in TPLIST]
        # FPLIST = [f + (iou.shape[0] - t ) for t,f in zip(TPLIST,FPLIST)]
        # pdb.set_trace()
        PLIST = [t / (t + f) for t, f in zip(TPLIST, FPLIST)]
        RLIST = [t / (t + f) for t, f in zip(TPLIST, FNLIST)]
        itm = 0
        for p, r in zip(PLIST, RLIST):
            if (p + r) == 0:
                F1LIST[itm] = 0
            else:
                F1LIST[itm] = 2 * p * r / (p + r)
            itm += 1

        FN = len(gt_area) - TP
        FP = (iou.shape[0] - TP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        if (recall + precision) == 0:
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        return PLIST, RLIST, F1, precision, recall

    def caclulateMetrics(self, ious, dsc, gt):

        dc_thread = 0.0
        # p = self.params
        try:
            D, G = ious.shape
        except:
            G = len(gt)
            D = 0
        # print("D IS {}, G IS {}".format(D,G))
        if D == 0:
            gtdsc = np.zeros((G))
            # mdsc = 0
            alldsc = gtdsc[gtdsc > dc_thread]
        else:
            gtdsc = np.zeros((G))
            dtdsc = np.zeros((D))
            # AJI
            # DSC = np.zeros((1, 1))
            dsc_shape = dsc.shape
            temp_dsc=dsc.copy()
            while temp_dsc.max() > dc_thread:
                maxind = np.argmax(temp_dsc)
                # [detect, gt]
                ind = np.unravel_index(maxind, dsc_shape)
                maxdsc = temp_dsc[ind]
                gtdsc[ind[1]] = maxdsc
                dtdsc[ind[0]] = maxdsc
                temp_dsc[ind[0]] = 0
                temp_dsc[:, ind[1]] = 0
            # pdb.set_trace()
            # alldsc = gtdsc
            # print(gtdsc,dtdsc)
            alldsc = np.concatenate((gtdsc,dtdsc[dtdsc==0]))
            # print(alldsc)
            # mdsc = np.mean(alldsc)
        return alldsc

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        AJI = np.zeros((T, 1))
        DSC = np.zeros((G, 1))
        F1 = 0
        mdsc = np.zeros(G)

        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
            # compute AJI, DSC
            if p.iouType == 'segm':
                # if len(self.ious[imgId, catId][0]) > 0 else self.ious[imgId, catId][0]
                ious_for_AJI = self.ious_for_AJI[imgId, catId][0]
                # if len(self.ious[imgId, catId][1]) > 0 else self.ious[imgId, catId][1]
                intersection_for_AJI = self.ious_for_AJI[imgId, catId][1]
                # if len(self.ious[imgId, catId][2]) > 0 else self.ious[imgId, catId][2]
                union_for_AJI = self.ious_for_AJI[imgId, catId][2]
                area_for_AJI = self.ious_for_AJI[imgId, catId][3]
                dsc_for_AJI = self.ious_for_AJI[imgId, catId][4]

                if len(gt) != 0 and len(dt) != 0:
                    PLIST, RLIST, F1, precision, recall = self.compute_F1(
                        area_for_AJI, ious_for_AJI, UseIOU=True)
                elif len(gt) == 0 and len(dt) > 0:
                    F1, precision, recall = 0, 0, 1
                    PLIST, RLIST = [0 for i in range(28)], [
                        1 for i in range(28)]
                elif len(gt) > 0 and len(dt) == 0:
                    F1, precision, recall = 0, 1, 0
                    PLIST, RLIST = [1 for i in range(28)], [
                        0 for i in range(28)]
                else:
                    F1, precision, recall = 1, 1, 1
                    PLIST, RLIST = [1 for i in range(28)], [
                        1 for i in range(28)]

                mdsc = self.caclulateMetrics(ious_for_AJI,dsc_for_AJI, gt)

                # calculate AJI
                dc_thread = 0.6
                iouThrsAJI = [0.5]
                T = len(iouThrsAJI)
                G = len(gt)
                D = len(dt)
                gtm_for_AJI = - np.ones((T, G))
                dtm_for_AJI = - np.ones((T, D))
                gtIg_for_AJI = np.array([g['_ignore'] for g in gt])
                dtIg_for_AJI = np.zeros((T, D))
                # AJI
                AJI = np.zeros((T, 1))
                # IOU = np.zeros((T,1))
                INTERSECTION = np.zeros((T, 1))
                UNION = np.zeros((T, 1))

                DSC = np.zeros((G, 1))
                if not len(ious_for_AJI) == 0:
                    for tind, t in enumerate(iouThrsAJI):
                        for gind, g in enumerate(gt):
                            iou = min([t, 1 - 1e-10])
                            _intersection = 0
                            _union = 0
                            m = -1
                            _dsc = 0
                            for dind, d in enumerate(dt):
                                # if the dt already matched, continue
                                if dtm_for_AJI[tind, dind] > 0:
                                    continue
                                # continue to next dt unless better match made
                                if ious_for_AJI[dind, gind] < iou:
                                    continue
                                # if match successful and best so far, store it
                                iou = ious_for_AJI[dind, gind]
                                _union = union_for_AJI[dind, gind]
                                _intersection = intersection_for_AJI[dind, gind]
                                # _dsc = dsc[dind, gind]
                                m = dind
                            if m == -1:
                                continue

                            dtm_for_AJI[tind, m] = g['id']
                            gtm_for_AJI[tind, gind] = dt[m]['id']
                            INTERSECTION[tind, 0] = INTERSECTION[tind,
                            0] + _intersection
                            UNION[tind, 0] = UNION[tind, 0] + _union
                            DSC[gind, 0] = _dsc
                        # add missing gt and dt

                        miss_gt = np.argwhere(gtm_for_AJI == -1)
                        miss_dt = np.argwhere(dtm_for_AJI == -1)
                        miss_gt = [gt[gt_index[1]]['segmentation']
                                   for gt_index in miss_gt]
                        miss_dt = [dt[dt_index[1]]['segmentation']
                                   for dt_index in miss_dt]
                        miss_gt = [maskUtils.area(f) for f in miss_gt]
                        miss_dt = [maskUtils.area(f) for f in miss_dt]
                        UNION[tind, 0] = UNION[tind, 0] + \
                                         sum(miss_dt) + sum(miss_gt)

                    AJI = np.divide(INTERSECTION, UNION)
                    gtIg_for_AJI = np.array([0 for g in gt])
                    # DSC > 0.7
                    # good_ins = DSC[np.where(DSC > dc_thread)]
                    # if len(np.nonzero(good_ins)[0]) == 0:
                    #     DSC_GOOD = 0
                    # else:
                    #     DSC_GOOD = np.mean(good_ins)
                    # DSC_GOOD = np.asarray(DSC_GOOD).reshape(1,1)
                    # if catId =='nuclei':
                    #     pdb.set_trace()
                    # fno
                    # DSC[DSC> dc_thread] = 1
                    # DSC[DSC<=dc_thread] = 0
                    # FNO =  (G - np.sum(DSC))/G

                else:
                    AJI = np.zeros((T, 1))
                    # DSC_GOOD = np.zeros((T,1))
                    # FNO = 1

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                      for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(
            dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'AJI': AJI,
            'F1': F1,
            'DSC': mdsc,
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
            'num_G': G,
            'num_D': D,
        }

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        # p.imgIds=p.imgIds[1:2]
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds
                     for catId in catIds}
        if p.iouType == 'segm':
            computeIoU_for_AJI = self.computeIoU_for_AJI
            self.ious_for_AJI = {(imgId, catId): computeIoU_for_AJI(imgId, catId)
                                 for imgId in p.imgIds
                                 for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds
                         ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params

        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(
                p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(
                1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large',
                                  maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            stats[11] = _summarize(
                0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats


        if not self.eval:
            raise Exception('Please run accumulate() first')

        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps

        self.stats = summarize()

    def __str__(self):
        self.summarize()

    def summarizeSegm(self):
        AJI = {}
        DSC = {}

        F1_score = {}
        summarize_metric = np.zeros((2, 3))
        from tabulate import tabulate
        seg_stats = np.zeros((2,3))
        for catId, cat in enumerate(self._paramsEval.catIds):
            _count = 0
            aji = np.zeros((len(self._paramsEval.iouThrs), 1))
            F1 = 0
            dsc = []
            num_G = 0
            num_D = 0
            for i,result in enumerate(self.evalImgs):
                # if i==0:
                    if result is None:
                        # skip len(gt)=0 & len(dt)
                        continue
                    if result['category_id'] == cat:
                        aji = aji + result['AJI']
                        F1 = F1 + result['F1']
                        dsc.extend(list(result['DSC']))
                        num_G = num_G + result["num_G"]
                        num_D = num_D + result["num_D"]
                        _count += 1

            if _count>0:
                aji = np.divide(aji, _count)
                F1 = F1 / _count

            dsc=np.array(dsc)
            dsc=np.mean(dsc)

            # hist, bin_edges = np.histogram(nonzero_dsc, bins=10, range=(0, 1), density=True)
            # for i in range(len(hist)):
            #     print(f'Bin {i + 1}: {hist[i]}')

            AJI[cat] = aji[0]
            F1_score[cat] = F1

            DSC[cat] = dsc
            cat_ids=[cat]
            cat_info = self.cocoGt.loadCats(ids=cat_ids)
            cat_name=cat_info[0]['name']
            seg_stats[catId][0]=F1
            seg_stats[catId][1]=dsc
            seg_stats[catId][2]=aji[0]


            summarize_metric[catId] = [
                AJI[cat][0], F1_score[cat], DSC[cat]]
            table = [["AJI", "F1", "DSC"], summarize_metric[catId]]
            # table = [["AJI", "F1", "DSC", "TPRp"], [AJI[cat][0], F1_score[cat], DSC[cat]]]
            print("===============", cat_name, "================")
            print(tabulate(table, headers='firstrow', tablefmt='github'))
        print("================= Average =================")
        avg = np.mean(summarize_metric, axis=0)
        table = [["AJI", "F1", "DSC"],
                 [avg[0], avg[1], avg[2]]]
        print(tabulate(table, headers='firstrow', tablefmt='github'))

        return seg_stats

class AmodalParams:
    '''
    Params for coco evaluation api
    '''

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(
            np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(
            np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2],
                        [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(
            np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(
            np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2],
                        [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None