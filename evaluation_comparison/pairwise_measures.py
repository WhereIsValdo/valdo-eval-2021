from __future__ import absolute_import, print_function

import numpy as np
from scipy import ndimage
from functools import partial
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import generate_binary_structure


class CacheFunctionOutput(object):
    """
    this provides a decorator to cache function outputs
    to avoid repeating some heavy function computations
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        return partial(self, obj)  # to remember func as self.func

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            value = cache[key]
        except KeyError:
            value = cache[key] = self.func(*args, **kw)
        return value


class MorphologyOps(object):
    """
    Class that performs the morphological operations needed to get the connected components.
    Used in the evaluation
    """

    def __init__(self, binary_img, connectivity):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.connectivity = connectivity

    def border_map(self):
        """
        Creates the border for a 3D image
        :return:
        """
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border

    def get_struct(self):
        dim = len(self.binary_map.shape)
        assert (dim == 2) or (dim == 3), "Error incorrect shape %d" % dim

        struct = generate_binary_structure(rank=dim, connectivity=self.connectivity)

        return struct

    def foreground_component(self):
        struct = self.get_struct()
        return ndimage.label(self.binary_map, struct)


class PairwiseMeasures(object):
    def __init__(self, ref_img, seg_img, analysis='lacunes',
                 measures=None, connectivity=3, pixdim=[1, 1, 1],
                 empty=False, list_labels=None, threshold=0.5, thresh_assign=0.1):

        # define metric functions and labels
        self.m_dict = {
            'f1_score': (self.f1_count_element, 'F1 score'),
            'absolute_volume_difference': (self.absolute_vol_difference, 'absolute volume difference'),
            'absolute_count_difference': (self.absolute_difference, 'absolute difference'),
            'mean_diceimage': (self.mean_dsc_perimage, 'mean dice'),
            'mean_diceover': (self.mean_dsc_whenoverlap, 'mean dice')

        }
        self.threshold = threshold
        self.thresh_assign = thresh_assign

        self.list_labels = list_labels
        self.flag_empty = empty
        self.measures = measures if measures is not None else self.m_dict
        self.connectivity = connectivity
        self.pixdim = pixdim

        if isinstance(seg_img, np.ndarray):
            # --> prepare images

            # original prediction image, voxels have values ranging between 0 and 1
            self.seg_img = seg_img.copy()

            # remove decompression errors when loading nifti, voxel have values 0 or 1
            self.ref = np.around(ref_img)

            # clipping prediction, voxels have values 0 or between threshold and 1
            self.seg_clipped = np.where(self.seg_img >= threshold, self.seg_img, np.zeros_like(self.seg_img))

            # binarize prediction, voxels have values 0 or 1
            self.seg_binary = np.where(self.seg_img >= threshold, np.ones_like(self.seg_img),
                                       np.zeros_like(self.seg_img))

            # get connected components maps
            self.seg_cc = MorphologyOps(self.seg_binary.astype(bool), self.connectivity).foreground_component()[0]
            self.ref_cc = MorphologyOps(self.ref.astype(bool), self.connectivity).foreground_component()[0]

            # check special cases
            #   case reference is empty (all false positive)
            self.flag_ref_empty = np.sum(self.ref) == 0
            #   case segmentation is empty (all false negative)
            self.flag_seg_empty = (np.sum(self.seg_clipped) == 0)

            if self.flag_seg_empty:
                print('WARNING: segmentation empty, all elements in ref are false negatives')
                self.df_classif = self.all_fn()

            elif self.flag_ref_empty:
                print('WARNING: reference empty, all elements in seg are false positives')
                self.df_classif = self.all_fp()

            elif analysis == 'epvs':
                self.df_classif = self.assign_tp_fp()
            elif analysis in ['lacunes', 'microbleeds']:
                self.df_classif = self.assign_tp_fp_distance()
            else:
                raise ValueError("Wrong task given: " + analysis)

            assert len(self.df_classif) == np.amax(self.seg_cc), \
                "Error df_classif has %d elements and prediction map has %d" % (len(self.df_classif),
                                                                                np.amax(self.seg_cc))

    def all_fp(self):
        '''
        Create a matching dataframe considering every single segmentation element as a false positive element
        '''
        label_seg = self.seg_cc
        list_matching = []
        for s in range(1, np.max(label_seg) + 1):
            dict_matching = {}
            dict_matching['seg'] = s
            dict_matching['ref'] = -1
            dict_matching['category'] = 'FP'
            list_matching.append(dict_matching)
        df_matching = pd.DataFrame.from_dict(list_matching)
        return df_matching

    def all_fn(self):
        '''
        Provide an empty matching dataframe when all is false negatives (i.e seg is empty)
        '''
        df_matching = pd.DataFrame(columns=['seg', 'ref', 'category'])
        return df_matching

    def assign_tp_fp_distance(self):
        print('Using distance threshold (dist < %.2f) for microbleed or lacune task' % self.thresh_assign)
        label_ref = self.ref_cc
        label_seg = self.seg_cc
        list_com_ref = []
        list_com_seg = []
        for r in range(1, np.max(label_ref) + 1):
            com_ref = {}
            temp_ref = np.where(label_ref==r, np.ones_like(label_ref), np.zeros_like(label_ref))
            com_tmp = ndimage.center_of_mass(temp_ref)
            com_ref['label'] = r
            com_ref['x'] = com_tmp[0]
            com_ref['y'] = com_tmp[1]
            com_ref['z'] = com_tmp[2]
            list_com_ref.append(com_ref)
        for s in range(1, np.max(label_seg) + 1):
            com_seg = {}
            temp_seg = np.where(label_seg==s,np.ones_like(label_seg), np.zeros_like(label_seg))
            com_tmp = ndimage.center_of_mass(temp_seg)
            com_seg['label'] = s
            com_seg['x'] = com_tmp[0]
            com_seg['y'] = com_tmp[1]
            com_seg['z'] = com_tmp[2]
            list_com_seg.append(com_seg)
        df_seg = pd.DataFrame.from_dict(list_com_seg)
        df_ref = pd.DataFrame.from_dict(list_com_ref)
        com_seg_array = np.asarray(df_seg[['x','y','z']].copy())
        com_ref_array = np.asarray(df_ref[['x','y','z']].copy())
        paired_dist = cdist(com_seg_array, com_ref_array)
        possible_match = np.where(paired_dist < self.thresh_assign, paired_dist, 1000. * np.ones_like(paired_dist))
        list_matching = []
        for s in range(0, np.max(label_seg)):
            row_valid = possible_match[s, :]
            row_paired_dist = paired_dist[s, :]

            dict_matching = {}
            dict_matching['seg'] = s+1
            if np.min(row_valid) == 1000:
                dict_matching['category'] = 'FP'
                dict_matching['distance'] = 1000.
                dict_matching['ref'] = -1
            else:
                dict_matching['category'] = 'PossibleTP'
                dict_matching['distance'] = np.min(row_valid)
                dict_matching['ref'] = np.argmin(row_valid) + 1

            dict_matching['min_original_distance'] = np.min(row_paired_dist)

            list_matching.append(dict_matching)
        df_matching = pd.DataFrame.from_dict(list_matching)
        # compute counts per ref label (see if ref elements are selected multiple times)
        df_matching['count_ref'] = df_matching.groupby('ref')['ref'].transform('count')
        # easy cases are ref elements that have been matched to 1 predicted element
        df_easy = df_matching[df_matching['count_ref'] == 1].copy()
        # complex cases are ref elements that have been matched to multiple predicted elements
        df_complex = df_matching[df_matching['count_ref'] > 1].copy()

        # assign FP/TP: easy cases (ref element matched with only 1 predicted element)
        df_easy['category'] = np.where(df_easy['ref'] == -1, 'FP', 'TP')

        # assign FP/TP: complex cases (ref element matched with multiple predictions)
        #df_complex['category'] = np.where(df_complex['ref'] == 0, 'FP', df_complex['category'])
        #df_complex['category'] = np.where(df_complex['distance'] < self.thresh_assign, 'TP', df_complex['category'])

        # choose element with highest overlap
        df_complex['min_distance'] = df_complex.groupby('ref')['distance'].transform('min')
        df_complex['category'] = np.where(df_complex['distance'] == df_complex['min_distance'], 'TP',
                                          df_complex['category'])

        df_complex['category'] = np.where(df_complex['ref'] == -1, 'FP', df_complex['category'])

        df_complex['category'] = np.where(df_complex['category'] == 'PossibleTP', 'FP', df_complex['category'])

        # check once more that all TP elements are below the distance threshold
        df_complex['category'] = np.where(df_complex['distance'] >= self.thresh_assign, 'FP', df_complex['category'])

        df_total = pd.concat([df_easy, df_complex])

        return df_total

    def assign_tp_fp(self):
        print('Using IOU threshold (IOU < %.2f) for epvs task' % self.thresh_assign)

        label_ref = self.ref_cc
        label_seg = self.seg_cc
        max_ref = np.max(label_ref)
        max_seg = np.max(label_seg)
        list_matching = []

        # loop over predicted elements
        for s in range(1, max_seg + 1):

            dict_matching = {}
            dict_matching['seg'] = s
            temp_seg = np.where(label_seg == s, np.ones_like(label_seg), np.zeros_like(label_seg))
            overlap = temp_seg * label_ref
            unique_ref, unique_counts = np.unique(overlap, return_counts=True)
            if len(unique_ref) == 1 and unique_ref[0] == 0:
                # case: no overlapping elements
                dict_matching['category'] = 'FP'
                dict_matching['ref'] = 0
                dict_matching['overlap_seg'] = 0
                dict_matching['overlap_ref'] = 0
            elif len(unique_ref) == 2 and 0 in unique_ref:
                # case: 1 overlapping element
                dict_matching['category'] = 'PossibleTP'
                dict_matching['ref'] = unique_ref[1]
                corresponding_ref = np.where(label_ref == unique_ref[1], np.ones_like(label_ref),
                                             np.zeros_like(label_seg))

                dict_matching['overlap_seg'] = unique_counts[1] / np.sum(temp_seg)
                dict_matching['overlap_ref'] = unique_counts[1] / np.sum(corresponding_ref)
                dict_matching['iou'] = unique_counts[1] / (np.sum(temp_seg) + np.sum(corresponding_ref)
                                                           - unique_counts[1])
            else:
                # case: multiple overlapping elements -> choose element with highest overlap
                count_max = 0
                current_best = 0
                dict_matching['category'] = 'PossibleTP'
                for (r, c) in zip(unique_ref, unique_counts):
                    if r > 0:
                        if c > count_max:
                            count_max = c
                            current_best = r
                dict_matching['overlap_seg'] = count_max / np.sum(temp_seg)
                corresponding_ref = np.where(label_ref == current_best, np.ones_like(label_ref),
                                             np.zeros_like(label_seg))
                dict_matching['overlap_ref'] = count_max / np.sum(corresponding_ref)
                dict_matching['iou'] = count_max / (np.sum(temp_seg) + np.sum(corresponding_ref)
                                                           - count_max)
                dict_matching['ref'] = current_best

            list_matching.append(dict_matching)

        df_matching = pd.DataFrame.from_dict(list_matching)
        # compute counts per ref label (see if ref elements are selected multiple times)
        df_matching['count_ref'] = df_matching.groupby('ref')['ref'].transform('count')
        # easy cases are ref elements that have been matched to 1 predicted element
        df_easy = df_matching[df_matching['count_ref'] == 1].copy()
        # complex cases are ref elements that have been matched to multiple predicted elements
        df_complex = df_matching[df_matching['count_ref'] > 1].copy()

        # assign FP/TP: easy cases (ref element matched with only 1 predicted element)
        df_easy['category'] = np.where(df_easy['iou'] > self.thresh_assign, 'TP', 'FP')
        df_easy['category'] = np.where(df_easy['ref'] == 0, 'FP', df_easy['category'])

        # assign FP/TP: complex cases (ref element matched with multiple predictions)
        df_complex['category'] = np.where(df_complex['ref'] == 0, 'FP', df_complex['category'])
        df_complex['category'] = np.where(df_complex['iou'] > self.thresh_assign, 'TP', df_complex['category'])

        # choose element with highest iou
        df_complex['max_iou'] = df_complex.groupby('ref')['iou'].transform('max')
        df_complex['category'] = np.where(df_complex['iou'] == df_complex['max_iou'], 'TP', 'FP')
        # only TP if higher than iou
        df_complex['category'] = np.where(df_complex['iou'] <= self.thresh_assign, 'FP', df_complex['category'])

        df_complex['category'] = np.where(df_complex['ref'] == 0, 'FP', df_complex['category'])

        df_complex['category'] = np.where(df_complex['category'] == 'PossibleTP', 'FP', df_complex['category'])

        df_total = pd.concat([df_easy, df_complex])
        return df_total

    def mean_dsc_perimage(self):
        # not used in ranking, but included for extra insight
        df_classif = self.df_classif
        label_seg = self.seg_cc
        label_ref = self.ref_cc
        list_dsc = []
        unique_ref = np.unique(df_classif['ref'])
        for r in unique_ref:
            list_seg = np.asarray(df_classif[df_classif['ref'] == r]['seg'].copy())
            temp_ref = np.where(label_ref == r, np.ones_like(label_ref), np.zeros_like(label_ref))
            temp_seg = np.zeros_like(label_seg)
            for s in list_seg:
                temp_seg += np.where(label_seg == s, np.ones_like(label_seg), np.zeros_like(label_seg))
            dsc_temp = (2. * np.sum(temp_seg * temp_ref)) / (np.sum(temp_seg) + np.sum(temp_ref))
            list_dsc.append(dsc_temp)
        return np.mean(np.asarray(list_dsc))

    def mean_dsc_whenoverlap(self):

        # used for ranking

        # both ref/rater and segmentation/prediction images are assumed binary, using the connected components maps

        df_classif = self.df_classif
        label_seg = self.seg_cc
        label_ref = self.ref_cc

        if self.flag_ref_empty:
            # case reference empty and seg may have something (all false positive)
            return np.nan
        elif self.flag_seg_empty:
            # case segmentation is empty & ref not empty (all false negative)
            return 0

        list_dsc = []

        df_classif_tp = df_classif.loc[df_classif['category'] == 'TP']
        unique_ref = np.unique(df_classif_tp['ref'])
        # skip background label
        unique_ref = unique_ref[unique_ref > 0]

        # loop over TP ref elements
        # any predictions that overlap with the ref element are taken into account when computing the dsc
        for r in unique_ref:
            list_seg = np.asarray(df_classif[df_classif['ref'] == r]['seg'])
            temp_ref = np.where(label_ref == r, self.ref, np.zeros_like(label_ref))
            temp_seg = np.zeros_like(label_seg)
            if len(list_seg) > 0:
                for s in list_seg:
                    temp_seg += np.where(label_seg == s, np.ones_like(label_seg), np.zeros_like(label_seg))
                # DSC = 2 TP / (2TP + FP + FN) = 2 * sum(seg * ref) / (sum(seg) + sum(ref))
                dsc_temp = (2. * np.sum(temp_seg * temp_ref)) / (np.sum(temp_seg) + np.sum(temp_ref))
                list_dsc.append(dsc_temp)
        if len(list_dsc) > 0:
            return np.mean(np.asarray(list_dsc))
        else:
            # when there is no associated  segmentation or reference overlapping
            return 0.

    def absolute_difference(self):
        label_ref = self.ref_cc
        label_seg = self.seg_cc
        return np.abs(np.max(label_ref) - np.max(label_seg))

    def absolute_vol_difference(self):
        # compute volume over clipped predicted maps
        ref = self.ref
        seg = self.seg_clipped
        sum_ref = np.sum(ref)
        sum_seg = np.sum(seg)
        difference = np.abs(float(sum_seg) - float(sum_ref))
        volume_voxel = np.prod(self.pixdim)
        return difference * volume_voxel

    def f1_count_element(self):
        label_ref = self.ref_cc
        df_classif = self.df_classif

        if self.flag_ref_empty:
            # case reference empty and seg may have something (all false positive)
            return np.nan
        elif self.flag_seg_empty:
            # case segmentation is empty & ref not empty (all false negative)
            return 0

        numb_fp = df_classif[df_classif['category'] == 'FP'].shape[0]
        numb_tp = df_classif[df_classif['category'] == 'TP'].shape[0]
        # compute FN by taking the nr of ref elements minus the nr of TPs
        numb_fn = np.max(label_ref) - numb_tp

        return (2 * numb_tp) / (numb_fn + numb_fp + 2 * numb_tp)

    def errormaps_elements(self):
        """
        This functions computes the elementwise error maps
        Visualization of detection performance (metrics: F1, count difference)
        :return:
        """
        ref_cc = self.ref_cc
        seg_cc = self.seg_cc

        df_classif_tp = self.df_classif.loc[self.df_classif['category'] == 'TP'].copy()
        list_tp_ref = np.unique(df_classif_tp['ref'])
        list_tp_seg = np.unique(df_classif_tp['seg'])

        max_ref = np.max(ref_cc)
        max_seg = np.max(seg_cc)

        print(max_ref)
        print(max_seg)

        list_blobs_ref = range(1, max_ref+1)
        list_blobs_seg = range(1, max_seg+1)

        list_fn = [x for x in list_blobs_ref if x not in list_tp_ref]
        list_fp = [x for x in list_blobs_seg if x not in list_tp_seg]

        print('FN', len(list_fn))
        print('FP', len(list_fp))

        tpc_ref_map = np.zeros_like(ref_cc)
        tpc_seg_map = np.zeros_like(seg_cc)
        lab_map = np.zeros_like(ref_cc)
        fpc_map = np.zeros_like(seg_cc)
        fnc_map = np.zeros_like(ref_cc)
        for i in list_tp_ref:
            tpc_ref_map[ref_cc == i] = 1
            lab_map[ref_cc == i] += 4
        for i in list_tp_seg:
            tpc_seg_map[seg_cc == i] = 1
            lab_map[seg_cc == i] += 8
        for i in list_fn:
            fnc_map[ref_cc == i] = 1
            lab_map[ref_cc == i] += 1
        for i in list_fp:
            fpc_map[seg_cc == i] = 1
            lab_map[seg_cc == i] += 2

        print('FPC: ', np.sum(fpc_map), 'FNC: ', np.sum(fnc_map), 'TPC Seg: ', np.sum(tpc_seg_map),
              'TPC Ref: ', np.sum(tpc_ref_map), 'Count Ref: ', np.sum(self.ref),
              'Count Seg: ', np.sum(self.seg_binary), 'Count Ref+Seg: ', np.count_nonzero(self.ref + self.seg_binary),
              'FPC + FNC + sumTP: ', np.sum(fpc_map) + np.sum(fnc_map) + np.count_nonzero(tpc_ref_map+tpc_seg_map))

        dmaps = {'tp_map_ref': tpc_ref_map, 'tp_map_seg': tpc_seg_map,
                 'fn_map': fnc_map, 'fp_map': fpc_map, 'lab_map': lab_map}

        return dmaps

    def errormaps_segm(self):
        """
        This functions computes the voxelwise maps indicating which voxels are used in the DSC whenoverlap function
        Visualization of segmentation performance (metric: DSC whenoverlap)
        :return:
        """
        ref_cc = self.ref_cc
        seg_cc = self.seg_cc
        df_classif = self.df_classif

        df_classif_tp = self.df_classif.loc[self.df_classif['category'] == 'TP'].copy()
        list_tp_ref = np.unique(df_classif_tp['ref'])
        list_tp_seg = np.unique(df_classif_tp['seg'])

        max_ref = np.max(ref_cc)
        max_seg = np.max(seg_cc)

        intersection = np.multiply(self.seg_binary, self.ref)

        unique_tp_ref = list_tp_ref

        temp_seg = np.zeros_like(seg_cc)
        temp_ref = np.zeros_like(ref_cc)

        for r in unique_tp_ref:
            list_seg = np.asarray(df_classif[df_classif['ref'] == r]['seg'])
            temp_ref += np.where(ref_cc == r, ref_cc, np.zeros_like(ref_cc))

            if len(list_seg) > 0:
                for s in list_seg:
                    temp_seg += np.where(seg_cc == s, seg_cc, np.zeros_like(seg_cc))

        temp_overlap = np.where(temp_seg * temp_ref, np.ones_like(temp_seg), np.zeros_like(temp_seg))

        temp_seg[temp_overlap > 0] = max_seg * 10
        temp_ref[temp_overlap > 0] = max_ref * 10

        assert not np.any((temp_overlap - intersection) > 0), \
            "Error, diff overlap: %d" % np.sum((temp_overlap - intersection) > 0)

        rest_seg = self.seg_binary - np.where(temp_seg, 1., 0.)
        rest_ref = self.ref - np.where(temp_ref, 1., 0.)

        self.check_overlap_labels(im_cc=(rest_seg * seg_cc), lab_list=list_tp_seg)
        self.check_overlap_labels(im_cc=(rest_ref * ref_cc), lab_list=list_tp_ref)

        return {'voxmap_seg': temp_seg, 'voxmap_ref': temp_ref, 'voxmap_tp': temp_overlap,
                'voxmap_ref_rest': rest_ref, 'voxmap_seg_rest': rest_seg}

    def check_overlap_labels(self, im_cc, lab_list):
        lab_im = np.unique(im_cc)
        lab_im = lab_im[lab_im > 0]
        overlap_seg = [x for x in lab_list if x in lab_im]
        assert len(overlap_seg) == 0, "Error, overlap in labels! %d" % len(overlap_seg)

    def header_str(self):
        result_str = [self.m_dict[key][1] for key in self.measures]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:.4f}'):
        result_str = ""
        list_space = ['com_ref', 'com_seg', 'list_labels']
        for key in self.measures:
            result = self.m_dict[key][0]()
            if key in list_space:
                result_str += ' '.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            else:
                result_str += ','.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            result_str += ','
        return result_str[:-1]  # trim the last comma
