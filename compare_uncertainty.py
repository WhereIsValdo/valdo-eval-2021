from __future__ import absolute_import, print_function

import os.path
import glob
import nibabel as nib
import numpy as np
import getopt
import sys
import pandas as pd
import argparse
from evaluation_comparison.pairwise_measures import PairwiseMeasures
from evaluation_comparison.pairwise_uncertainty import PairwiseUncertainty
from evaluation_comparison.pairwise_measures import MorphologyOps
from nifty_utils.file_utils import (create_name_save, reorder_list_presuf)
import pathlib

# TODO check neighborhood parameter
# TODO for PVS should run for all 3 neighborhood options (6, 18 and 26) and then average over this, for the other tasks
#  use only direct neighbors (neighborhood of 6)

# TODO for element assignment use IOU for PVS and distance threshold for microbleeds and lacunes,
#  we wrote in our design document:
#  PVS: "Elements are true positive when the intersection over union (IOU) is above the chosen threshold. The IOU
#        threshold will be chosen by comparing the segmentation masks of 2 raters (to be announced)."
#  MB/Lac: "Elements are true positive when the distance between the centers of mass is below a chosen
#           threshold (to be announced)"

# TODO check that we use the raters agreement correctly for PVS and Lacunes. In our design document:
#   "volume and counts metrics [..] will be computed per rater and the metrics will be averaged over the raters."
#   "The segmentation (DSC) and detection (Detection F1) metrics are weighed by the raters agreement"

# TODO sanity check if raters segmentation merge, check nr of elements are correct
#   TODO if rater segmentations merge, think about how to get elementwise agreement between raters

# TODO how are we currently handling the empty cases (no annotations by rater)?
#  We should only compute absolute diff for these cases and average separately over these cases right?

DICT_MEASURES = {}

DICT_MEASURES['lacunes'] = ('f1_count_uncert', 'mean_dsc_uncertainty','f1_count_uncert_seg', 'mean_dsc_uncertainty_seg')

# MEASURES_STREAMS = ('prob_match', 'summary_match')

# MEASURES_NEW = ('ref volume', 'seg volume', 'tp', 'fp', 'fn', 'outline_error',
#             'detection_error', 'dice')
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'PairwiseMeasure'


class Parameters:
    def __init__(self, seg_path, ref_path, uncert_path, seg_exp, ref_exp, uncert_exp, save_name, out_path='',
                 threshold=0.5, task='epvs', step=0.1, save_maps=True, connectivity=3, thresh_assign=0.1):
        self.save_name = save_name
        self.threshold = threshold
        self.ref_path = ref_path
        self.seg_path = seg_path
        self.uncert_path = uncert_path
        self.ref_exp = ref_exp
        self.seg_exp = seg_exp
        self.uncert_exp = uncert_exp
        self.task = task
        self.step = step
        self.save_maps = save_maps
        self.connectivity = connectivity
        self.thresh_assign = thresh_assign
        self.out_path = out_path


def run_compare(param):
    # prepare save name output csv file, assure no files are overwritten
    list_format = [param.seg_path, param.ref_path, param.uncert_path]
    if param.out_path != '':
        dir_init = param.out_path
    else:
        dir_init, name_save_init = create_name_save(list_format)
    out_name = '{}_{}_{}.csv'.format(
        OUTPUT_FILE_PREFIX,
        param.save_name,
        param.task)
    iteration = 0
    while os.path.exists(os.path.join(dir_init, out_name)):
        iteration += 1
        out_name = '{}_{}_{}_{}.csv'.format(
            OUTPUT_FILE_PREFIX,
            param.save_name,
            param.task, str(iteration))

    print("Writing {} to {}".format(out_name, dir_init))

    # get input files for segmentation (predictions) and ref (raters annotations)
    seg_names_init = glob.glob("/".join(param.seg_path.parts)+os.path.sep+param.seg_exp)
    ref_names_init = glob.glob("/".join(param.ref_path.parts)+os.path.sep+param.ref_exp)
    uncert_names_init = glob.glob("/".join(param.uncert_path.parts) + os.path.sep + param.uncert_exp)

    seg_names = []
    ref_names = []
    uncert_names = []
    # seg_names = util.list_files(param.seg_dir, param.ext)
    # ref_names = util.list_files(param.ref_dir, param.ext)

    # matching files on subject
    ind_s, ind_r = reorder_list_presuf(seg_names_init, ref_names_init)
    ind_su, ind_u = reorder_list_presuf(seg_names_init, uncert_names_init)
    print(len(ind_s))
    for i in range(0, len(ind_s)):
        if ind_s[i] > -1:
            print(i, ind_s[i])
            print(seg_names_init[i], ref_names_init[
                ind_s[i]])
            seg_names.append(seg_names_init[i])
            ref_names.append(ref_names_init[ind_s[i]])
            uncert_names.append(uncert_names_init[ind_su[i]])
    pair_list = list(zip(seg_names, ref_names, uncert_names))
    # import itertools
    # pair_list = list(itertools.product(seg_names, ref_names))
    print("List of references is {}".format(ref_names))
    print("List of segmentations is {}".format(seg_names))
    print("List of uncertainty maps is {}".format(seg_names))
    # prepare a header for csv
    with open(os.path.join(dir_init, out_name), 'w+') as out_stream:

        # a trivial PairwiseMeasures obj to produce header_str
        m_headers = PairwiseUncertainty(0, 0,0,
                                     measures=DICT_MEASURES[param.task], analysis=param.task,
                                     connectivity=param.connectivity, empty=True,
                                     thresh_assign=param.thresh_assign).header_str()

        out_stream.write("Name (ref), Name (seg)" + m_headers + '\n')

        measures_fin = DICT_MEASURES[param.task]

        for i, pair_ in enumerate(pair_list):
            seg_name = pair_[0]
            _, seg_namefin = os.path.split(seg_name)
            ref_name = pair_[1]
            _, ref_namefin = os.path.split(ref_name)
            uncert_name = pair_[2]
            _, uncert_namefin = os.path.split(uncert_name)
            print('>>> {} of {} evaluations, comparing {} and {}.'.format(
                i + 1, len(pair_list), ref_name, seg_name))
            seg_nii = nib.load(seg_name)
            ref_nii = nib.load(ref_name)
            uncert_nii = nib.load(uncert_name)


            # get voxel spacing/size
            voxel_sizes = seg_nii.header.get_zooms()[0:3]

            seg = np.squeeze(seg_nii.get_data())
            ref = ref_nii.get_data()
            uncert = np.squeeze(uncert_nii.get_data())

            # check image size and intensity range, if not correct skip this case
            if not seg.shape == ref.shape:
                print("WARNING: Issue of shape for comparison of %s and %s" % (seg_name, ref_name))
                continue
            if not np.all(seg) >= 0:
                print("WARNING: negatives in %s" % seg_name)
                continue
            if not np.all(ref) >= 0:
                print("WARNING: negatives in %s" % ref_name)
                continue
            if not np.all(ref) <= 1:
                print("WARNING: values > 1 in %s" % ref_name)
                continue
            assert (np.all(seg) >= 0)
            assert (np.all(ref) <= 1)
            assert (np.all(seg) <= 1)
            assert (np.all(ref) >= 0)
            assert (seg.shape == ref.shape)

            # Create and save nii files of map of differences (FP FN TP OEMap
            #  DE if flag_save_map is on and binary segmentation

            pe = PairwiseUncertainty(ref_img=ref, uncert_img=uncert, seg_img=seg, analysis=param.task, measures=measures_fin,
                                  connectivity=param.connectivity, pixdim=voxel_sizes, empty=True,
                                  threshold=param.threshold, thresh_assign=param.thresh_assign)

            if param.save_maps:
                # save connected component maps
                ref_lab = pe.ref_cc
                seg_lab = pe.seg_cc

                save_map_folder = os.path.join(dir_init, 'saved_maps')
                if not os.path.exists(save_map_folder):
                    os.makedirs(save_map_folder)

                label_ref_nii = nib.Nifti1Image(ref_lab, ref_nii.affine)
                label_seg_nii = nib.Nifti1Image(seg_lab, seg_nii.affine)
                name_ref_label = os.path.join(save_map_folder, 'ElementsRef_' + os.path.split(ref_name)[1])
                name_seg_label = os.path.join(save_map_folder, 'ElementsSeg_' + os.path.split(seg_name)[1])
                nib.save(label_ref_nii, name_ref_label)
                nib.save(label_seg_nii, name_seg_label)

                dmaps = pe.errormaps_elements()
                dmaps.update(pe.errormaps_segm())

                for key, value in dmaps.items():
                    im_nii = nib.Nifti1Image(value, ref_nii.affine)
                    map_save_path = os.path.join(save_map_folder, key + '_' + os.path.split(ref_name)[1])
                    nib.save(im_nii, map_save_path)

            fixed_fields = "{}, {}" \
                           ",".format(ref_namefin, seg_namefin)

            # compute metrics and write them to csv file
            out_stream.write(fixed_fields + pe.to_string(OUTPUT_FORMAT) + '\n')
            out_stream.flush()
            os.fsync(out_stream.fileno())

    out_stream.close()


def main(argv):
    parser = argparse.ArgumentParser(description='Create evaluation file when comparing two segmentations')
    parser.add_argument('-seg_path', dest='seg_path', metavar='seg pattern',
                        type=pathlib.Path, required=True, nargs='+',
                        help='RegExp pattern for the segmentation (prediction) files')
    parser.add_argument('-uncert_path', dest='uncert_path', metavar='uncert pattern',
                        type=pathlib.Path, required=True, nargs='+',
                        help='RegExp pattern for the uncertainty (prediction) files')
    parser.add_argument('-ref_path', dest='ref_path', action='store',
                        default='', type=pathlib.Path,
                        help='RegExp pattern for the reference (annotator) files')
    parser.add_argument('-seg_exp', dest='seg_exp', metavar='seg pattern',
                        type=str, required=True,
                        help='RegExp pattern for the segmentation (prediction) files')
    parser.add_argument('-uncert_exp', dest='uncert_exp', metavar='seg pattern',
                        type=str, required=True,
                        help='RegExp pattern for the uncertainty (prediction) files')
    parser.add_argument('-ref_exp', dest='ref_exp', action='store',
                        default='', type=str,
                        help='RegExp pattern for the reference (annotator) files')
    parser.add_argument('-t', dest='threshold', action='store', default=0.5,
                        type=float)
    parser.add_argument('-m', dest='min_assign', action='store', default=0.1,
                        type=float)
    parser.add_argument('-task', dest='task', action='store', type=str,
                        default='epvs', choices=['lacunes', 'microbleeds', 'epvs', ])
    parser.add_argument('-out_path', dest='out_path', action='store',
                        default='', help='path where to save results')
    parser.add_argument('-save_name', dest='save_name', action='store',
                        default='', help='name to save results')
    parser.add_argument('-c', dest='connectivity', action='store', default=3,
                        help='connectivity, 1: direct neighbors, 2: diagonal neighbors, 3: all neighbors for 3D image',
                        type=int)
    parser.add_argument('-save_maps', dest='save_maps', action='store_true',
                        help='flag to indicate that the maps of differences '
                             'and error should be saved')
    try:
        args = parser.parse_args()
        # print(args.accumulate(args.integers))
    except argparse.ArgumentTypeError:
        print('compare_segmentation.py -s <segmentation_pattern> -r '
              '<reference_pattern> -t <threshold> -task <task_type> '
              '-save_name <name for saving> -save_maps  ')

        sys.exit(2)

    param = Parameters(seg_path=args.seg_path[0], ref_path=args.ref_path, uncert_path=args.uncert_path[0],
                       seg_exp=args.seg_exp,
                       ref_exp=args.ref_exp, uncert_exp=args.uncert_exp,
                       threshold=args.threshold,
                       save_name=args.save_name, task=args.task, out_path=args.out_path,
                       save_maps=args.save_maps, connectivity=args.connectivity, thresh_assign=args.min_assign)
    run_compare(param)


if __name__ == "__main__":
    main(sys.argv[1:])
