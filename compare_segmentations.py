from __future__ import absolute_import, print_function

import os.path
import glob
import nibabel as nib
import numpy as np
import sys
import argparse
import pathlib
from evaluation_comparison.pairwise_measures import PairwiseMeasures
from nifty_utils.file_utils import (create_name_save, reorder_list_presuf)

DICT_MEASURES = {}

DICT_MEASURES['epvs'] = ('absolute_count_difference', 'f1_score', 'mean_diceover', 'absolute_volume_difference',)
DICT_MEASURES['lacunes'] = ('f1_score', 'mean_diceover', 'absolute_volume_difference')
DICT_MEASURES['microbleeds'] = ('absolute_count_difference', 'f1_score', 'mean_diceover', 'absolute_volume_difference')

OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'PairwiseMeasure'


class Parameters:
    def __init__(self, seg_path, ref_path, seg_exp, ref_exp, save_name, out_path='',
                 threshold=0.5, task='epvs', step=0.1, save_maps=True, connectivity=3, thresh_assign=0.1):
        self.save_name = save_name
        self.threshold = threshold
        self.ref_path = ref_path
        self.seg_path = seg_path
        self.ref_exp = ref_exp
        self.seg_exp = seg_exp
        self.task = task
        self.step = step
        self.save_maps = save_maps
        self.connectivity = connectivity
        self.thresh_assign = thresh_assign
        self.out_path = out_path


def run_compare(param):
    # prepare save name output csv file, assure no files are overwritten
    list_format = [param.seg_path, param.ref_path]
    if param.out_path != '':
        dir_init = param.out_path
    else:
        dir_init, name_save_init = create_name_save(list_format)
    out_name = '{}_{}_{}.csv'.format(
        OUTPUT_FILE_PREFIX,
        param.save_name,
        param.task)
    iteration = 0

    if not os.path.exists(dir_init):
        os.makedirs(dir_init)

    while os.path.exists(os.path.join(dir_init, out_name)):
        iteration += 1
        out_name = '{}_{}_{}_{}.csv'.format(
            OUTPUT_FILE_PREFIX,
            param.save_name,
            param.task, str(iteration))

    print("Writing {} to {}".format(out_name, dir_init))

    # get input files for segmentation (predictions) and ref (raters annotations)
    seg_path_list = list(param.seg_path.parts) + [param.seg_exp]
    seg_names_init = glob.glob(os.path.join(*seg_path_list))
    ref_path_list = list(param.ref_path.parts) + [param.ref_exp]
    ref_names_init = glob.glob(os.path.join(*ref_path_list))
    seg_names = []
    ref_names = []

    # matching files on subject
    ind_s, ind_r = reorder_list_presuf(seg_names_init, ref_names_init)
    print(len(ind_s))
    for i in range(0, len(ind_s)):
        if ind_s[i] > -1:
            print(i, ind_s[i])
            print(seg_names_init[i], ref_names_init[
                ind_s[i]])
            seg_names.append(seg_names_init[i])
            ref_names.append(ref_names_init[ind_s[i]])
    pair_list = list(zip(seg_names, ref_names))
    print("List of references is {}".format(ref_names))
    print("List of segmentations is {}".format(seg_names))

    # prepare a header for csv
    with open(os.path.join(dir_init, out_name), 'w+') as out_stream:

        # a trivial PairwiseMeasures obj to produce header_str
        m_headers = PairwiseMeasures(0, 0,
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
            print('>>> {} of {} evaluations, comparing {} and {}.'.format(
                i + 1, len(pair_list), ref_name, seg_name))
            seg_nii = nib.load(seg_name)
            ref_nii = nib.load(ref_name)

            # get voxel spacing/size
            voxel_sizes = seg_nii.header.get_zooms()[0:3]

            seg = np.squeeze(seg_nii.get_data())
            ref = ref_nii.get_data()

            # check image size and intensity range, if not correct skip this case
            if not seg.shape == ref.shape:
                print("WARNING: Issue of shape for comparison of %s and %s" % (seg_name, ref_name))
                continue
            if not np.all(seg) >= 0:
                print("WARNING: negatives in %s" % seg_name)
                continue
            if not np.all(seg) <= 1:
                print("WARNING: values > 1 in %s" % seg_name)
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

            pe = PairwiseMeasures(ref_img=ref, seg_img=seg, analysis=param.task, measures=measures_fin,
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
                        help='Path to folder with segmentation (prediction) files')
    parser.add_argument('-ref_path', dest='ref_path', action='store',
                        default='', type=pathlib.Path, required=True,
                        help='Path to folder with reference (annotator) files')
    parser.add_argument('-seg_exp', dest='seg_exp', metavar='seg pattern',
                        type=str, required=True,
                        help='RegExp pattern for the segmentation (prediction) files in seg_path folder')
    parser.add_argument('-ref_exp', dest='ref_exp', action='store',
                        default='', type=str, required=True,
                        help='RegExp pattern for the reference (annotator) files in ref_path folder')
    parser.add_argument('-t', dest='threshold', action='store', default=0.5,
                        type=float, help='Threshold used to clip/binarize segmentation (prediction) images')
    parser.add_argument('-m', dest='min_assign', action='store', default=0.1, type=float,
                        help='TP threshold, for EPVS this is an IOU threshold, for Microbleeds/Lacunes a distance threshold')
    parser.add_argument('-task', dest='task', action='store', type=str,
                        default='epvs', choices=['lacunes', 'microbleeds', 'epvs', ],
                        help='Specify challenge task: epvs, microbleeds or lacunes')
    parser.add_argument('-out_path', dest='out_path', action='store',
                        default='', help='path where to save results')
    parser.add_argument('-save_name', dest='save_name', action='store',
                        default='results', help='name to save csv with results')
    parser.add_argument('-c', dest='connectivity', action='store', default=3,
                        help='connectivity, 1: direct neighbors, 2: diagonal neighbors, 3: all neighbors for 3D image',
                        type=int)
    parser.add_argument('-save_maps', dest='save_maps', action='store_true',
                        help='boolean flag to indicate that the maps of differences and errors should be saved')
    try:
        args = parser.parse_args()
    except:
        # print('compare_segmentation.py -seg_path <segmentation_folder> -seg_exp <RegExp pattern segm files>'
        #       '-ref_path <reference_pattern>  -ref_exp <RegExp pattern ref files> '
        #       '-t <threshold> -task <task_type> -m <pvs: IOU threshold, mb/lac: distance threshold>'
        #       '-save_name <name for saving results csv> -out_path <path to folder to save csv in>'
        #       '-c <connectivity: 1, 2 or 3> <-save_maps>')
        parser.print_help(sys.stderr)
        sys.exit(2)

    param = Parameters(seg_path=args.seg_path[0], ref_path=args.ref_path, seg_exp=args.seg_exp,
                       ref_exp=args.ref_exp,
                       threshold=args.threshold,
                       save_name=args.save_name, task=args.task, out_path=args.out_path,
                       save_maps=args.save_maps, connectivity=args.connectivity, thresh_assign=args.min_assign)
    run_compare(param)


if __name__ == "__main__":
    main(sys.argv[1:])
