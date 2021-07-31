import os
from difflib import SequenceMatcher
import numpy as np
from .matching_filename import match_first_degree


def expand_to_5d(img_data):
    """
    Expands an array up to 5d if it is not the case yet
    :param img_data:
    :return:
    """
    while img_data.ndim < 5:
        img_data = np.expand_dims(img_data, axis=-1)
    return img_data


def split_filename(file_name):
    """
    Operation on filename to separate path, basename and extension of a filename
    :param file_name: Filename to treat
    :return pth, fname, ext:
    """
    pth = os.path.dirname(file_name)
    fname = os.path.basename(file_name)

    ext = None
    for special_ext in '.nii', '.nii.gz':
        ext_len = len(special_ext)
        if fname[-ext_len:].lower() == special_ext:
            ext = fname[-ext_len:]
            fname = fname[:-ext_len] if len(fname) > ext_len else ''
            break
    if ext is None:
        fname, ext = os.path.splitext(fname)
    return pth, fname, ext


def reorder_list(list_seg, list_ref):
    """
    Reorder list of segmentation and reference images to have matching pairs
    based on filenames
    :param list_seg: list of segmentation files
    :param list_ref: list of reference files
    :return:
    """
    new_seg = list(list_seg)
    new_ref = list(list_ref)
    common_seg = find_longest(list_seg)
    common_ref = find_longest(list_ref)
    # common_seg_sub = list_seg[0][common_seg.a:common_seg.a+common_seg.size]
    # common_ref_sub = list_ref[0][common_ref.a:common_ref.a + common_ref.size]
    print(common_seg, common_ref, "are common")
    for s in range(0, len(new_seg)):
        new_seg[s] = new_seg[s].replace(common_seg, '')
    for r in range(0, len(new_ref)):
        new_ref[r] = new_ref[r].replace(common_ref, '')
    common_seg2 = find_longest(new_seg)
    common_ref2 = find_longest(new_ref)
    # common_seg_sub = new_seg[0][common_seg.a:common_seg.a + common_seg.size]
    # common_ref_sub = new_ref[0][common_ref.a:common_ref.a + common_ref.size]
    for s in range(0, len(new_seg)):
        new_seg[s] = new_seg[s].replace(common_seg2, '')
    for r in range(0, len(new_ref)):
        new_ref[r] = new_ref[r].replace(common_ref2, '')
    print(new_ref, new_seg)
    _, _, ind_s, ind_r = match_first_degree(new_seg, new_ref)
    print(ind_s, ind_r)
    return ind_s, ind_r


def reorder_list_presuf(list_seg, list_ref):
    """
    Reorder list of segmentation and reference files using prefix and
    suffixes of different files
    :param list_seg: list of segmentation files
    :param list_ref: list of reference files
    :return:
    """
    new_seg = list(list_seg)
    new_ref = list(list_ref)
    pre_seg, suf_seg = find_prefix_suffix(list_seg)
    pre_ref, suf_ref = find_prefix_suffix(list_ref)

    for s in range(0, len(new_seg)):
        if pre_seg is not None:
            new_seg[s] = new_seg[s].replace(pre_seg, '')
        if suf_seg is not None:
            new_seg[s] = new_seg[s].replace(suf_seg, '')
    for r in range(0, len(new_ref)):
        if pre_ref is not None:
            new_ref[r] = new_ref[r].replace(pre_ref, '')
        if suf_ref is not None:
            new_ref[r] = new_ref[r].replace(suf_ref, '')
    print(new_ref, new_seg)
    _, _, ind_s, ind_r = match_first_degree(new_seg, new_ref)
    print(ind_s, ind_r)
    return ind_s, ind_r


def find_prefix_suffix(list_seg):
    """
    Find common prefix and suffix in list of files
    :param list_seg: list of filenames to analyse
    :return: longest prefix and suffix
    """
    comp_s = SequenceMatcher()
    initial = list_seg[0]
    prefix_fin = None
    suffix_fin = None
    for i in range(1, len(list_seg)):
        comp_s.set_seqs(initial, list_seg[i])
        all_poss = comp_s.get_matching_blocks()
        if all_poss[0].a == 0:
            prefix = initial[0:all_poss[0].size]
        else:
            prefix = ''
        comp_pre = SequenceMatcher()
        if prefix_fin is None:
            prefix_fin = prefix
        comp_pre.set_seqs(prefix, prefix_fin)
        pre_poss = comp_pre.get_matching_blocks()

        prefix_fin = prefix[0:pre_poss[0].size]
        if all_poss[-1].size == 0:
            suffix = initial[all_poss[-2].a: all_poss[-2].a+all_poss[-2].size]
        else:
            suffix = initial[all_poss[-1].a:]
        comp_suf = SequenceMatcher()
        if suffix_fin is None:
            suffix_fin = suffix
        comp_suf.set_seqs(suffix, suffix_fin)
        suf_poss = comp_suf.get_matching_blocks()
        suffix_fin = suffix[suf_poss[-2].a:]
    return prefix_fin, suffix_fin


def create_name_save(list_format):
    """
    Create the name under which to save the elements 
    :param list_format:
    :return:
    """
    list_elements = []
    common_path = os.path.split(os.path.commonprefix(list_format))[0]
    print(common_path)
    list_common = common_path.split(os.sep)
    for l in list_format:
        l = str(l)
        split_string = l.lstrip(common_path).split(os.sep)
        for s in split_string:
            if s not in list_common and s not in list_elements:
                list_elements.append(s.replace("*", '_'))
    return common_path, '_'.join(list_elements)


def find_longest(list_seg):
    """
    find the longest common string in a list of filenames
    :param list_seg: list of filenames
    :return: 
    """
    comp_s = SequenceMatcher()
    initial = list_seg[0]
    comp_s.set_seqs(initial, list_seg[1])
    all_poss = comp_s.get_matching_blocks()
    list_size = [c.size for c in all_poss]
    order = np.argsort(list_size)[::-1]
    all_poss_ordered = [all_poss[i] for i in order]
    possible_common = ['']
    len_common = [0]
    for p in all_poss_ordered:
        common = initial[p.a:p.a+p.size]
        for i in range(2, len(list_seg)):
            comp_s.set_seqs(common, list_seg[i])
            common_seg = comp_s.find_longest_match(0, len(common), 0,
                                                   len(list_seg[i]))
            size = common_seg.size
            if size == 0:
                break
            else:
                common = common[common_seg.a: common_seg.a + common_seg.size]

        if len(common) > 0:
            possible_common.append(common)
            len_common.append(len(common))
    return possible_common[np.argmax(len_common)]

# def find_longest(list_seg):
#     comp_s = SequenceMatcher()
#     comp_s.set_seqs(list_seg[0], list_seg[-1])
#     common_seg = comp_s.find_longest_match(0, len(list_seg[0]), 0,
#                                            len(list_seg[-1]))
#     size = common_seg.size
#     for s in range(2, len(list_seg)):
#         comp_s.set_seq2(list_seg[s])
#
#         common_seg_temp = comp_s.find_longest_match(0, len(list_seg[0]), 0,
#                                                     len(list_seg[-1]))
#         if size > common_seg_temp.size:
#             size = common_seg_temp.size
#             common_seg = common_seg_temp
#             print(list_seg[0][common_seg.a:common_seg.a + size])
#     return common_seg