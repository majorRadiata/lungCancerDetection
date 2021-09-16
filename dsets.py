import csv
import functools
import glob
import os
import SimpleITK as sitk
import numpy as np
from collections import namedtuple

# hold information for each nodule in named tuple
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'is_nodule_bool, diameter_mm, series_uid, center_xyz',
)


@functools.cache(1)
def get_candidate_info_list(require_on_disk_bool=True):
    # getting information from two .csv files by series_uid
    mhd_list = glob.glob('E:/upload/LUNA/subset*/*.mhd')
    present_on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    diameter_dict = {}
    with open('E:/upload/LUNA/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append((annotation_center_xyz, annotation_diameter_mm))

    candidate_info_list = []
    with open('E:/upload/LUNA/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in present_on_disk_set and require_on_disk_bool:
                continue

            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            candidate_diameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                    if delta_mm > annotation_diameter_mm / 4:
                        break
                    else:
                        candidate_diameter_mm = annotation_diameter_mm
                        break
            candidate_info_list.append(CandidateInfoTuple(is_nodule_bool, candidate_diameter_mm, series_uid,
                                                          candidate_center_xyz))

    candidate_info_list.sort(reverse=True)
    return candidate_info_list


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob('E:/upload/LUNA/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
