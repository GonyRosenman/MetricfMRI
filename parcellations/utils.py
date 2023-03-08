import torch
import numpy as np
from pathlib import Path
from general_utils import PathHandler

def make_overlap_description(combined_mask1_path,combined_mask2_path):
    combined_mask1_path = Path(combined_mask1_path)
    combined_mask2_path = Path(combined_mask2_path)
    text = ''
    mask1 = torch.load(combined_mask1_path.joinpath('combined_mask.pt'))
    label_dict1 = torch.load(combined_mask1_path.joinpath('combined_label_dict.pt'))
    mask2 = torch.load(combined_mask2_path.joinpath('combined_mask.pt'))
    label_dict2 = torch.load(combined_mask2_path.joinpath('combined_label_dict.pt'))
    for label1 in np.unique(mask1):
        label1_name = label_dict1[str(float(label1))]
        label1_voxels = mask1 == label1
        label1_voxels_in_mask2 = mask2[label1_voxels]
        label2_overlap_label1 = np.unique(label1_voxels_in_mask2)
        if len(label2_overlap_label1) > 1:
            overlap_names = []
            overlap_percentages = []
            label2_fractions = []
            text += 'label {} overlap:\n'.format(label1_name)
            for label2 in label2_overlap_label1:
                label2_name = label_dict2[str(float(label2))]
                overlap_names.append(label2_name)
                label2_voxels = mask2 == label2
                label2_voxels_in_label1 = (label2_voxels * label1_voxels).sum()
                label2_fraction = label2_voxels_in_label1 / label2_voxels.sum()
                label2_fractions.append(label2_fraction)
                overlap_percentage = label2_voxels_in_label1 / label1_voxels.sum()
                overlap_percentages.append(overlap_percentage)
            for label2_index in reversed(np.argsort(overlap_percentages)):
                label2_name = overlap_names[label2_index]
                p = overlap_percentages[label2_index]
                p_f = label2_fractions[label2_index]
                text += '|---- label {}\n'.format(label2_name)
                text += '|-------- fraction of {} that is overlapping with {} - {}\n'.format(label1_name,label2_name,p)
                text += '|-------- fraction of {} that is overlapping with {} - {}\n'.format(label2_name,label1_name,p_f)
    file_name = PathHandler().work_dir.joinpath('parcellations','overlap_analysis_{}_{}.txt'.format(combined_mask1_path.name,combined_mask2_path.name))
    with open(file_name,'a+') as f:
        f.write(text)
if __name__ == '__main__':
    primary_mask = '/media/data2/itzik/dev/ETFF/parcellations/yeo_2011_harvard_oxford_sub_combined_parcellation_with_threshold_0.8'
    #secondary_mask = '/media/data2/itzik/dev/ETFF/parcellations/harvard_oxford_cort_harvard_oxford_sub_combined_parcellation_with_threshold_0.8'
    secondary_mask = '/media/data2/itzik/dev/ETFF/parcellations/juelich_aal_combined_parcellation_with_threshold_0.8'
    make_overlap_description(primary_mask,secondary_mask)