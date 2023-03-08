import numpy as np
import nibabel as nib
import torch
import sys
import os
from pathlib import Path
from nilearn import image as nimg
from nilearn.masking import apply_mask,unmask,compute_epi_mask
sys.path.append(str(Path(os.getcwd()).parent))
from parcellations.make_parcellation import RawDataParcellation
import matplotlib.pyplot as plt
import warnings


class PreprocessFMRI():
    def __init__(self,main_path,atlases=['juelich','aal'],manually_check_cutoff=False):
        self.main_path = Path(main_path)
        self.anomalies_path = self.main_path.joinpath('anomalies')
        self.anomalies_path.mkdir(exist_ok=True)
        self.total_anomaly_masks = 0
        self.manually_check_cutoff = manually_check_cutoff
        self.mask_lower_cutoff = 0.8
        self.mask_upper_cutoff = 0.9
        #anomaly detection requires manual calibration. steps:
        #1. set anomaly threshold to very large number
        #2. comment the bottom part of run_preprocess_on_scan func (exact location is shown in the function)
        #3. run the preprocessing pipeline for a few minutes, so that you will be able to collect enough samples to calibrate the threshold values
        #4. manually check that the first k masks genereated do not contain anomalies (in "main_path/anomalies"), they will serve as non verified examplars
        #5. manually check the distance of good samples, manually check the distance of bad examples. try to find a threshold that acts as a good separator.
        #6. after re-setting the anomaly threshold value and the mask cutoffs, comment the uncommented lines and run again on the entire data.
        self.all_distances = []
        self.anomaly_threshold = 10000
        self.manual_k_inspections = 20
        self.inspection_figure_depth = 30
        self.anomalies_text_path = self.anomalies_path.joinpath('first_{}_masks_disatnce.txt'.format(self.manual_k_inspections))
        open(self.anomalies_text_path,'w').close()
        self.parcellation = RawDataParcellation(atlases=atlases,parcellation_threshold=0.8, top_k_parcels_to_show=5)
        self.fc_atlas_name = self.parcellation.specific_parcellation_dir.name

    def compute_global_mask(self):
        self.global_mask = compute_epi_mask(self.example_sample,lower_cutoff=self.mask_lower_cutoff,upper_cutoff=self.mask_upper_cutoff)

    def check_anomalies(self,new_mask,file_path):
        #todo: report anomalies per voxel and not only per scan. calculate mean value per voxel, detect specific voxels that are below a certain threshold.
        if not hasattr(self,'mean_mask'):
            self.mean_mask = new_mask
            self.total_masks = 1
            return False
        self.mean_mask += new_mask
        self.total_masks += 1
        distance = abs(self.mean_mask/self.total_masks - new_mask).sum()
        if self.total_masks <= self.manual_k_inspections:
            self.all_distances.append(distance)
            current_state = 'first_{}-{}_masks'.format(self.total_masks,self.manual_k_inspections)
            warnings.warn('warning: assuming {} first samples do not contain anomalies.'.format(self.manual_k_inspections))
            print(r'user is advised to check manually for any distortion in image saved to "main_path\anomalies" folder'+'\n')
            plt.imshow(self.mean_mask[:,:,self.inspection_figure_depth]/self.total_masks)
            plt.colorbar()
            plt.savefig(self.anomalies_path.joinpath('manual_inspection_'+current_state))
            plt.close('all')
            with open(self.anomalies_text_path,'a+') as f:
                f.write(current_state + ': ' + str(self.all_distances) + '\n\n')
        if self.anomaly_threshold < distance:
            with open(self.main_path.joinpath('anomalies','anomalies_report.txt'),'a+') as f:
                f.write('detected {} distance anomaly at {}\n'.format(distance,file_path))
            return True
        else:
            self.total_masks += 1
            self.mean_mask = self.mean_mask + (new_mask - self.mean_mask) / self.total_masks
            return False

    def convert_niimg_to_TFF_tensor(self,data):
        img = self.parcellation.crop_(self.parcellation.resample_(data)).dataobj
        img = torch.from_numpy(np.asanyarray(img)).to(dtype=torch.float32)
        return img

    def verify_only_finites(self,TR,i,path):
        if not TR.isfinite().all():
            TR = TR.nan_to_num()
            text = 'found non finite values at:\n{}\nTR:{}\n'.format(str(path), i)
            with open(self.main_path.joinpath('faulty_files.txt'), 'a+') as f:
                f.write(text)
        return TR

    def check_if_complete(self,data,g_path,pv_path):
        TR_num = data.shape[-1]
        return TR_num == len(list(g_path.iterdir())) and TR_num == len(list(pv_path.iterdir()))

    def save_TRs(self,img,path):
        img = torch.split(img, 1, 3)
        for i, TR in enumerate(img):
            TR = self.verify_only_finites(TR,i,path)
            if i == 0:
                plt.imshow(TR[:,:,40,0])
                plt.savefig('del')
            torch.save(TR.clone(), path.joinpath('TR_' + str(i) + '.pt'))

    def global_normalization(self,masked_data):
        normed_data = (masked_data - masked_data.mean()) / (masked_data.std() + 1e-11)
        return normed_data

    def per_voxel_normalization(self,masked_data):
        normed_data = (masked_data - masked_data.mean(0,keepdims=True)) / (masked_data.std(0,keepdims=True) + 1e-11)
        return normed_data

    def run_cutoff_test(self,examplar_niimg):
        cutoff_test_folder = self.main_path.joinpath('cutoff_manual_testing')
        for lower_cutoff in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            for upper_cutoff in [0.5,0.6,0.7,0.8,0.9,0.95]:
                kwargs = {'lower_cutoff':lower_cutoff,'upper_cutoff':upper_cutoff,'opening':1}
                name = 'lower_{}_upper_{}'.format(lower_cutoff,upper_cutoff)
                mask = compute_epi_mask(examplar_niimg,**kwargs)
                mask = mask.get_fdata()
                img = examplar_niimg.get_fdata()
                folder = cutoff_test_folder.joinpath(name)
                folder.mkdir(exist_ok=True,parents=True)
                for depth in range(mask.shape[2]):
                    fig,axs = plt.subplots(1,2)
                    axs[0].imshow(mask[:,:,depth])
                    axs[1].imshow(img[:,:,depth,0])
                    plt.savefig(folder.joinpath('depth_{}'.format(depth)))
                    plt.close('all')
        assert False,'check manual_testing_folder\nfind optimal threshold\nuncheck manual testing optinon after recalibrating'

    def mask_data(self,niimg,return_mask=True):
        if self.manually_check_cutoff and not hasattr(self,'verified_cutoff'):
            self.run_cutoff_test(niimg)
            self.verified_cutoff = True
        if hasattr(self,'global_mask'):
            mask = self.global_mask
        else:
            mask = compute_epi_mask(niimg,lower_cutoff=self.mask_lower_cutoff,upper_cutoff=self.mask_upper_cutoff)
            self.global_mask = mask
        #resampled_mask = compute_epi_mask(niimg, lower_cutoff=self.mask_lower_cutoff, upper_cutoff=self.mask_upper_cutoff)
        resampled_mask = nimg.resample_img(mask,niimg.affine,niimg.shape[:-1],interpolation='nearest')
        masked_niimg = apply_mask(niimg,resampled_mask)
        if return_mask:
            return masked_niimg,resampled_mask
        else:
            return masked_niimg

    def run_preprocess_on_scan(self,file_path,g_norm_path=None,pv_norm_path=None,fc_path=None,confounds_path=None):
        skip_preprocessing = False
        data = nib.load(file_path)
        if self.check_if_complete(data,g_path=g_norm_path,pv_path=pv_norm_path):
            print('already finished preprocessing... {}'.format(str(file_path)))
            if fc_path.joinpath('full_scan_fc.pt').exists():
                print('also finished extracting ROI time series... {}'.format(str(file_path)))
                return
            else:
                skip_preprocessing = True
        if not skip_preprocessing:
            if hasattr(self,'example_sample') and data.shape != self.example_sample.shape:
                #debug
                data = nimg.resample_img(data,target_affine=self.example_sample.affine,target_shape=self.example_sample.shape[:-1])
                print('found different shape to sample,resampling to example shape...')
            masked_data, mask = self.mask_data(data)
            #TODO: should be thoroughly debugged
            #if self.check_anomalies(mask.get_fdata(),str(file_path)):
            #    print('found anomaly!!!\nproceed to next sample')
            #    return
            #comment all lines below for manual calibration step
            if g_norm_path is not None:
                global_normed_masked_data = self.global_normalization(masked_data.copy())
                final_form = self.convert_niimg_to_TFF_tensor(unmask(global_normed_masked_data,mask))
                self.save_TRs(final_form,g_norm_path)
            if pv_norm_path is not None:
                pv_normed_masked_data = self.per_voxel_normalization(masked_data.copy())
                final_form = self.convert_niimg_to_TFF_tensor(unmask(pv_normed_masked_data,mask))
                self.save_TRs(final_form,pv_norm_path)
        if fc_path is not None:
            confounds = str(confounds_path) if confounds_path is not None else None
            fc = self.parcellation.extract_time_signal(data,confounds)
            torch.save(torch.from_numpy(fc),fc_path.joinpath('full_scan_fc.pt'))
#
