from tqdm import tqdm
from pathlib import Path
from data_preprocess_and_load.preprocess import PreprocessFMRI
from general_utils import save_code_to_disk,reproducibility

def main():
    ucla_nifti_files_path = '/media/data2/itzik/fmri_data/ucla_final_version'
    assert ucla_nifti_files_path is not None
    reproducibility(seed=1, cuda=False)
    save_code_to_disk(Path(ucla_nifti_files_path))
    manually_check = False
    preprocess = PreprocessFMRI(ucla_nifti_files_path,atlases=['harvard_oxford_cort','harvard_oxford_sub'],manually_check_cutoff=manually_check)
    preprocess.mask_upper_cutoff = 0.85
    preprocess.mask_lower_cutoff = 0.2
    preprocess.anomaly_threshold = 33000
    preprocess.parcellation.slices[1] = slice(16,None,None)
    #debug for different shapes
    #example_sample = None
    #preprocess.example_sample = nib.load(example_sample)
    #preprocess.compute_global_mask()
    use_confound_regressors = False
    count = 0
    #run this once to see files are loadable
    #for file in tqdm(Path(preprocess.main_path).rglob('*.nii.gz')):
    #    name = file.name
    #    file = str(file)
    #    try:
    #        nib.load(file)
    #    except EOFError:
    #        os.rename(file,preprocess.main_path.joinpath('bad_files',name))
    for file in tqdm(Path(preprocess.main_path).rglob('*.nii.gz')):
        f = file.name
        if not 'rest' in f:
            continue
        subject = f[f.find('sub'):f.find('_task')]
        task = f[f.find('task')+5:f.find('_bold')]
        if use_confound_regressors:
            confounds = file.parent.joinpath('{}_{}_task-{}_desc-confounds_regressors.tsv'.format(subject,session,task))
            confounds = confounds if confounds.exists() else None
        else:
            confounds = None
        count += 1
        g_norm_path = preprocess.main_path.joinpath('MNI_to_TRs', subject, task, 'global_normalize')
        pv_norm_path = preprocess.main_path.joinpath('MNI_to_TRs', subject, task, 'per_voxel_normalize')
        fc_path = preprocess.main_path.joinpath('functional_connectivity_{}'.format(preprocess.fc_atlas_name), subject, task, 'per_parcel_normalize')
        g_norm_path.mkdir(exist_ok=True, parents=True)
        pv_norm_path.mkdir(exist_ok=True, parents=True)
        fc_path.mkdir(exist_ok=True, parents=True)
        print('start working on subject ' + subject)
        try:
            preprocess.run_preprocess_on_scan(file,g_norm_path=g_norm_path,pv_norm_path=pv_norm_path,fc_path=fc_path,confounds_path=confounds)
        except Exception as e:
            print('found the following error:\n{}\nwith file path: {}'.format(e,str(file)))


if __name__ == '__main__':
    main()