import matplotlib.pyplot as plt
from nilearn import datasets,plotting
from nilearn import image as nimg
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib
from matplotlib.lines import Line2D
from general_utils import *

class Parcellation():
    def __init__(self,atlases=['juelich','aal'],**kwargs):
        threshold = kwargs.get('parcellation_threshold')
        if threshold is None:
            threshold = 0.8
        self.work_dir = PathHandler().work_dir
        self.data_dir = self.work_dir.joinpath('parcellations')
        self.specific_parcellation_dir = self.data_dir.joinpath(('{}_' * len(atlases)).format(*atlases) + 'combined_parcellation_with_threshold_{}'.format(threshold))
        self.specific_parcellation_dir.mkdir(exist_ok=True)
        self.parcel_sizes_dict_path = self.specific_parcellation_dir.joinpath('parcel_sizes_dict.pt')
        self.parcel_color_dict_path = self.specific_parcellation_dir.joinpath('parcel_color_dict.pt')
        self.index_coordinates_dict_path = self.specific_parcellation_dir.joinpath(('parcel_coordinates_dict.pt'))
        self.combined_label_dict_path = self.specific_parcellation_dir.joinpath('combined_label_dict.pt')
        self.mask_path = self.specific_parcellation_dir.joinpath('combined_mask.pt')
        self.exemplar_image = nib.load(PathHandler().exemplar)
        self.resample_kwargs = {'target_affine': self.exemplar_image.affine,'target_shape': self.exemplar_image.shape[:-1]}
        self.slices = [slice(8, -8), slice(8, -8), slice(0, -10)]
        self.top_k_parcels_to_show = kwargs.get('top_k_parcels_to_show') if kwargs.get('top_k_parcels_to_show') is not None else 3
        self.correlative_gradients_quantile_value = kwargs.get('correlative_gradients_quantile_value') if kwargs.get('correlative_gradients_quantile_value') is not None else 0.99
        if self.mask_path.exists():
            self.load_existing_parcellation()
        else:
            self.combine_parcellations(atlases,threshold)
            self.dim = self.combined_mask.shape
            self.save_parcel_size()
            self.num_parcels = len(np.unique(self.combined_mask))
            self.convert_to_niimg(name='combined_mask',ref_mask_name=atlases[0],save=True)
            self.save_coordinates_and_color()
            self.visualzie_3d()
            self.visualzie_2d(atlases)

    def save_parcel_size(self):
        self.parcel_sizes = {}
        for parcel in np.unique(self.combined_mask):
            parcel_mask = self.combined_mask == parcel
            self.parcel_sizes[str(parcel)] = parcel_mask.sum()
        torch.save(self.parcel_sizes,self.parcel_sizes_dict_path)
        with open(str(self.parcel_sizes_dict_path).replace('.pt','.txt'),'w+') as f:
            for parcel_index,size in self.parcel_sizes.items():
                f.write('{} : {}\n'.format(self.combined_label_dict[parcel_index],size))

    def load_existing_parcellation(self):
        self.parcel_sizes = torch.load(self.parcel_sizes_dict_path)
        self.combined_label_dict = torch.load(self.combined_label_dict_path)
        try:
            self.parcel_color_dict = torch.load(self.parcel_color_dict_path)
        except FileNotFoundError:
            self.parcel_color_dict = {name: np.random.rand(3, ) for name in self.combined_label_dict.keys()}
            torch.save(self.parcel_color_dict,self.parcel_color_dict_path)
        self.index_coordinates_dict = torch.load(self.index_coordinates_dict_path)
        self.combined_mask = torch.load(self.mask_path)
        self.combined_mask_niimg = nib.load(self.specific_parcellation_dir.joinpath('combined_mask_nifti_file.nii.gz'))
        self.num_parcels = len(np.unique(self.combined_mask))
        self.dim = self.combined_mask.shape

    def load_yeo_2011(self,resample=True):
        self.yeo_2011 = datasets.fetch_atlas_yeo_2011(data_dir=self.data_dir,verbose=1)
        self.yeo_2011.maps = nib.load(self.yeo_2011['thick_7'])
        #self.yeo_2011.labels = ['Background'] + ['7Networks_'+str(x) for x in range(1,8)]
        self.yeo_2011.labels = ['Background','Visual_7','SomatoMotor_7','DorsAtt_7','VentAtt_7','Limbic_7','Control_7','Default_7']
        if resample:
            self.yeo_2011.maps = self.crop_(self.resample_(self.yeo_2011.maps))
            self.yeo_2011.mask_ = np.round(self.yeo_2011.maps.get_fdata()).squeeze()

    def load_harvard_oxford_sub(self,resample=True):
        self.harvard_oxford_sub = datasets.fetch_atlas_harvard_oxford(atlas_name='sub-maxprob-thr0-1mm',data_dir=self.data_dir,symmetric_split=True,verbose=1)
        if resample:
            self.harvard_oxford_sub.maps = self.crop_(self.resample_(self.harvard_oxford_sub.maps))
            self.harvard_oxford_sub.mask_ = np.round(self.harvard_oxford_sub.maps.get_fdata())

    def load_harvard_oxford_cort(self,resample=True):
        self.harvard_oxford_cort = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr0-1mm',data_dir=self.data_dir,symmetric_split=True,verbose=1)
        if resample:
            self.harvard_oxford_cort.maps = self.crop_(self.resample_(self.harvard_oxford_cort.maps))
            self.harvard_oxford_cort.mask_ = np.round(self.harvard_oxford_cort.maps.get_fdata())

    def load_juelich(self,resample=False):
        self.juelich= datasets.fetch_atlas_juelich(atlas_name='maxprob-thr0-1mm',data_dir=self.data_dir,symmetric_split=True)
        if resample:
            self.juelich.maps = self.crop_(self.resample_(self.juelich.maps))
            self.juelich.mask_ = np.round(self.juelich.maps.get_fdata())

    def load_aal(self,resample=False):
        self.aal = datasets.fetch_atlas_aal(data_dir=self.data_dir)
        self.aal.labels = ['Background'] + self.aal.labels
        if resample:
            self.aal.maps = self.crop_(self.resample_(nib.load(self.aal.maps)))
            self.aal.mask_ = np.round(self.aal.maps.get_fdata())

    def load_schaefer(self,resample=False):
        self.schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400,resolution_mm=1,data_dir=self.data_dir)
        self.schaefer.labels = ['Background'] + [x.decode('utf-8') for x in self.schaefer.labels]
        if resample:
            self.schaefer.maps = self.crop_(self.resample_(nib.load(self.schaefer.maps)))
            self.schaefer.mask_ = np.round(self.schaefer.maps.get_fdata())

    def resample_(self,niimg):
        return nimg.resample_img(niimg,**self.resample_kwargs)

    def crop_(self,niimg):
        return nimg.image._crop_img_to(niimg,self.slices)

    def convert_to_niimg(self,name,ref_mask_name,save=False):
        ref_mask_niimg = getattr(self,ref_mask_name)
        niimg = nimg.new_img_like(ref_mask_niimg.maps,getattr(self,name))
        if save:
            nib.save(niimg,self.specific_parcellation_dir.joinpath(name + '_nifti_file.nii.gz'))
        setattr(self,name + '_niimg',niimg)

    def save_coordinates_and_color(self):
        self.index_coordinates_dict = {}
        self.parcel_color_dict = {}
        coords,labels = plotting.find_parcellation_cut_coords(self.combined_mask_niimg,return_label_names=True)
        with open(self.specific_parcellation_dir.joinpath('parcellation_report_{}.txt'.format(datestamp())), 'w+') as f:
            for coord,label in zip(coords,labels):
                f.write('name - {} - index - {} - coordinates - {}\n'.format(self.combined_label_dict[str(label)],label,coord))
                self.index_coordinates_dict[str(label)] = coord
        self.parcel_color_dict = {name:np.random.rand(3,) for name in self.combined_label_dict.keys()}
        torch.save(self.index_coordinates_dict,self.index_coordinates_dict_path)
        torch.save(self.parcel_color_dict,self.parcel_color_dict_path)

    def combine_parcellations(self,atlases,threshold):
        report_combinations = ''
        self.threshold = threshold
        for mask in atlases:
            getattr(self,'load_' + mask)(resample=True)
        base = getattr(self,atlases[0]) #first mask in atlases serves as a base mask for initial combined mask
        self.combined_mask = base.mask_.copy()
        self.combined_label_dict = {str(x): y for x, y in zip(np.unique(self.combined_mask), base.labels)}
        if len(atlases) > 1:
            sec = getattr(self,atlases[1])
            for i,sec_c_parcel in enumerate(np.unique(sec.mask_)):
                if sec_c_parcel == 0:
                    continue
                sec_c_parcel_name = sec.labels[i]
                sec_c_parcel_mask = sec.mask_ == sec_c_parcel
                # original mask "empty" areas are equal to zero so this line finds the voxels that are "empty" in the original mask and associate with a ROI in the secondary mask
                voxels_to_replace = (self.combined_mask == 0) * sec_c_parcel_mask
                if voxels_to_replace.sum() / sec_c_parcel_mask.sum() > self.threshold:
                    new_parcel = float(len(self.combined_label_dict))
                    self.combined_label_dict[str(new_parcel)] = sec_c_parcel_name
                    self.combined_mask[voxels_to_replace] = new_parcel
        torch.save(self.combined_mask,self.mask_path)
        torch.save(self.combined_label_dict,self.combined_label_dict_path)

    def visualzie_2d(self,atlases):
        base = getattr(self, atlases[0])
        sec = getattr(self, atlases[1]) if len(atlases) > 1 else None
        self.visualization_folder = self.specific_parcellation_dir.joinpath('parcellation_visualizations')
        self.visualization_folder.mkdir(exist_ok=True)
        self.visualzie_2d_folder = self.visualization_folder.joinpath('compare_slices')
        self.visualzie_2d_folder.mkdir(exist_ok=True)
        for depth in range(3,self.combined_mask.shape[2]-3):
            fig,axs = plt.subplots(3,1,figsize=[4,12])
            axs[0].imshow(self.combined_mask[:,:,depth])
            axs[0].set_title('combined parcellation')
            axs[1].imshow(base.mask_[:,:,depth])
            axs[1].set_title(atlases[0])
            if sec:
                axs[2].imshow(sec.mask_[:,:,depth])
                axs[2].set_title(atlases[1])
            fig.savefig(self.visualzie_2d_folder.joinpath('depth_{}'.format(depth)))
            plt.close(fig)

    def visualzie_3d(self):
        self.visualization_folder = self.specific_parcellation_dir.joinpath('parcellation_visualizations')
        self.visualization_folder.mkdir(exist_ok=True)
        title = 'Combined Parcellation 3D'
        extended_title = self.visualization_folder.joinpath(title)
        coords,labels = plotting.find_parcellation_cut_coords(self.combined_mask_niimg,return_label_names=True)
        if not 'yeo_2011' in str(self.specific_parcellation_dir):
            debug = [torch.rand(1) > 0.85 for _ in range(len(coords))]
            coords = [x for i,x in enumerate(coords) if debug[i]]
            labels = [x for i, x in enumerate(labels) if debug[i]]
        labels = [self.combined_label_dict[str(float(label))] for label in labels]
        label_colors = [np.random.rand(3,) for _ in labels] if not hasattr(self,'parcel_color_dict') else list(self.parcel_color_dict.values())[1:]
        adjacency = np.zeros((len(labels),len(labels)))
        plotting.plot_connectome(adjacency,np.stack(coords),label_colors,
                                           output_file=extended_title,
                                           node_kwargs={'label':labels})
        legend_elements = [Line2D([0],[0],color=x,label=name,marker='o',linewidth=0) for x,name in zip(label_colors,labels)]
        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements, loc='center')
        fig.savefig(str(extended_title) + '_legend')
        plt.close(fig)

    def get_one_parcel_time_series(self,tensor,parcel_name):
        T = tensor.shape[-1]
        parcel_mask = self.combined_mask * self.combined_label_dict[parcel_name]
        parcel_mask = torch.from_numpy(parcel_mask).unsqueeze(-1).repeat(1, 1, 1, T)
        if tensor.dim == 5:
            batch_size = tensor.shape[0]
            parcel_mask = parcel_mask.repeat(batch_size,1,1,1)
        time_series = (tensor * parcel_mask).mean(dim=list(range(1,len(tensor.shape)-1)))
        return time_series

    def corr_mat_from_parcellated_time_series(self,parcellated_time_series):
        cormat = torch.from_numpy(np.corrcoef(parcellated_time_series.numpy())).nan_to_num()
        return cormat

    def corr_mat_from_tensor4d(self,tensor4d):
        parcellated_time_series = self.get_parcellated_time_series(tensor4d)
        cormat = self.corr_mat_from_parcellated_time_series(parcellated_time_series)
        return cormat

    def get_parcellated_time_series(self,tensor4d):
        tensor4d = tensor4d.cpu()
        T = tensor4d.shape[3]
        parcellated_time_series = torch.zeros(self.num_parcels,T)
        for parcel in np.unique(self.combined_mask):
            parcel_mask = self.combined_mask == parcel
            parcel_size = self.parcel_sizes[str(parcel)]
            parcel_mask = torch.from_numpy(parcel_mask).unsqueeze(-1).repeat(1,1,1,T)
            parcellated_time_series[int(parcel),:] = (tensor4d * parcel_mask).sum(dim=[0,1,2]) / parcel_size
        return parcellated_time_series

    def get_parcellated_volume(self,tensor3d):
        parcellated_volume = torch.zeros(self.num_parcels)
        for parcel in np.unique(self.combined_mask):
            parcel_mask = self.combined_mask == parcel
            parcel_size = self.parcel_sizes[str(parcel)]
            parcellated_volume[int(parcel)] = (tensor3d.cpu() * parcel_mask).sum() / parcel_size
        return parcellated_volume


    def parcellated_volume_to_bar_chart(self,ax,title,array,error_bar=None):
        indices = [str(x) for x in range(len(array))]
        ax.bar(indices, array,yerr=error_bar)
        ax.set_title(title)
        ax.set_xlim(-1, len(array))
        ax.tick_params(labelsize=8,rotation=90)

    def plot_correlative_gradients(self, corr_mat, parcel_values, figure_path, p_val_mat=None, p_val_orc=None):
        corr_mat = corr_mat[1:, 1:]
        parcel_values = parcel_values[1:]
        p_val_mat = p_val_mat[1:, 1:] if p_val_mat is not None else p_val_mat
        p_val_orc = p_val_orc[1:] if p_val_orc is not None else p_val_orc
        top_parcel_values, top_parcel_indices = torch.topk(abs(parcel_values), self.top_k_parcels_to_show)
        if p_val_mat is not None:
            for row in range(p_val_mat.shape[0]):
                for col in range(p_val_mat.shape[1]):
                    if p_val_mat[row, col] > 0.05:
                        corr_mat[row, col] = 0
            p_val_list = [] if p_val_mat is not None else None
        node_indices = []
        node_coords = []
        node_size = [0 for _ in range(self.num_parcels - 1)]
        node_color = []
        node_names = []
        # remove background
        FC_mat = torch.zeros(corr_mat.shape)
        for strong_node_value, strong_node_index in zip(top_parcel_values, top_parcel_indices):
            node_size[strong_node_index] = 100
            k = 5
            node_connections = corr_mat[:, strong_node_index]
            connection_values, connection_indices = torch.topk(abs(node_connections), k)
            while any(connection_values <= 0):
                k -= 1
                connection_values, connection_indices = torch.topk(abs(node_connections), k)
                if k == 0:
                    connection_values, connection_indices = [], []
            connection_values = node_connections[connection_indices]
            for connection_value, connection_index in zip(connection_values, connection_indices):
                node_size[connection_index] = 100 if connection_index == strong_node_index else 35
                FC_mat[connection_index, strong_node_index] = connection_value
                FC_mat[strong_node_index, connection_index] = connection_value
        for strong_node_value, strong_node_index in zip(top_parcel_values, top_parcel_indices):
            node_size[strong_node_index] = 100
        for node_index in range(self.num_parcels - 1):
            name = self.combined_label_dict[str(node_index + 1.0)]
            name += ' (+)' if parcel_values[node_index] > 0 else ' (-)'
            if p_val_orc is not None and p_val_orc[node_index] < 0.01:
                name += ' p<0.01'
            name = name.replace('division', '')
            name = name.replace(' part', '')
            node_names.append(name)
            node_indices.append(node_index + 1.0)
            node_coords.append(self.index_coordinates_dict[str(node_index + 1.0)])
            node_color.append(self.parcel_color_dict[str(node_index + 1.0)])
        plotting.plot_connectome(FC_mat, np.stack(node_coords), node_color, node_size, output_file=figure_path)
        legend_elements = []
        for x, name, power in zip(node_color, node_names, node_size):
            if power == 100:
                legend_elements.append(Line2D([0], [0], color=x, label=name, marker='o', linewidth=0, markersize=10))
            elif power == 35:
                legend_elements.append(Line2D([0], [0], color=x, label=name, marker='o', linewidth=0))
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.legend(handles=legend_elements, loc='center')
        fig.savefig(str(figure_path).replace('.png', '') + '_legend.png')
        plt.close(fig)
        if p_val_mat is not None:
            with open(str(figure_path).replace('.png', '.txt'), 'a+') as f:
                f.write(
                    'mean p value of pairs that are connecting a strong parcel and a strongly correlated parcel - {}'.format(
                        p_val_list))

    def create_background_mask(self):
        self.background_mask = nib.load(
            r'D:\users\Gony\ayam\sub-003\func\sub-003_task-batns_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii')
        self.background_mask = self.crop_(self.resample_(self.background_mask))
        self.background_mask = np.round(self.background_mask.get_fdata())

    def apply_background_mask(self, tensor3d):
        if not hasattr(self, 'background_mask'):
            self.create_background_mask()
        masked = torch.from_numpy(self.background_mask) * tensor3d
        masked = masked.to(dtype=tensor3d.dtype)
        return masked

    def extract_important_pairs(self,corr_mat):
        corr_mat = abs(corr_mat)
        quantile = torch.quantile(torch.tensor([x.item() for x in torch.triu(corr_mat, diagonal=1).reshape(-1) if x != 0]),self.correlative_gradients_quantile_value)
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                if corr_mat[i][j] < quantile or corr_mat[i][j] == 1:
                    corr_mat[i][j] = 0
        return corr_mat
        #result = []
        #final_result = []
        #corr_mat_reduced = torch.triu(corr_mat, diagonal=1)
        #for i in range(corr_mat.shape[0]):
        #    for j in range(corr_mat.shape[1]):
        #        value = corr_mat_reduced[i][j]
        #        if value <= 0:
        #            continue
        #        result.append((i, j, value))
        #values = torch.tensor([x[2] for x in result])
        #threshold = torch.quantile(values, 0.95)
        #for x in result:
        #    if x[2] >= threshold:
        #        final_result.append({'value':x[2],
        #                             'ind_1':x[0],
        #                             'name_1':self.combined_label_dict[str(x[0])],
        #                             'coordinates_1':self.index_coordinates_dict[str(x[0])],
        #                             'ind_2':x[1],
        #                             'name_2':self.combined_label_dict[str(x[1])],
        #                             'coordinates_2':self.index_coordinates_dict[str(x[1])]})
        #return final_result

class RawDataParcellation(Parcellation):
    def __init__(self,atlases=['juelich','aal'],**kwargs):
        super(RawDataParcellation, self).__init__(atlases,**kwargs)
        self.masker = NiftiLabelsMasker(labels_img=self.combined_mask_niimg,
                                        labels=list(self.combined_label_dict.values()),
                                        standardize=True,
                                        memory=str(self.specific_parcellation_dir),
                                        verbose=5)
    def extract_time_signal(self,fmri_files,confounds):
        timeseries = self.masker.fit_transform(fmri_files,confounds)
        return timeseries

if __name__ == '__main__':
    P = Parcellation()
