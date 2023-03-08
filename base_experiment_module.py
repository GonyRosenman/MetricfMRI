from data_preprocess_and_load.dataloaders import DataHandler
from visualizations.visualization_class import Visualizer
from model import *
from train_utils import LrHandler
from losses import temporal_regularization_hook

class BaseModule():
    """
        base class to handle training, validation, testing and explainability.
        note: the order of commands in the constructor is necessary
    """
    def __init__(self,sets,**kwargs):
        self.device = torch.device('cuda:0') if kwargs.get('cuda') else torch.device('cpu')
        self.register_args(**kwargs)
        self.lr_handler = LrHandler(**kwargs)
        self.train_loader, self.val_loader, self.test_loader = DataHandler(any(['test' in set_ for set_ in sets]),**kwargs).create_dataloaders()
        if hasattr(self,'experiment_folder') and 'baseline' in str(self.experiment_folder).lower():
            pass
        else:
            self.create_model()
            self.initialize_weights(self.load_cls_embedding)
            #print('mega debugggg!!!')
            self.visualizer = Visualizer(**kwargs)


    def register_args(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs
        assert kwargs.get('cuda') or not kwargs.get('parallel')

    def create_model(self):
        task = self.task.lower()
        try:
            dim = self.train_loader.dataset.dataset.get_input_shape()
            label_dict = self.train_loader.dataset.dataset.get_label_dict()
        except AttributeError:
            dim = self.train_loader.dataset.dataset.datasets[0].get_input_shape()
            label_dict = self.train_loader.dataset.dataset.datasets[0].get_label_dict()
        self.input_dim = dim
        self.kwargs['label_dict'] = label_dict
        if task == 'fine_tune':
            self.model = Encoder_Transformer_finetune(dim,**self.kwargs)
        elif 'autoencoder' in task and 'reconstruction' in task:
            self.model = AutoEncoder(dim,**self.kwargs)
        elif 'transformer' in task and 'reconstruction' in task:
            self.model = Encoder_Transformer_Decoder(dim,**self.kwargs)
        elif task == 'noisy_label' or task == 'subject_triplet' or task == 'cosine_similarity' or task == 'tsne':
            self.model = Encoder_Transformer_Vector(dim,**self.kwargs)
        elif task == 'shuffle':
            self.model = Encoder_Transformer_Shuffle(dim,**self.kwargs)
        elif task == 'dfc':
            self.kwargs['num_parcellations'] = self.train_loader.dataset.dataset.parcellation.num_parcels
            self.model = Encoder_Transformer_DFC(dim,**self.kwargs)
        elif task == 'fingerprint_finetune':
            self.model = Encoder_Transformer_Binary_Fingerprint(dim,**kwargs)
        if hasattr(self,'temporal_regularization') and self.temporal_regularization:
            self.model.transformer.register_forward_hook(temporal_regularization_hook)
        if self.cuda:
            self.model = self.model.cuda()
        if self.parallel:
            self._pytorch_model = torch.nn.DataParallel(self.model)
        else:
            self._pytorch_model = self.model

    def initialize_weights(self,load_cls_embedding):
        if self.loaded_model_weights_path is not None:
            state_dict = torch.load(self.loaded_model_weights_path,map_location=self.device)
            #debug

            try:
                self.lr_handler.set_lr(state_dict['lr'])
            except KeyError:
                state_dict['lr'] = state_dict['schedule_state_dict']['_last_lr'][0]
                self.lr_handler.set_lr(state_dict['lr'])
            self.model.load_partial_state_dict(state_dict['model_state_dict'],load_cls_embedding)
            self.model.loaded_model_weights_path = self.loaded_model_weights_path
        if 'shuffle' in self.task.lower() and self.freeze_weights:
            for name,param in self.model.encoder.named_parameters():
                print('freezing {} weights...'.format(name))
                param.requires_grad = False

    def eval(self,set):
        if not hasattr(self,'_pytorch_model'):
            print('notice: could not convert to eval mode because there is not model loaded...\nthis is expected if running fingerprinting on baseline!')
            return
        self.mode = set
        self._pytorch_model = self._pytorch_model.eval()

    def train(self):
        self.mode = 'train'
        self._pytorch_model = self._pytorch_model.train()