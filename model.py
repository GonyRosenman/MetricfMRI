import os
from abc import ABC, abstractmethod
from transformers import BertConfig,BertPreTrainedModel, BertModel
from nvidia_blocks import *
from general_utils import datestamp


class BaseModel(nn.Module, ABC):
    def __init__(self,**kwargs):
        super().__init__()
        self.experiment_path = kwargs.get('experiment_folder')
        self.save_all_checkpoints = kwargs.get('save_all_checkpoints')
        self.best_loss = 1000000
        self.best_accuracy = 0
        self.best_auroc = 0
        self.label_num = 1
        self.inChannels = 2 if not kwargs.get('only_per_voxel') else 1
        self.outChannels = 1
        self.model_depth = 4
        self.intermediate_vec = 2640
        self.use_cuda = kwargs.get('cuda')
        self.device_ = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
        self.shapes = kwargs.get('shapes')

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def determine_shapes(self,encoder,dim):
        def get_shape(module,input,output):
            module.input_shape = tuple(input[0].shape[-3:])
            module.output_shape = tuple(output[0].shape[-3:])
        hook1 = encoder.down_block1.register_forward_hook(get_shape)
        hook2 = encoder.down_block3.register_forward_hook(get_shape)
        input_shape = (1,encoder.inChannels,) + dim  #batch,norms,H,W,D,time
        x = torch.rand((input_shape))
        with torch.no_grad():
            encoder(x)
            del x
        self.shapes = {'dim_0':encoder.down_block1.input_shape,
                       'dim_1':encoder.down_block1.output_shape,
                       'dim_2':encoder.down_block3.input_shape,
                       'dim_3':encoder.down_block3.output_shape}
        hook1.remove()
        hook2.remove()

    def register_vars(self,**kwargs):
        #todo: verify the feasibility of removing register_vars all together and adding it to the constructor
        self.residual = kwargs.get('residual')
        intermediate_vec = 2640
        if kwargs.get('task') == 'fine_tune':
            self.dropout_rates = {'input': 0.1, 'green': 0.2,'Up_green': 0,'transformer':0.1}
        else:
            self.dropout_rates = {'input': 0.15, 'green': 0.2, 'Up_green': 0.2,'transformer':0.1}
            #self.dropout_rates = {'input': 0, 'green': 0.05, 'Up_green': 0.1, 'transformer': 0.1}
        for name in ['input', 'green','Up_green', 'transformer']:
            i = kwargs.get('dropout_' + name)
            if i is not None:
                print('modifying dropout values according to kwargs...\n{} --> {}'.format(name,i))
                self.dropout_rates[name] = i
        self.BertConfig = BertConfig(hidden_size=intermediate_vec, vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=16, max_position_embeddings=100,
                                     hidden_dropout_prob=self.dropout_rates['transformer'])
        #self.BertConfig.max_position_embeddings = 21
        #print('notice!!!!!\nchange position embedding back to 100!!!!!!!!!!!!!!!')





    def load_partial_state_dict(self, state_dict,load_cls_embedding):
        print('loading parameters onto new model...')
        name_of_loaded = []
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if 'position' in name and param.shape != own_state[name].shape:
                print('{} shapes are not correct. probably a change of sequence length...\n ignoring'.format(name))
                continue
            if not load_cls_embedding and 'cls_embedding' in name:
                print('notice: random initialization for the cls embedding layer in the fine tune task')
                continue
            if 'transformer.regression' in name and name not in own_state:
                name = name.replace('transformer.','')
                print('notice: loading old model and therefore transformer.regression_head weights are loaded into the equivalent (yet with different name) regression_head layer')
            if name not in own_state:
                print('notice: {} is not part of new model and was not loaded.'.format(name))
                continue
            param = param.data
            own_state[name].copy_(param)
            name_of_loaded.append(name)
        for name in own_state.keys():
            if name not in name_of_loaded:
                print('notice: {} was initialized randomly'.format(name))

    def save_checkpoint(self, directory, title, epoch, optimizer=None, schedule=None,**metrics):
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':self.state_dict(),
            'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
            'epoch':epoch}
        if schedule is not None:
            ckpt_dict['schedule_state_dict'] = schedule.state_dict()
            ckpt_dict['lr'] = schedule.get_last_lr()[0]
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path
        for name,value in metrics.items():
            if value is not None:
                ckpt_dict[name] = value

        loss = metrics.get('val_MAE_history') if metrics.get('val_MAE_history') is not None else metrics.get('total_val_loss_history')
        accuracy = metrics.get('val_Balanced_Accuracy_history')
        auroc = metrics.get('val_AUROC_history')

        # Save the file with specific name
        core_name = title
        name = "{}_last_epoch.pth".format(core_name)
        torch.save(ckpt_dict, os.path.join(directory, name))
        if self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST_val_loss.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            if self.save_all_checkpoints:
                name_2 = "{}_{}_mid_BEST_val_loss.pth".format(core_name,datestamp())
                torch.save(ckpt_dict,os.path.join(directory,name_2))
                print('saving mid BEST model...')
            print('updating best saved model...')
        if accuracy is not None and self.best_accuracy < accuracy:
            self.best_accuracy = accuracy
            name = "{}_BEST_val_accuracy.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print('updating best saved model...')
        if auroc is not None and self.best_auroc < auroc:
            self.best_auroc = auroc
            name = "{}_BEST_val_auroc.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print('updating best saved model...')


class Encoder(BaseModel):
    def __init__(self,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.register_vars(**kwargs)
        self.down_block1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(self.inChannels, self.model_depth, kernel_size=3, stride=1, padding=1)),
            ('sp_drop0', nn.Dropout3d(self.dropout_rates['input'])),
            ('green0', GreenBlock(self.model_depth, self.model_depth, self.dropout_rates['green'])),
            ('downsize_0', nn.Conv3d(self.model_depth, self.model_depth * 2, kernel_size=3, stride=2, padding=1))]))
        self.down_block2 = nn.Sequential(OrderedDict([
            ('green10', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('green11', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('downsize_1', nn.Conv3d(self.model_depth * 2, self.model_depth * 4, kernel_size=3, stride=2, padding=1))]))
        self.down_block3 = nn.Sequential(OrderedDict([
            ('green20', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('green21', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('downsize_2', nn.Conv3d(self.model_depth * 4, self.model_depth * 8, kernel_size=3, stride=2, padding=1))]))
        self.final_block = nn.Sequential(OrderedDict([
            ('green30', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green31', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green32', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green33', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green']))]))

    def forward(self,x):
        x = self.down_block1(x)
        x = self.down_block2(x)
        x = self.down_block3(x)
        x = self.final_block(x)
        return x

class Encoder_Res(Encoder):
    def __init__(self,**kwargs):
        super(Encoder_Res, self).__init__(**kwargs)
        self.register_vars(**kwargs)

    def forward(self,x):
        x = self.down_block1(x)
        if self.residual:
            res = x
        x = self.down_block2(x)
        x = self.down_block3(x)
        x = self.final_block(x)
        return x,res

class BottleNeck_in(BaseModel):
    def __init__(self,**kwargs):
        super(BottleNeck_in, self).__init__(**kwargs)
        self.register_vars(**kwargs)
        self.reduce_dimension = nn.Sequential(OrderedDict([
            ('group_normR', nn.GroupNorm(num_channels=self.model_depth * 8, num_groups=8)),
            # ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.LeakyReLU(inplace=True)),
            ('convR0', nn.Conv3d(self.model_depth * 8, self.model_depth // 2, kernel_size=(3, 3, 3), stride=1, padding=1)),
        ]))
        flat_factor = tuple_prod(self.shapes['dim_3'])
        self.flatten = nn.Flatten()
        if (flat_factor * self.model_depth // 2) == self.intermediate_vec:
            self.into_bert = nn.Identity()
            print('flattened vec identical to intermediate vector...\ndroppping fully conneceted bottleneck...')
        else:
            self.into_bert = nn.Linear(in_features=(self.model_depth // 2) * flat_factor, out_features=self.intermediate_vec)

    def forward(self, inputs):
        x = self.reduce_dimension(inputs)
        x = self.flatten(x)
        x = self.into_bert(x)
        return x


class BottleNeck_out(BaseModel):
    def __init__(self,**kwargs):
        super(BottleNeck_out, self).__init__(**kwargs)
        self.register_vars(**kwargs)
        flat_factor = tuple_prod(self.shapes['dim_3'])
        minicube_shape = (self.model_depth // 2,) + self.shapes['dim_3']
        self.out_of_bert = nn.Linear(in_features=self.intermediate_vec, out_features=(self.model_depth // 2) * flat_factor)
        self.expand_dimension = nn.Sequential(OrderedDict([
            ('unflatten', nn.Unflatten(1, minicube_shape)),
            ('group_normR', nn.GroupNorm(num_channels=self.model_depth // 2, num_groups=2)),
            # ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.LeakyReLU(inplace=True)),
            ('convR0', nn.Conv3d(self.model_depth // 2, self.model_depth * 8, kernel_size=(3, 3, 3), stride=1, padding=1)),
        ]))

    def forward(self, x):
        x = self.out_of_bert(x)
        return self.expand_dimension(x)

class Decoder(BaseModel):
    def __init__(self,**kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.register_vars(**kwargs)
        self.decode_block = nn.Sequential(OrderedDict([
            ('upgreen0', UpGreenBlock(self.model_depth * 8, self.model_depth * 4, self.shapes['dim_2'], self.dropout_rates['Up_green'])),
            ('upgreen1', UpGreenBlock(self.model_depth * 4, self.model_depth * 2, self.shapes['dim_1'], self.dropout_rates['Up_green'])),
            ('upgreen2', UpGreenBlock(self.model_depth * 2, self.model_depth, self.shapes['dim_0'], self.dropout_rates['Up_green'])),
            ('blue_block', nn.Conv3d(self.model_depth, self.model_depth, kernel_size=3, stride=1, padding=1)),
            ('output_block', nn.Conv3d(in_channels=self.model_depth, out_channels=self.outChannels, kernel_size=1, stride=1))
        ]))

    def forward(self, x):
        x = self.decode_block(x)
        return x


class AutoEncoder(BaseModel):
    def __init__(self,dim,**kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        # ENCODING
        self.task = 'autoencoder_reconstruction'
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder,dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)

        # BottleNeck out of bert
        self.from_bert = BottleNeck_out(**kwargs)

        # DECODER
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        if x.isnan().any():
            print('nans in data!')
        batch_size, Channels_in, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, Channels_in, W, H, D)
        encoded  = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = self.from_bert(encoded)
        reconstructed_image = self.decoder(encoded)
        _, Channels_out, W, H, D = reconstructed_image.shape
        reconstructed_image = reconstructed_image.reshape(batch_size, T, Channels_out, W, H, D).permute(0, 2, 3, 4, 5, 1)
        return {'reconstructed_fmri_sequence': reconstructed_image}

class Transformer_Block(BertPreTrainedModel, BaseModel):
    def __init__(self,config,**kwargs):
        super(Transformer_Block, self).__init__(config)
        self.register_vars(**kwargs)
        self.cls_pooling = kwargs.get('cls_pooling')
        self.bert = BertModel(self.BertConfig, add_pooling_layer=self.cls_pooling)
        self.init_weights()
        if self.cls_pooling:
            self.cls_embedding = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.BertConfig.hidden_size), nn.LeakyReLU())
            batch = 3 if any([x in kwargs.get('task') for x in ['triplet','noisy','cosine']]) else kwargs.get('batch_size')
            self.register_buffer('cls_id',torch.ones((batch, 1, self.BertConfig.hidden_size)) * 0.5,persistent=False)

    def concatenate_cls(self, x):
        if self.cls_pooling:
            cls_token = self.cls_embedding(self.cls_id[:x.shape[0]])
            x = torch.cat([cls_token, x], dim=1)
        return x


    def forward(self, x ):
        inputs_embeds = self.concatenate_cls(x=x)
        outputs = self.bert(input_ids=None,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=self.BertConfig.use_return_dict
                            )

        if self.cls_pooling:
            sequence_output = outputs[0][:, 1:, :]
            pooled_cls = outputs[1]
        else:
            sequence_output = outputs[0]
            pooled_cls = sequence_output.mean(1)

        return {'sequence': sequence_output, 'cls': pooled_cls}


class Encoder_Transformer_Decoder(BaseModel):
    def __init__(self, dim,**kwargs):
        super(Encoder_Transformer_Decoder, self).__init__(**kwargs)
        self.task = 'transformer_reconstruction'
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder,dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig, **kwargs)

        # BottleNeck out of bert
        self.from_bert = BottleNeck_out(**kwargs)

        # DECODER
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        out = transformer_dict['sequence'].reshape(batch_size * T, -1)
        out = self.from_bert(out)
        reconstructed_image = self.decoder(out)
        output_dict = {'reconstructed_fmri_sequence':reconstructed_image.reshape(batch_size, T, self.outChannels, W, H, D).permute(0, 2, 3, 4, 5, 1)}
        if 'init_vector_sequence' in transformer_dict:
            output_dict['init_vector_sequence'] = transformer_dict['input_vector_sequence']
        return output_dict

class Encoder_Transformer_finetune(BaseModel):
    def __init__(self,dim,**kwargs):
        super(Encoder_Transformer_finetune, self).__init__(**kwargs)
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs)
        # finetune classifier
        if kwargs.get('fine_tune_task') == 'regression':
            self.final_activation_func = nn.LeakyReLU()
        elif kwargs.get('fine_tune_task') == 'classification':
            self.final_activation_func = nn.Identity()#nn.Softmax(dim=1)
            self.label_num = len(kwargs.get('label_dict'))
            self.label_num = len(kwargs.get('label_dict'))
        self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.label_num),self.final_activation_func)


    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        CLS = transformer_dict['cls']
        output_dict = {self.task:self.regression_head(CLS)}
        if 'init_vector_sequence' in transformer_dict:
            output_dict['init_vector_sequence'] = transformer_dict['init_vector_sequence']
        return output_dict


class Encoder_Transformer_Vector(BaseModel):
    def __init__(self,dim,**kwargs):
        super(Encoder_Transformer_Vector, self).__init__(**kwargs)
        self.task = kwargs.get('task')
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)
        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        output_dict = {}
        for name,param in transformer_dict.items():
            if param.isnan().any():
                print('nans in {}!'.format(name))
            output_dict[name] = param
        return output_dict

class Encoder_Transformer_Binary_Fingerprint(BaseModel):
    def __init__(self,dim,**kwargs):
        super(Encoder_Transformer_Binary_Fingerprint, self).__init__(**kwargs)
        self.task = kwargs.get('task')
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)
        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs)
        self.regression_head = nn.Sequential(nn.Linear(2 * self.BertConfig.hidden_size, 2),self.final_activation_func)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        anchor,positive,negative = transformer_dict['cls']
        pair1 = self.regression_head(torch.cat([anchor,positive],dim=0)).unsqueeze(0)
        pair2 = self.regression_head(torch.cat([anchor,negative],dim=0)).unsqueeze(0)
        preds = torch.cat([pair1,pair2],dim=0)
        output_dict = {'classification':preds}
        for name,param in transformer_dict.items():
            if param.isnan().any():
                print('nans in {}!'.format(name))
            output_dict[name] = param
        return output_dict


class Encoder_Transformer_Shuffle(BaseModel):
    def __init__(self,dim,**kwargs):
        super(Encoder_Transformer_Shuffle, self).__init__(**kwargs)
        self.shuffle_kernel_size = round(kwargs.get('sequence_length') * 0.2)
        self.task = kwargs.get('task')
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)
        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs)
        self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, 1),nn.Sigmoid())

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        shuffled = (torch.randint(0,2,(batch_size,),) * 1.0).to(device=self.device_)
        for i,label in enumerate(shuffled):
            if label:
                T_init = torch.randint(0,T-(self.shuffle_kernel_size-1),(1,))
                Ts = torch.arange(T)
                Ts[T_init:T_init+self.shuffle_kernel_size] = Ts[T_init:T_init+self.shuffle_kernel_size][torch.randperm(self.shuffle_kernel_size)]
                encoded[i] = encoded[i,Ts,:]
        transformer_dict = self.transformer(encoded)
        CLS = transformer_dict['cls']
        prediction = self.regression_head(CLS)
        return {'is_shuffled':prediction,'ground_truth':shuffled}

class Encoder_Transformer_DFC(BaseModel):
    def __init__(self,dim,**kwargs):
        super(Encoder_Transformer_DFC, self).__init__(**kwargs)
        self.task = kwargs.get('task')
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)
        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs)
        self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size,kwargs.get('num_parcellations')**2),nn.LeakyReLU())

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        cls = transformer_dict['cls']
        prediction = self.regression_head(cls)
        return {'dfc':prediction}

class Encoder_Transformer_Cosine(BaseModel):
    def __init__(self,dim,**kwargs):
        super(Encoder_Transformer_Cosine, self).__init__(**kwargs)
        self.task = kwargs.get('task')
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs)
        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs)
        self.transformer.register_buffer('cls_id', torch.ones((2, 1, self.BertConfig.hidden_size)) * 0.5,persistent=False)
        self.cosine = nn.CosineSimilarity()

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        assert batch_size == 2
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        transformer_dict = self.transformer(encoded)
        base,pair = transformer_dict['sequence'].mean(-2).unsqueeze(1)
        similarity = self.cosine(base,pair)
        return {'cosine_similarity':similarity}


