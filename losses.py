import torch
import torch.nn as nn
from torchvision import models
from pytorch_metric_learning.distances.lp_distance import LpDistance

def make_custome_triplet_loss(temporal=False,**kwargs):
    margin = kwargs.get('margin')
    if temporal:
        margin = margin / 2
    assert margin is not None
    distance = LpDistance(normalize_embeddings=True, p=2)
    print('using margin: {}'.format(margin))
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=distance,margin=margin)
    return triplet_loss

def get_intense_voxels(yy,shape):
    y = yy.clone()
    low_quantile, high_quantile, = (0.9,0.99)
    voxels = torch.empty(shape)
    for batch in range(y.shape[0]):
        for TR in range(y.shape[-1]):
            yy = y[batch, :, :, :, TR]
            #TODO:what is this
            background = yy[0, 0, 0]
            yy[yy <= background] = 0
            yy = abs(yy)
            voxels[batch, :, :, :, :, TR] = (yy > torch.quantile(yy[yy > 0], low_quantile)).unsqueeze(0)
    return voxels.view(shape)>0

def temporal_regularization_hook(model,input,output):
    output['init_vector_sequence'] = input[0]

class Vgg16(nn.Module):
    def __init__(self,**kwargs):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False
        self.layers = kwargs.get('perceptual_layers')

    def forward(self, x):
        out = []
        h = self.to_relu_1_2(x)
        if self.layers[0]:
            out.append(h)
        #h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        if self.layers[1]:
            out.append(h)
        #h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        if self.layers[2]:
            out.append(h)
        #h_relu_3_3 = h
        #out = (h_relu_2_2, h_relu_3_3)
        return out

class Percept_Loss(nn.Module):
    def __init__(self,**kwargs):
        super(Percept_Loss, self).__init__()
        task = kwargs.get('task')
        if 'autoencoder' in task:
            self.memory_constraint = 0.25
        elif 'transformer' in task:
            self.memory_constraint = 0.05
        if 'reconstruction' in task:
            self.vgg = Vgg16(**kwargs)
            if kwargs.get('cuda'):
                self.vgg.cuda()
            if kwargs.get('parallel'):
                self.vgg_ = torch.nn.DataParallel(self.vgg)
            else:
                self.vgg_ = self.vgg
            self.loss = nn.MSELoss()

    def forward(self, input, target):
        assert input.shape == target.shape, 'input and target should have identical dimension'
        assert len(input.shape) == 6
        batch, channel, width, height, depth, T = input.shape
        num_slices = batch * T * depth
        represent = torch.randperm(num_slices)[:int(num_slices * self.memory_constraint)]
        input = input.permute(0, 5, 1, 4, 2, 3).reshape(num_slices, 1, width, height)
        target = target.permute(0, 5, 1, 4, 2, 3).reshape(num_slices, 1, width, height)
        input = input[represent, :, :, :].repeat(1,3,1,1)
        target = target[represent, :, :, :].repeat(1,3,1,1)

        input = self.vgg_(input)
        target = self.vgg_(target)
        loss = 0
        for i,j in zip(input,target):
            loss += self.loss(i,j)
        return loss


