import torch
import torch.nn as nn
import math

class MultiScaleLoss(nn.Module):

    def __init__(self, scales, downscale, weights=None, loss= 'Abs'):
        super(MultiScaleLoss,self).__init__()
        self.downscale = downscale
        self.weights = torch.Tensor(scales).fill_(1) if weights is None else torch.Tensor(weights)
        assert(len(weights) == scales)

        if type(loss) is str:
            assert(loss in ['L1','MSE','SmoothL1'])
            
            if loss == 'L1':
                self.loss = nn.L1Loss()
            elif loss == 'MSE':
                self.loss = nn.MSELoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
        else:
            self.loss = loss
        self.multiScales = [nn.AvgPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]

    def forward(self, input, target):
        if type(input) is tuple:
            out = 0
            for i,input_ in enumerate(input):
                target_ = self.multiScales[i](target)
                out += self.weights[i]*self.loss(input_,target_)
        else:
            out = self.loss(input,self.multiScales[0](target))

        return out

def multiscaleloss(scales=5, downscale=4, weights=None, loss='L1'):
    if weights is None:
        weights = (0.005,0.01,0.02,0.08,0.32) #as in original article
        #weights = (0.32,0.08,0.02,0.01,0.005)
    if scales ==1 and type(weights) is not tuple: #a single value needs a particular syntax to be considered as a tuple
        weights = (weights,)
    return MultiScaleLoss(scales,downscale,weights,loss)