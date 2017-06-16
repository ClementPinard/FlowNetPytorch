import torch
import torch.nn as nn
import math

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,2,1)

class MultiScaleLoss(nn.Module):

    def __init__(self, scales, downscale, weights=None, loss= 'L1'):
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
                EPE_ = EPE(input_,target_)
                out += self.weights[i]*self.loss(EPE_,EPE_.detach()*0) #Compare EPE_ with A Variable of the same size, filled with zeros)
        else:
            out = self.loss(input,self.multiScales[0](target))
        return out

class MultiScaleLossSparse(MultiScaleLoss):

    def __init__(self, scales, downscale, weights=None, loss= 'L1'):
        super(MultiScaleLoss,self).__init__()
        self.multiScales = [nn.MaxPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]

    def forward(self, input, target):
        if type(input) is tuple:
            out = 0
            for i,input_ in enumerate(input):
                target_ = self.multiScales[i](target)
                input_[target_==0] = 0
                EPE_ = EPE(input_,target_)
                out += self.weights[i]*self.loss(EPE_,torch.zeros(EPE_.size()))
        else:
            target_ = self.multiScales[0](target)
            input[target_==0] = 0
            out = self.loss(input,target_)
        return out


def multiscaleloss(scales=5, downscale=4, weights=None, loss='L1', sparse=False):
    if weights is None:
        weights = (0.005,0.01,0.02,0.08,0.32) #as in original article
    if scales ==1 and type(weights) is not tuple: #a single value needs a particular syntax to be considered as a tuple
        weights = (weights,)
    return MultiScaleLoss(scales,downscale,weights,loss)