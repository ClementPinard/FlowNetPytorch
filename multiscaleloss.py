import torch
import torch.nn as nn


def EPE(input_flow, target_flow, sparse=False, sparse_mode='zero', mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    if sparse:
        # invalid flow is defined with either one of flow coordinate to be NaN, or both coordinates to be 0
        if sparse_mode == 'zero':
            mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        elif sparse_mode == 'nan':
            mask = (target_flow[:,0] != target_flow[:,0]) | (target_flow[:,1] == target_flow[:,1])

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/EPE_map.size(0)


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = nn.functional.adaptive_max_pool2d(target, (h, w))
        else:
            target_scaled = nn.functional.adaptive_avg_pool2d(target, (h, w))
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005,0.01,0.02,0.08,0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = nn.functional.upsample(output, size=(h,w), mode='bilinear')
    return EPE(upsampled_output, target, sparse, mean=True)
