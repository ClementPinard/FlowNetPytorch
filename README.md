# FlowNetPytorch
Pytorch implementation of FlowNet by Dosovitskiy et al.

This repository is a torch implementation of [FlowNet](http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/), by [Alexey Dosovitskiy](http://lmb.informatik.uni-freiburg.de/people/dosovits/) et al. in PyTorch. See Torch implementation [here](https://github.com/ClementPinard/FlowNetTorch)

This code is mainly inspired from official [imagenet example](https://github.com/pytorch/examples/tree/master/imagenet).
It has not been tested for multiple GPU, but it should work just as in original code.

The code provides a training example, using [the flying chair dataset](http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) , with data augmentation. An implementation for [Scene Flow Datasets](http://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) may be added in the future.

As Graph versions are no longer needed for Pytorch, the two neural network models that are currently provided are :

 - **FlowNetS**
 - **FlowNetSBN**

There is not current implementation of FlowNetC as a specific Correlation layer module would need to be written (feel free to contribute !)

##Training on Flying Chair Dataset

First, you need to download the [the flying chair dataset](http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) . It is ~64GB big and we recommend you put in a SSD Drive.

Default HyperParameters provided in `main.python` are the same as in the caffe training scripts.

Example usage for FlowNetSBN :

     python main.py /path/to/flying_chairs/ -b 8 -j 8 -a flownetsbn

We recommend you set j (number of data threads) to high if you use DataAugmentation as to avoid data loading to slow the training.

For further help you can type

	python main.py -h
  
## Note on dataset and transform function

In this repo we address the question of splitted dataset and random transformations for both input and target, which are not currently formalized in official repo. It may change greatly in the future as Pytorch gets updated.

### Flow Transformations

To allow data augmentation, we have considered rotation and translations for inputs and their result on target flow Map.
Here is a set of things to take care of in order to achieve a proper data augmentation

#### The Flow Map is directly linked to img1
If you apply a transformation on img1, you have to apply the very same to Flow Map, to get coherent origin points for flow.

#### Translation between img1 and img2
Given a translation `(tx,ty)` applied on img2, we will have
```
flow[:,:,0] += tx
flow[:,:,1] += ty
```

#### Scale
A scale applied on both img1 and img2 with a zoom parameters `alpha` multiplies the flow by the same amount
```
flow *= alpha
```

#### Rotation applied on both images
A rotation applied on both images by an angle `theta` also rotates flow vectors (`flow[i,j]`) by the same angle
```
\for_all i,j flow[i,j] = rotate(flow[i,j], theta)

rotate: x,y,theta ->  (x*cos(theta)-x*sin(theta), y*cos(theta), x*sin(theta))
```

#### Rotation applied on img2
We consider the angle `theta` small enough to linearize `cos(theta)` to 1 and `sin(theta)` to `theta` .

x flow map ( `flow[:,:,0]` ) will get a shift proportional to distance from center horizontal axis `j-h/2`
y flow map ( `flow[:,:,1]` ) will get a shift proportional to distance from center vertical axis `i-w/2`
```
\for_all i,j flow[i,j] += theta*(j-h/2), theta*(i-w/2)
```
