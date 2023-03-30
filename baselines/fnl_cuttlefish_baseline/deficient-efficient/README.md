# Deficient Linear Transforms for Efficient Deep Learning

Substitute compressed linear transforms for deep learning. Substitute
convolutions into an existing WideResNet or DARTS network and train as
normal. Details of the research are provided in the [research
log](./research-log.md).

# tl;dr

In a deep neural network, you can replace the matrix multiply using a
weight matrix (a *linear transform*) with an alternative that uses fewer
parameters or mult-adds or both. Such as:

* [HashedNet](./models/hashed.py)
* [Tensor-Train or Tucker decompositions](./models/decomposed.py)
* [A linear ShuffleNet block](https://github.com/BayesWatch/deficient-efficient/blob/master/models/blocks.py#L131-L155)
* [A Sequence of ACDC Layers](https://github.com/gngdb/pytorch-acdc)
* [A Rank Factorized Matrix](https://github.com/BayesWatch/deficient-efficient/blob/master/models/blocks.py#L57-L81).

But, this will only train if you [scale the original weight decay used to
train the network by the compression
ratio](https://github.com/BayesWatch/deficient-efficient/blob/master/models/wide_resnet.py#L49).

## WRN-28-10 on CIFAR-10

![](./images/wrn-28-10-at.png)

## DARTS on CIFAR-10

![](./images/darts-at-params.png)

## WRN-50-2 on ImageNet

![](./images/imagenet-params-context.png)

# Citations

If you would like to cite this work, please cite our paper using the
following bibtex entry:

```
@article{gray2019separable,
  author    = {Gavin Gray and
               Elliot J. Crowley and
               Amos Storkey},
  title     = {Separable Layers Enable Structured Efficient Linear
Substitutions},
  journal   = {CoRR},
  volume    = {abs/1906.00859}, pending
  year      = {2019},
  url       = {https://arxiv.org/abs/1906.00859},
  archivePrefix = {arXiv}
  eprint    = {1906.00859}
}

```

# Acknowledgements

Based on: `https://github.com/BayesWatch/pytorch-moonshine`

