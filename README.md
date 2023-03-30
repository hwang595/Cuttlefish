## Cuttlefish
The implementation for MLSys 2023 paper: "Cuttlefish: Low-rank Model Training without All The Tuning"

### Overview
---
Training low-rank neural network models has been recently shown to reduce the total number of trainable parameters, while maintaining predictive accuracy, resulting in end-to-end speedups. The catch, however, is that several extra hyper-parameters must be finely tuned, such as determining the rank of the factorization at each layer. In this work, we overcome this issue and propose Cuttlefish, an automated low-rank training method that does not require tuning the factorization hyper-parameters. Cuttlefish leverages the observation that after a few epochs of full-rank training, the stable rank of each layer (i.e., an approximation to the true rank), converges to a constant. Cuttlefish switches from full-rank to low-rank  training when the stable ranks of all layers have converged, while setting the dimension of each factorization to the respective stable rank. We show that this leads to 4.2 $\times$ smaller models compared to state-of-the-art low-rank model training techniques, and a 1.2 $\times$ end-to-end training speedup, while achieving better accuracy.

### Depdendencies
---
#### For CIFAR-10, CIFAR-100, SVHN, and ImageNet experiments
* PyTorch 1.6.0
* CUDA 11.0  
(configured with Docker container [nvcr.io/nvidia/pytorch:20.07-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags))

#### For BERT experiments
* PyTorch 1.11.0
* CUDA 11.6  
* Huggingface 4.17.0.dev0
(configured with Docker container [nvcr.io/nvidia/pytorch:22.01-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags); for BERT fine-tuning results)

#### Public AMI
We also provide the public Amazom EC2 AMI - ami-0a52060b49b26770b for your convenience where the ImageNet dataset, Docker environments, and etc are ready.

### Installation
---
#### Docker env configurations
(On a machine with Docker installed)
```
sudo docker run -t -d --gpus all --shm-size 8G --name cuttlefish -v YOUR-LOCAL-PATH:/workspace nvcr.io/nvidia/pytorch:20.07-py3
sudo docker run -t -d --gpus all --shm-size 8G --name cuttlefish-bert -v YOUR-LOCAL-PATH:/workspace nvcr.io/nvidia/pytorch:22.01-py3
```
#### For CIFAR-10, CIFAR-100, SVHN, and ImageNet experiments
```
docker exec -it cuttlefish bash # enter the docker interactive env
git clone https://github.com/hwang595/Cuttlefish.git
cd Cuttlefish
bash install.sh
```

#### For BERT experiments
```
docker exec -it cuttlefish-bert bash # enter the docker interactive env
git clone https://github.com/hwang595/Cuttlefish.git
cd Cuttlefish/transformers
bash install.sh
```

### Executing experiments
#### An Example Experiment
To run Cuttlefish on ResNet-18+CIFAR-10 Task, just simply run (without changing any script modification)
```
docker exec -it cuttlefish bash
cd Cuttlefish/scripts # this step is important
bash run_main.sh # you need to run `run_main.sh` under the Cuttlefish/scripts dir
```
The code will run until the network is trained (for 300 epochs), and at the end you should be able to see something like
```
INFO:root:### Epoch: 298, Current effective lr: 0.008
INFO:root:Epoch: 298, lowrank training ...
INFO:root:Train Epoch: 298 [0/50000 (0%)]  Loss: 0.000547
INFO:root:Train Epoch: 298 [40960/50000 (82%)]  Loss: 0.001204
INFO:root:####### Comp Time Cost for Epoch: 298 is 7.750053878784178, os time: 8.45917797088623
INFO:root:
Epoch: 298, Test set: Average loss: 0.0011, Accuracy: 9461/10000 (94.61%)

INFO:root:### Epoch: 299, Current effective lr: 0.008
INFO:root:Epoch: 299, lowrank training ...
INFO:root:Train Epoch: 299 [0/50000 (0%)]  Loss: 0.000213
INFO:root:Train Epoch: 299 [40960/50000 (82%)]  Loss: 0.000243
INFO:root:####### Comp Time Cost for Epoch: 299 is 7.7537792053222665, os time: 8.437126636505127
INFO:root:
Epoch: 299, Test set: Average loss: 0.0011, Accuracy: 9466/10000 (94.66%)

INFO:root:Comp-Time: 2513.8327392120354
INFO:root:Best-Val-Acc: 94.66
```

#### For CIFAR-10 and CIFAR-100 experiments (Table 1)
```
docker exec -it cuttlefish bash

cd Cuttlefish/scripts
bash run_main.sh
```
The script `run_main.sh` supports Cuttlefish, vanilla full-rank training, and [Pufferfish (MLSys21)](https://proceedings.mlsys.org/paper/2021/file/84d9ee44e457ddef7f2c4f25dc8fa865-Paper.pdf). Example scripts are provided below:

Cuttlefish; ResNet-18; CIFAR-10 (Frobenius Decay on; Extra BNs off)
```
#!/bin/bash

cd ..

SEED=0
TRIAL=0
EPOCHS=300
DATASET=cifar10
MODEL=resnet18

CUDA_VISIBLE_DEVICES=0 python main.py \
--arch=${MODEL} \
--mode=lowrank \
--rank-est-metric=scaled-stable-rank \
--dataset=${DATASET} \
--batch-size=1024 \
--epochs=${EPOCHS} \
--full-rank-warmup=True \
--fr-warmup-epoch=$((EPOCHS + 1)) \
--seed=${SEED} \
--lr=0.1 \
--frob-decay=True \
--extra-bns=False \
--resume=False \
--evaluate=False \
--scale-factor=8 \
--lr-warmup-epochs=5 \
--ckpt_path=./checkpoint/resnet18_best.pth \
--momentum=0.9
```

Pufferfish; ResNet-18; CIFAR-10
```
SEED=0
TRIAL=0
EPOCHS=300
DATASET=cifar10
MODEL=resnet18
WARMUP_EPOCH=80

CUDA_VISIBLE_DEVICES=0 python main.py \
--arch=${MODEL} \
--mode=pufferfish \
--rank-est-metric=scaled-stable-rank \ # this is not effective for Pufferfish
--dataset=${DATASET} \
--batch-size=1024 \
--epochs=${EPOCHS} \
--full-rank-warmup=True \
--fr-warmup-epoch=${WARMUP_EPOCH} \
--seed=${SEED} \
--lr=0.1 \
--frob-decay=False \
--extra-bns=True \ # extra BNs are always enabled in Pufferfish
--resume=False \
--evaluate=False \
--scale-factor=8 \
--lr-warmup-epochs=5 \
--ckpt_path=./checkpoint/resnet18_best.pth \
--momentum=0.9
```

Vanilla Full-rank; ResNet-18; CIFAR-10
```
SEED=0
TRIAL=0
EPOCHS=300
DATASET=cifar10
MODEL=resnet18
WARMUP_EPOCH=301 # we can directly train vanilla model by setting warmup epoch 

CUDA_VISIBLE_DEVICES=0 python main.py \
--arch=${MODEL} \
--mode=pufferfish \
--rank-est-metric=scaled-stable-rank \ # this is not effective for Pufferfish
--dataset=${DATASET} \
--batch-size=1024 \
--epochs=${EPOCHS} \
--full-rank-warmup=True \
--fr-warmup-epoch=${WARMUP_EPOCH} \
--seed=${SEED} \
--lr=0.1 \
--frob-decay=False \
--extra-bns=True \ # extra BNs are always enabled in Pufferfish
--resume=False \
--evaluate=False \
--scale-factor=8 \
--lr-warmup-epochs=5 \
--ckpt_path=./checkpoint/resnet18_best.pth \
--momentum=0.9
```

#### For ImageNet on DeiT and ResMLP experiments (Table 2)
```
docker exec -it cuttlefish bash

cd Cuttlefish/cuttlefish_deit
bash run.sh
```

#### For ImageNet on (Wide)ResNet-50 experiments (Table 3)
```
docker exec -it cuttlefish bash

cd Cuttlefish/scripts
bash run_cuttlefish_imagenet.sh

# or

bash run_pufferfish_imagenet.sh
```

#### For BERT finetuning experiments (on GLUE datasets) (Table 5)
```
docker exec -it cuttlefish-bert bash

cd Cuttlefish/transformers/examples/text-classification

bash run.sh # for Cuttlefish experiments
bash run_vanilla.sh # for vanilla BERT fine-tuning
bash run_distill_bert.sh # for distill BERT
bash run_tiny_bert.sh # for tiny BERT
```

### Benchmarking of Cuttlefish
Cuttlefish leverages tiny and lightweight benchmarking to determine the selection of hyper-parameter $K$. To conduct benchmarking (as demonstrated in Figure 4), one will need to run the following code
```
docker exec -it cuttlefish bash
cd Cuttlefish/scripts
bash run_cifar_block_benchmark.sh 
```
where `--rank-ratio` can be modified to be, e.g., 2, 4, 8, 16 (for 1/2, 1/4, 1/8, 1/16 of the experimented rank ratios). If `--rank-ratio` is set to be `0.0`, then full-rank network will be used for benchmarking.

### Baseline experiments
We compared Cuttlefish with many popular baseline methods. We also provide code we used to replicate their results.

For instance, to run the method ["SI&FD"](https://openreview.net/pdf?id=KTlJT1nof6d) (a low-rank training method with Spectural Initialization and Frobenius Decay) one can do:
```
docker exec -it cuttlefish bash

# for ResNet-18 on CIFAR-10
cd Cuttlefish/baselines/fnl_cuttlefish_baseline/pytorch_resnet_cifar10
bash run_resnet18.sh # with modifications on `--rank-scale`

# for VGG-19 on CIFAR-10
cd Cuttlefish/baselines/fnl_cuttlefish_baseline/EigenDamage-Pytorch
bash run_vgg19.sh # with modifications on `--target-ratio`
```

To run ["XNOR-Net"](https://arxiv.org/abs/1603.05279) one can do
```
docker exec -it cuttlefish bash
cd Cuttlefish/baselines/XNOR_CIFAR10
bash run_xnor_net.sh
```

To run [GraSP](https://github.com/alecwangcq/GraSP) one can do
```
docker exec -it cuttlefish bash
cd Cuttlefish/baselines/GraSP

# pre-prune resnet50 on ImageNet
bash prune_imagenet.sh

# finetuning pruned resnet50
bash finetune_imagenet.sh
```

We also leverage the great code bases, e.g., [open_lth](https://github.com/facebookresearch/open_lth) and [LC-model-compression](https://github.com/UCMerced-ML/LC-model-compression/tree/master/examples/cvpr2020) for our baseline comparisons.


### Citing Cuttlefish
If you found the code/scripts here are useful to your work, please cite Pufferfish by
```
@inproceedings{wang2023cuttlefish,
  title={Cuttlefish: Low-rank Model Training without All The Tuning},
  author={Wang, Hongyi and Agarwal, Saurabh and  U-chupala, Pongsakorn and Tanaka, Yoshiki and Xing, Eric and Papailiopoulos, Dimitris},
  journal={MLSys},
  year={2023}
}
```
