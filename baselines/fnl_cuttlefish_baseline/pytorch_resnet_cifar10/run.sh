#!/bin/bash

#for model in resnet20 resnet32 resnet44 resnet56 resnet110 resnet1202
for model in resnet20 
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model 2>&1 | tee -a log/$model"
    python -u trainer.py --spectral --wd2fd --rank-scale 0.1 --arch=$model --device 1 --save-dir=save_$model 2>&1 | tee -a log/$model
done
