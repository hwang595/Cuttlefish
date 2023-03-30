python main_finetune_imagenet.py -a resnet50 \
/l/users/hongyiwa/datasets/ILSVRC2012 \
-j 10 \
-p 100 \
-b 256 \
--resume_pruned ./prune_imagenet_resnet5050_r0.3_it0.pth.tar
