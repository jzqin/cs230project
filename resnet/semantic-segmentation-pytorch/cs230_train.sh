#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1" python train.py --num_gpus 2 --num_class 2 --root_data /data2/yeom/ky_aneur/sah_png/train --list_train ~/cs230/code/cta-scripts/train_list.odgt --num_epoch 5 --id train3 --optim adam --epoch_iters 1000
# CUDA_VISIBLE_DEVICES="0,1" python train.py --num_gpus 2 --num_class 2 --root_data /data2/yeom/ky_aneur/sah_png/test --list_train ~/cs230/code/cta-scripts/test_list.odgt --num_epoch 2 --id train3 --optim adam --arch_encoder resnet18 --epoch_iters 100
