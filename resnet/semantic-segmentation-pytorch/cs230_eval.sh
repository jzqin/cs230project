#!/bin/bash

python eval.py --id train3-resnet50dilated-ppm_deepsup-ngpus2-batchSize4-imgMaxSize1000-paddingConst8-segmDownsampleRate8-LR_encoder0.02-LR_decoder0.02-epoch5 --list_val /home/ky_aneur/cs230/code/cta-scripts/dev_list.odgt --root_dataset /data2/yeom/ky_aneur/sah_png/dev/ --num_class 2 --ckpt ./ckpt --visualize --result /data2/yeom/ky_aneur/results/dev/ --suffix _epoch_5.pth
