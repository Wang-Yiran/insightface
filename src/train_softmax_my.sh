#!/usr/bin/env bash
python3 -u  train_softmax_my.py --prefix ../my_model/ --pretrained ../models/model-r50-am-lfw/model,0000 --loss-type 4 \
--margin-m 0.1 --data-dir /Users/wangyiran/Documents/dataset_train/ --per-batch-size 32 --version-se 1 --verbose 1000 \
--target valid --margin-s 64 --emb-size 512

