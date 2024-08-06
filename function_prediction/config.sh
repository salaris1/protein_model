#!/bin/bash

rootfolder=/home/salaris/protein_model/function_prediction/
datafolder="${rootfolder}data/"
modelfolder="${rootfolder}model/"


python train_model.py \
  --root_folder "$rootfolder" \
  --datafolder  "$datafolder"\
  --modelfolder "$modelfolder" \
  --modelname facebook/esm2_t6_8M_UR50D \
  --num_epochs 5 \
  --num_batches 8 \
  --emb_features 384 \ 
  
