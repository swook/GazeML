#!/bin/bash

if [ ! -d outputs ]; then
	mkdir outputs
fi
cd outputs

## ELG model
# eye_image_shape    = (108, 180)
# first_layer_stride = 3
# num_modules        = 3
# num_feature_maps   = 64
wget -Nnv https://ait.ethz.ch/projects/2018/landmarks-gaze/downloads/ELG_i180x108_f60x36_n64_m3.zip
unzip -oq ELG_i180x108_f60x36_n64_m3.zip

## ELG model
# eye_image_shape    = (36, 60)
# first_layer_stride = 1
# num_modules        = 2
# num_feature_maps   = 32
wget -Nnv https://ait.ethz.ch/projects/2018/landmarks-gaze/downloads/ELG_i60x36_f60x36_n32_m2.zip
unzip -oq ELG_i60x36_f60x36_n32_m2.zip
