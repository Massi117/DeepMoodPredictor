#!/bin/bash

for i in {35..49}
do
    python loo_validation.py $i 'masks/MVP_rois/amygdala-mask.nii.gz'
done

python plot_accuracies.py