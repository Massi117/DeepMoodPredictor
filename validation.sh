#!/bin/bash

for i in {0..9}
do
    python loo_validation.py $i
done

python plot_accuracies.py