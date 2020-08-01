#!/bin/bash
LR=("1e-4" "5e-5" "2e-5" "1e-5")
WEIGHT_DECAY=("1e-7" "1e-6" "1e-5" "1e-4")
SCHEDULER=("True")

for sched in "${SCHEDULER[@]}";
    do
        for weight_decay in "${WEIGHT_DECAY[@]}";
            do
                for lr in "${LR[@]}";
                    do
                        python train.py --lr $lr --weight-decay $weight_decay --use-scheduler $sched
                    done
            done
    done