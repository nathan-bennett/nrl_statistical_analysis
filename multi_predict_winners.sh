#!/bin/bash
start=$1
end=$2
for i in $(seq $start $end);
do
    python3 predict_nrl_winners.py --seed $i
done
