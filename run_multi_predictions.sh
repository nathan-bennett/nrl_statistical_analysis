#!/usr/bin/env bash
start=$1
end=$2

for ((i=$start; i<=$end; i++))
do
    python3 predict_nrl_winners.py --seed "$i"
done
