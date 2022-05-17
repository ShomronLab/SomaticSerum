#!/bin/bash

. "/data/hadasvol/tools/miniconda3/etc/profile.d/conda.sh"
conda activate torch

training_bam_dir="/data/hadasvol/projects/cancer_plasma/seqmerge/DLbams_rand"
sample_split=True
model="CnnLinear"
hid=64
batch_size=512
lr=0.00001
max_epoch=100
dp=0.005
out=MassiveLoop3

for i in {1..10};do
    echo "Training model $model with lr=$lr, hid=$hid, bs=$bs, ep=$ep"
    python ./src/main.py \
        $training_bam_dir \
        --sample_split $sample_split \
        --model $model \
        --hidden_size 64 \
        --batch_size 512 \
        --learning_rate 0.00001 \
        --dropout 0.005 \
        --max_epoch $max_epoch \
        --out $out

    rm ${training_bam_dir}/${out}/data_trainSampleSplit.pkl
    rm ${training_bam_dir}/${out}/data_testSampleSplit.pkl
    rm ${training_bam_dir}/${out}/data_valSampleSplit.pkl
    
done

