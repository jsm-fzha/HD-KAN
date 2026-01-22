#!/bin/bash

# =============================
# Create log folders
# =============================
mkdir -p ./logs/ShortForecasting/HDKAN/s2

model_name=HDKAN
root_path_name=./dataset/short-term

# =============================
# ILI
# =============================
data_path_name=national_illness.csv
model_id_name=illness
data_name=custom
seq_len=36

for p in 24 36 48 60; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${p} \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $p \
        --enc_in 7 \
        --train_epochs 20 \
        --patience 5 \
        --patch_len 9 \
        --alpha 0.35 \
        --batch_size 8 \
        --learning_rate 0.0003 \
        --degree1 7 \
        --degree2 7 \
        --d_model 128 \
        --dropout 0 \
        --lradj type3 \
        > logs/ShortForecasting/HDKAN/s2/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# COVID-19
# =============================
data_path_name=covid.csv
model_id_name=covid
data_name=custom2

for p in 24 36 48 60; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${p} \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $p \
        --enc_in 55 \
        --train_epochs 20 \
        --patience 5 \
        --patch_len 9 \
        --alpha 0.5 \
        --batch_size 1 \
        --learning_rate 0.0016 \
        --degree1 4 \
        --degree2 7 \
        --d_model 512 \
        --dropout 0 \
        --lradj type1 \
        > logs/ShortForecasting/HDKAN/s2/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# METR-LA
# =============================
data_path_name=METR-LA.csv
model_id_name=METR-LA
data_name=custom

for p in 24 36 48 60; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${p} \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $p \
        --enc_in 207 \
        --train_epochs 20 \
        --patience 5 \
        --patch_len 18 \
        --alpha 0.35 \
        --batch_size 16 \
        --learning_rate 0.0006 \
        --degree1 10 \
        --degree2 5 \
        --d_model 512 \
        --dropout 0.1 \
        --lradj type3 \
        > logs/ShortForecasting/HDKAN/s2/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# NASDAQ
# =============================
data_path_name=nasdaq.csv
model_id_name=nasdaq
data_name=custom

for p in 24 36 48 60; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${p} \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $p \
        --enc_in 12 \
        --train_epochs 20 \
        --patience 5 \
        --patch_len 36 \
        --alpha 0.35 \
        --batch_size 16 \
        --learning_rate 0.000025 \
        --degree1 7 \
        --degree2 5 \
        --d_model 128 \
        --dropout 0 \
        --lradj type3 \
        > logs/ShortForecasting/HDKAN/s2/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# Wiki
# =============================
data_path_name=wiki_mini.csv
model_id_name=wiki
data_name=custom

for p in 24 36 48 60; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${p} \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $p \
        --enc_in 99 \
        --train_epochs 20 \
        --patience 5 \
        --patch_len 36 \
        --alpha 0 \
        --use_revin 0 \
        --batch_size 8 \
        --learning_rate 0.00005 \
        --degree1 8 \
        --degree2 7 \
        --d_model 256 \
        --dropout 0 \
        --lradj type1 \
        > logs/ShortForecasting/HDKAN/s2/${model_id_name}_${model_name}_${p}.logs
done
