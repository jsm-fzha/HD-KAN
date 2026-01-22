#!/bin/bash

# =============================
# Create log folders
# =============================
mkdir -p ./logs/LongForecasting/HDKAN

model_name=HDKAN

# =============================
# ETTh1
# =============================
root_path_name=./dataset/ETT-small
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
seq_len=96

for p in 96 192 336 720; do
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
        --train_epochs 10 \
        --patience 3 \
        --patch_len 12 \
        --batch_size 64 \
        --learning_rate 0.0008 \
        --degree1 7 \
        --degree2 2 \
        --d_model 256 \
        --dropout 0.2 \
        > logs/LongForecasting/HDKAN/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# ETTh2
# =============================
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

for p in 96 192 336 720; do
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
        --train_epochs 10 \
        --patience 3 \
        --patch_len 6 \
        --batch_size 64 \
        --learning_rate 0.0003 \
        --degree1 8 \
        --degree2 5 \
        --d_model 512 \
        --dropout 0.2 \
        > logs/LongForecasting/HDKAN/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# ETTm1
# =============================
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

for p in 96 192 336 720; do
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
        --train_epochs 10 \
        --patience 3 \
        --patch_len 24 \
        --batch_size 64 \
        --alpha 0.2 \
        --learning_rate 0.0001 \
        --degree1 5 \
        --degree2 5 \
        --d_model 512 \
        --dropout 0.1 \
        > logs/LongForecasting/HDKAN/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# ETTm2
# =============================
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

for p in 96 192 336 720; do
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
        --train_epochs 10 \
        --patience 3 \
        --patch_len 48 \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --degree1 6 \
        --degree2 5 \
        --d_model 256 \
        --dropout 0.3 \
        > logs/LongForecasting/HDKAN/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# Exchange
# =============================
root_path_name=./dataset/exchange_rate
data_path_name=exchange_rate.csv
model_id_name=exchange
data_name=custom

for p in 96 192 336 720; do
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
        --enc_in 8 \
        --train_epochs 10 \
        --patience 3 \
        --patch_len 48 \
        --alpha 0.2 \
        --batch_size 32 \
        --learning_rate 0.00005 \
        --degree1 4 \
        --degree2 2 \
        --d_model 256 \
        --dropout 0.1 \
        > logs/LongForecasting/HDKAN/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# Weather
# =============================
root_path_name=./dataset/weather
data_path_name=weather.csv
model_id_name=weather

for p in 96 192 336 720; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${p} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $p \
        --enc_in 21 \
        --train_epochs 10 \
        --patience 3 \
        --patch_len 12 \
        --alpha 0.1 \
        --batch_size 64 \
        --learning_rate 0.0004 \
        --degree1 6 \
        --degree2 3 \
        --d_model 512 \
        --dropout 0.1 \
        > logs/LongForecasting/HDKAN/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# Solar
# =============================
root_path_name=./dataset/Solar
data_path_name=solar_AL.txt
model_id_name=solar

for p in 96 192 336 720; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${p} \
        --model $model_name \
        --data Solar \
        --features M \
        --seq_len $seq_len \
        --pred_len $p \
        --enc_in 137 \
        --train_epochs 10 \
        --patience 3 \
        --patch_len 12 \
        --alpha 0.05 \
        --use_revin 0 \
        --batch_size 32 \
        --learning_rate 0.0006 \
        --degree1 5 \
        --degree2 2 \
        --d_model 256 \
        --dropout 0 \
        > logs/LongForecasting/HDKAN/${model_id_name}_${model_name}_${p}.logs
done

# =============================
# Electricity
# =============================
root_path_name=./dataset/electricity
data_path_name=electricity.csv
model_id_name=electricity

for p in 96 192 336 720; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${p} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $p \
        --enc_in 321 \
        --train_epochs 10 \
        --patience 3 \
        --patch_len 24 \
        --alpha 0.2 \
        --batch_size 16 \
        --learning_rate 0.0006 \
        --degree1 10 \
        --degree2 9 \
        --d_model 512 \
        --dropout 0.1 \
        > logs/LongForecasting/HDKAN/${model_id_name}_${model_name}_${p}.logs
done
