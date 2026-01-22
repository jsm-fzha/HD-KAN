@echo off
setlocal enabledelayedexpansion

REM === Create log folders ===
if not exist ".\logs" (
    mkdir ".\logs"
)
if not exist ".\logs\ShortForecasting" (
    mkdir ".\logs\ShortForecasting"
)
if not exist ".\logs\ShortForecasting\HDKAN" (
    mkdir ".\logs\ShortForecasting\HDKAN"
)
if not exist ".\logs\ShortForecasting\HDKAN\s2" (
    mkdir ".\logs\ShortForecasting\HDKAN\s2"
)

set model_name=HDKAN
set root_path_name=./dataset/short-term



REM =============================
REM ILI
REM =============================
set data_path_name=national_illness.csv
set model_id_name=illness
set data_name=custom
set seq_len=36

for %%p in (24 36 48 60) do (
    python -u run.py ^
        --is_training 1 ^
        --root_path %root_path_name% ^
        --data_path %data_path_name% ^
        --model_id %model_id_name%_%seq_len%_%%p ^
        --model %model_name% ^
        --data %data_name% ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 7 ^
        --train_epochs 20 ^
        --patience 5 ^
        --patch_len 9 ^
        --alpha 0.35 ^
        --batch_size 8 ^
        --learning_rate 0.0003 ^
        --degree1 7 ^
        --degree2 7 ^
        --d_model 128 ^
        --dropout 0 ^
        --lradj type3 ^
        > logs\ShortForecasting\HDKAN\s2\!model_id_name!_!model_name!_%%p.logs
)





REM =============================
REM COVID-19
REM =============================
set data_path_name=covid.csv
set model_id_name=covid
set data_name=custom2
set seq_len=36

for %%p in (24 36 48 60) do (
    python -u run.py ^
        --is_training 1 ^
        --root_path %root_path_name% ^
        --data_path %data_path_name% ^
        --model_id %model_id_name%_%seq_len%_%%p ^
        --model %model_name% ^
        --data %data_name% ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 55 ^
        --train_epochs 20 ^
        --patience 5 ^
        --patch_len 9 ^
        --alpha 0.5 ^
        --batch_size 1 ^
        --learning_rate 0.0016 ^
        --degree1 4 ^
        --degree2 7 ^
        --d_model 512 ^
        --dropout 0 ^
        --lradj type1 ^
        > logs\ShortForecasting\HDKAN\s2\!model_id_name!_!model_name!_%%p.logs
)





set data_path_name=METR-LA.csv
set model_id_name=METR-LA
set data_name=custom
set seq_len=36

for %%p in (24 36 48 60) do (
    python -u run.py ^
        --is_training 1 ^
        --root_path %root_path_name% ^
        --data_path %data_path_name% ^
        --model_id %model_id_name%_%seq_len%_%%p ^
        --model %model_name% ^
        --data %data_name% ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 207 ^
        --train_epochs 20 ^
        --patience 5 ^
        --patch_len 18 ^
        --alpha 0.35 ^
        --batch_size 16 ^
        --learning_rate 0.0006 ^
        --degree1 10 ^
        --degree2 5 ^
        --d_model 512 ^
        --dropout 0.1 ^
        --lradj type3 ^
        > logs\ShortForecasting\HDKAN\s2\!model_id_name!_!model_name!_%%p.logs
)





REM =============================
REM NASDAQ
REM =============================
set data_path_name=nasdaq.csv
set model_id_name=nasdaq
set data_name=custom
set seq_len=36

for %%p in (24 36 48 60) do (
    python -u run.py ^
        --is_training 1 ^
        --root_path %root_path_name% ^
        --data_path %data_path_name% ^
        --model_id %model_id_name%_%seq_len%_%%p ^
        --model %model_name% ^
        --data %data_name% ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 12 ^
        --train_epochs 20 ^
        --patience 5 ^
        --patch_len 36 ^
        --alpha 0.35 ^
        --batch_size 16 ^
        --learning_rate 0.000025 ^
        --degree1 7 ^
        --degree2 5 ^
        --d_model 128 ^
        --dropout 0 ^
        --lradj type3 ^
        > logs\ShortForecasting\HDKAN\s2\!model_id_name!_!model_name!_%%p.logs
)




REM =============================
REM Wiki
REM =============================
set data_path_name=wiki_mini.csv
set model_id_name=wiki
set data_name=custom
set seq_len=36

for %%p in (24 36 48 60) do (
    python -u run.py ^
        --is_training 1 ^
        --root_path %root_path_name% ^
        --data_path %data_path_name% ^
        --model_id %model_id_name%_%seq_len%_%%p ^
        --model %model_name% ^
        --data %data_name% ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --enc_in 99 ^
        --train_epochs 20 ^
        --patience 5 ^
        --patch_len 36 ^
        --alpha 0 ^
        --use_revin 0 ^
        --batch_size 8 ^
        --learning_rate 0.00005 ^
        --degree1 8 ^
        --degree2 7 ^
        --d_model 256 ^
        --dropout 0 ^
        --lradj type1 ^
        > logs\ShortForecasting\HDKAN\s2\!model_id_name!_!model_name!_%%p.logs
)




endlocal
