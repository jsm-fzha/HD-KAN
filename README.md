# HD-KAN


## Updates
The detailed training logs:
- Long-term forecasting: [logs/LongForecasting/AdaCycle](logs/LongForecasting/HDKAN)
- Short-term forecasting: [logs/ShortForecasting/AdaCycle](logs/ShortForecasting/HDKAN/s1)

## Usage
1. Install the dependencies
```bash
    pip install -r requirements.txt
```
2. Obtain the dataset from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) and extract it to the root directory of the project. Make sure the extracted folder is named `dataset` and has the following structure:
```
    dataset
    ├── electricity
    │   └── electricity.csv
    ├── ETT-small
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    │   └── ETTm2.csv
    ├── exchange_rate
    │   └── exchange_rate.csv
    ├── Solar
    │   └── solar_AL.txt
    ├── weather
    │   └── weather.csv
    └── short-term
        ├── covid.csv
        ├── METR-LA.csv
        ├── nasdaq.csv
        ├── national_illness.csv
        └── wiki_mini.csv
```
3. Train and evaluate the model. All the training scripts are located in the `scripts` directory.
   
   For Linux:
    ```bash
    sh scripts/LongForecasting/AdaCycle.sh
    sh scripts/ShortForecasting/AdaCycle.sh
    ```
    For Windows:
    ```
    scripts/LongForecasting/AdaCycle.bat
    scripts/ShortForecasting/AdaCycle.bat
    ```

