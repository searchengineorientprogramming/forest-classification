# Raw signal for machine learning

This folder contains filtered FFT spectrum for machine learning and machine learning code to train model, test and inspection

## Contents
- `prepare_FFT_data` - Includes code (`downlaod_data.ipynb`) to download the original seismic raw data, need to specify the net code, date range, station index and channel. Include the python code (`prepare_FFT_DP1.ipynb`) to read the raw signal, perform segmentation (split the daily raw signal into 12 10-minute windows) and do FFT on each of the 12 time window and save to .parquet format `raw_data_fft_DP1.parquet`. 

- `ML_raw` - The machine learning code for training, testing and inspection for 11, 22 and ZZ components. The code directly read the FFT dataframe from `data/raw/raw_data_fft_DP1.parquet`.

- `data` - The raw signal FFT results obtained from code in `prepare_FFT_data`.

- `hyper_param.json` - The hyper-parameter of SVM and RF model for DP1, DP2, and DPZ components.

- `results` - Save the machin learning model results and inspections figures.

## Usage
For quick check, directly run `ML_raw`, need to select the component of signal ('DP1', 'DP2' or 'DPZ') to start machine learning model training, testing and inspection.

User can also:
1. Download the seismic raw data, with specified net code, date range, station index and channel using `download_data.ipynb`
2. Run `prepare_FFT_DP1.ipynb` to obtain the FFT dataframe saved in .parquet format. Need to specify the raw signal path and component of data.
2. Run the code `ML_raw` to start machine learning model training, testing and inspection based on the output of `prepare_FFT_DP1.ipynb`.