# Cross-correlation signal machine learning

This folder contains cross-correlation FFT data, machine learning code to train model and inspection, and the code to calculate the average geometric phase change.

## Contents

- `prepare_FFT_data` - Include the `cc.sh` to perform cross-correlation of raw seismic data, the input file path need to set to the folder contains raw seismic data (`Raw_ML/prepare_FFT_data`). `custom.py` is a python helper script loaded by `cc.sh`. `prepare_FFT.ipynb` is the python code to read cross-correlated signal, perform FFT and saved as .parquet format (`cc_data_fft_DPX.parquet`, DPX can be DP1, DP2 or DPZ). 

- `ML_CC.py` - The machine learning code for training, testing and inspection for 11, 22 or ZZ components. The code directly read the FFT dataframe from `data/cc/cc_data_fft_DP1.parquet`. User need to specify which component to calculate.
- `hyper_param.json` - The hyper-parameter of SVM and RF model for DP1, DP2, and DPZ components.

- `ML_CC_intra.py` - The machine learning code for training, testing and inspection for intraclass testing for ZZ components.
- `hyper_param_intra.json` - The hyper-parameter of SVM and RF model for intraclass testing.

- `results` - Save the machin learning model results and inspections figures.

- `eta` - Seismic data and code to calculate the avergae geometric phaes change based on the cross-correlation signal of DPZ component.


## Usage
For quick calculation, directly run `ML_CC.py` and `ML_CC_intra.py` to start machine learning model training, testing and inspection. Need to select which component to calculate in code.

User can also:
1. Install MSNoise (v1.6.3: http://msnoise.org/doc/).
2. Download the required raw signal.
3. Run cross-correlation with:
   ```bash
      bash cc.sh
4. run `ML_CC.py` and `ML_CC_intra.py` to start machine learning model training, testing and inspection.