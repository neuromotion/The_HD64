### Creating a runtime environment
You can use the anaconda package manager to install the required dependencies

```
conda env create -f environment.yml
conda activate eesinference
```

### Data downloading
See README in "data" directory


### Forward model training/evaluating/inference

Driver function: **main.py**

*Usage:* 
```
python main.py --exp_name [electrode used ratio] --exp_type StimToEMG --dataset [preprocessed dataset] --mode [mode to run] --raw_data [raw data to load] --target_idx [index of target response]
```

*Arguments:*

  --exp_name [= S2E_100, S2E_50, S2E_25]
  
  --exp_type [= StimToEMG]
  
  --mode [= train, eval, inference]
  
  --dataset: This defines the data prefix for any particular run [= 100, 50, 25]
  
  --exp_name: This defines the model/ckpt prefixes for any particular run [= S2E_100, S2E_50, S2E_25]

  --raw_data: This defines the directory of data [= data/density100.parquet, data/density50.parquet, data/density25.parquet]

  (inference mode only)
  --target_idx Defaults to 0. Used to specify the "target" for inference
