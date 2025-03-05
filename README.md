## Ogirinal MolMCL GitHub:
https://github.com/yuewan2/MolMCL.git

## Follow the below steps for dependency installation.
```
conda create -n molmcl python=3.10
conda activate molmcl
bash build.sh  # this will install all dependencies using pip
```
## Fine-tuning
Examples of creating the configuration files can be found:

### generate_YAML_configuration_file_example.ipynb
```
python ./scripts/finetune.py <data_folder>/<data_name>  # e.g., moleculenet/bace (save under ./config/)
```
## Predicting
Examples of creating the configuration files for prediction can be found:

### generate_prediction_YAML_configuration_file_example.ipynb
