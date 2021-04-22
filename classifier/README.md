
# Classifier
Modified version of DD-Net Skeleton-based action recognition model.






# Train the classifier
1. Prepare your dataset.  
A video of a person performing sing language. You can get the dataset from this [`Gdrive`]()

```
my_dataset_folder
├───Hksl_bear
│   ├───vid_01.mp4
│   ├───vid_02.mp4
│   ...
├───Hksl_bicycle
│   ├───vid_01.mp4
│   ├───vid_02.mp4
│   ...
├───Hksl_carrot
│   ...
├───Hksl_chef
│   ...
.
.
.
```
2. Preprocess data for training.

```
python ../create_dataset.py my_dataset_folder output_folder
```
3. Edit LABELS in [`constants.py`](../constants.py#L55)


3. Train  
See [`notebook`](train_dd.ipynb)  
  
  
4. Convert checkpoint to TFJS model.  
Use the TFJSConverter command and locate your trained .h5 checkpoint.  
```
pip install tensorflowjs
tfjs_wizard
```
