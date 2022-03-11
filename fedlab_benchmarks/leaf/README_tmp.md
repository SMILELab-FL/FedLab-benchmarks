# PROCESS_DATA README

This folders contains processed dataset pickle files for leaf datasets.

You can run `create_pickle_dataset.py` script for processed leaf data

Notice:
1. please make sure leaf dataset is downloaded and processed by leaf. (leaf code in `fedlab_benchmarks/datasets/data`)
2. please make sure `fedlab_benchmarks/datasets/data/{dataset_name}/{train,test}` path existing for train data and test data.
3. example script: 
   `python create_pickle_dataset.py --data_root "../../datasets/data" --save_root "pickle_dataset" --dataset_name "shakespeare"`
3. usage example:
   ```
   pdataset = PickleDataset(pickle_root="pickle_datasets", dataset_name="shakespeare")
   pdataset.create_pickle_dataset(data_root="../datasets")
   dataset = pdataset.get_dataset_pickle(dataset_type="test", client_id="2")
   ```
