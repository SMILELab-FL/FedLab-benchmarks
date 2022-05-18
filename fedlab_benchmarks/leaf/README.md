# The usage of the benchmark datasets LEAF - FedLab and PyTorch version

**FedLab migrates the TensorFlow version of LEAF dataset to the PyTorch framework, and provides the implementation of dataloader for the corresponding dataset. The unified interface is in `fedlab_benchmarks/leaf/dataloader.py`**

This markdown file introduces the process of using LEAF dataset in FedLab.

Read this in another language: [简体中文](./README_zh_cn.md).

## Description of LEAF datasets

The LEAF benchmark contains the federation settings of Celeba, femnist, Reddit, sent140, shakespeare and synthetic datasets. With reference to [leaf-readme.md](https://github.com/talwalkarlab/leaf) , the introduction the total number of users and the corresponding task categories of leaf datasets are given below.

1. FEMNIST

- **Overview:** Image Dataset
- **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
- **Task:** Image Classification

2. Sentiment140

- **Overview:** Text Dataset of Tweets
- **Details** 660120 users
- **Task:** Sentiment Analysis

3. Shakespeare

- **Overview:** Text Dataset of Shakespeare Dialogues
- **Details:** 1129 users (reduced to 660 with our choice of sequence length. See [bug](https://github.com/TalwalkarLab/leaf/issues/19).)
- **Task:** Next-Character Prediction

4. Celeba

- **Overview:** Image Dataset based on the [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Details:** 9343 users (we exclude celebrities with less than 5 images)
- **Task:** Image Classification (Smiling vs. Not smiling)

5. Synthetic Dataset

- **Overview:** We propose a process to generate synthetic, challenging federated datasets. The high-level goal is to create devices whose true models are device-dependant. To see a description of the whole generative process, please refer to the paper
- **Details:** The user can customize the number of devices, the number of classes and the number of dimensions, among others
- **Task:** Classification

6. Reddit

- **Overview:** We preprocess the Reddit data released by [pushshift.io](https://files.pushshift.io/reddit/) corresponding to December 2017.
- **Details:** 1,660,820 users with a total of 56,587,343 comments.
- **Task:** Next-word Prediction.


## Usage Flow
### Download and preprocess data

> For the six types of leaf datasets, refer to [leaf/data](https://github.com/talwalkarlab/leaf/tree/master/data) and provide data download and preprocessing scripts in `fedlab _ benchmarks/datasets/data`.
> In order to facilitate developers to use leaf, fedlab integrates the download and processing scripts of leaf six types of data sets into `fedlab_benchmarks/datasets/data`, which stores the download scripts of various data sets.

Common structure of leaf dataset folders:

```
/FedLab/fedlab_benchmarks/datasets/{leaf_dataset_name}

   ├── {other_useful_preprocess_util}
   ├── prerpocess.sh
   ├── stats.sh
   └── README.md
```
- `preprocess.sh`: downloads and preprocesses the dataset
- `stats.sh`: performs information statistics on all data (stored in `./data/all_data/all_data.json`) processed by `preprocess.sh`
- `README.md`: gives a detailed description of the process of downloading and preprocessing the dataset, including parameter descriptions and precautions.

> **Developers can directly run the executable script `create_datasets_and_save.sh` to obtain the dataset, process and store the corresponding dataset data in the form of a pickle file.**
> This script provides an example of using the preprocess.sh script, and developers can modify the parameters according to application requirements.

**preprocess.sh Script usage example:**

```shell
cd fedlab_benchmarks/datasets/data/femnist
bash preprocess.sh -s niid --sf 0.05 -k 0 -t sample --tf 0.8

cd fedlab_benchmarks/datasets/data/shakespeare
bash preprocess.sh -s niid --sf 0.2 -k 0 -t sample  
# bash preprocess.sh -s niid --sf 1.0 -k 0 -t sample  # get 660 users (with default --tf 0.9)
# bash preprocess.sh -s iid --iu 1.0 --sf 1.0 -k 0 -t sample   # get all 1129 users

cd fedlab_benchmarks/datasets/data/sent140
bash ./preprocess.sh -s niid --sf 0.01 -k 3 -t sample

cd fedlab_benchmarks/datasets/data/celeba
bash ./preprocess.sh -s niid --sf 0.05 -k 5 -t sample

cd fedlab_benchmarks/datasets/data/synthetic
bash ./preprocess.sh -s niid --sf 1.0 -k 5 -t sample --tf 0.6

# for reddit, see its README.md to download preprocessed dataset manually
```

By setting parameters for `preprocess.sh`, the original data can be sampled and spilted. **The `readme.md` in each dataset folder provides the example and explanation of script parameters, the common parameters are: **

1. `-s` := 'iid' to sample in an i.i.d. manner, or 'niid' to sample in a non-i.i.d. manner; more information on i.i.d. versus non-i.i.d. is included in the 'Notes' section
2. `--sf` := fraction of data to sample, written as a decimal; default is 0.1
3. `-k ` := minimum number of samples per user
4. `-t` := 'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups
5. `--tf` := fraction of data in training set, written as a decimal; default is 0.9, representing train set: test set = 9:1.
6. `--iu` := number of users, only if i.i.d. sampling; expressed as a fraction of the total number of users; default is 0.01, representing the final total number of users in this setting : total number of users in the whole dataset = 0.01.
   
   It can be used to control the number of preprocessed users in i.i.d. sampling. 
   For example, the total number of users is 3500 in FEMNIST. We can use `--iu=0.01` to set the total number of users 35 in i.i.d. sampling . And sampled data will be allocated to these 35 users.

At present, FedLab's Leaf experiment need provided training data and test data, so **we needs to provide related data training set-test set splitting parameter for `preprocess.sh`** to carry out the experiment, default is 0.9.

> If you need to re-divide data with other settings, delete the 'data/sampled_data', 'data/rem_user_data', 'data/train', 'data/test' folders in each dataset's folder. Then run the `preprocess.sh` script (located in the corresponding dataset) for downloading and processing. **
> 
> If you need to re-download the original data and read the data, you need to delete other folders such as `data/raw_data`, `data\all_data` and `data/intermediate`, and then run the corresponding `preprocess.sh` script file to re-download and process again.

### pickle file stores DataSet.
In order to speed up developers' reading data, fedlab provides a method of processing raw data into DataSet and storing it as a pickle file. The DataSet of the corresponding data of each client can be obtained by reading the pickle file after data processing.

**Set the parameters and instantiate the PickleDataset object (located in pickle_dataset.py), the usage example is as follows:**

```python
from .pickle_dataset import PickleDataset
pdataset = PickleDataset(pickle_root="pickle_datasets", dataset_name="shakespeare")
# create responding dataset in pickle file form
pdataset.create_pickle_dataset(data_root="../datasets")
# read saved pickle dataset and get responding dataset
train_dataset = pdataset.get_dataset_pickle(dataset_type="train", client_id="0")
test_dataset = pdataset.get_dataset_pickle(dataset_type="test", client_id="2")
```

Parameter Description:

1. `data_root`: The root path for storing leaf data sets, which contains leaf data sets; if you use the `fedlab_benchmarks/datasets/` provided by fedlab to download leaf data, then `data_root` can be set to this path. The relative address of the path is out.
2. `pickle_root`: Store the pickle file address of the processed DataSet, each data set _DataSet_ will be saved as `{pickle_root}/{dataset_name}/{train,test}`; the example is to create a `pickle_datasets` folder under the current path Store all pickle dataset files.
3. `dataset_name`: Specify the name of the leaf data set to be processed. There are six options {feminist, Shakespeare, celeba, sent140, synthetic, reddit}.

> Besides, you can directly run the `gen_pickle_dataset.sh` script (located in `fedlab_benchmarks/leaf`) to instantiate the corresponding PickleDataset object for the dataset and store it as a pickle file.
```shell
bash gen_pickle_dataset.sh "femnist" "../datasets" "./pickle_datasets"
```
And parameters 1, 2, and 3 correspond to dataset_name, data_root, and pickle_root respectively.

---
**Supplementary instruction of the vocabulary used by the nlp dataset:**

NLP tasks need to perform operations such as word tokenization and word vocabulary mapping on the input text data, and then pass in the model for training and prediction. Some operations need to use word lists.
When a Dataset object corresponding to an NLP Dataset is generated, the Vocab Object corresponding to the dataset is obtained at the specified location and applied. If the Vocab does not exist, it will be created in the default address.

Currently, for NLP tasks that require vocabulary to be constructed, FedLab wraps the dataset vocabulary generation method in the `gen_pickle_dataset. sh` script (located in `fedlab_benchmarks /leaf`).
By specifying the parameter `build_vocab` representing whether to build vocab and other vocab-related parameters, the vocab construction can be completed before the Dataset object is generated for subsequent processing.

The following is an example of script usage:
```shell
cd fedlab_benchmarks/leaf/nlp_utils
bash gen_pickle_dataset.sh "sent140" "../datasets" "./pickle_datasets" 1 './nlp_utils/dataset_vocab' './nlp_utils/glove' 50000
```

Parameter Description:
1. Parameter 1: `dataset`, Specify the name of the leaf data set to be processed. There are six options {sent140, reddit}.
2. Parameter 2: `data_root`, representing the root path for storing leaf data sets, which contains leaf data sets. If you use the `fedlab_benchmarks/datasets/` provided by fedlab to download leaf data, then `data_root` can be set to this path. The example shows the relative address of this path.
3. Parameter 2: `pickle_root`, storing the pickle file storge address of the processed DataSet, each data set _DataSet_ will be saved as `{pickle_root}/{dataset_name}/{train,test}`. The example is to create a `pickle_datasets` folder under the current path to store all pickle dataset files.
4. Parameter 4: `build_vocab`: representing whether the vocabulary needs to be constructed. The value is 0/1
5. Parameter 5: `vocab_save_root`, representing the location where the pretrained table is stored, default to `fedlab_benchmarks/leaf/nlp_utils/glove`. The example shows the relative address of this path.
6. Parameter 6: `vector_save_root`, storing the location of the generated dataset vocabulary, defaulting to `fedlab_benchmarks/leaf/nlp_utils/dataset_vocab`. The example shows the relative address of this path.
7. Parameter 7: `vocab_limit_size`, represents the maximum amount of the vocabulary, which limits the size of the vocabulary. Default to 50000


### dataloader loading data set

Leaf datasets are loaded by `dataloader.py` (located under `fedlab_benchmarks/leaf/dataloader.py`). All returned data types are pytorch [Dataloader](https://pytorch.org/docs/stable/data.html).

By calling this interface and specifying the name of the data set, the corresponding Dataloader can be obtained.

**Example of use:**

```python
from leaf.dataloader import get_LEAF_dataloader
def get_femnist_shakespeare_dataset(args):
    if args.dataset == 'femnist' or args.dataset == 'shakespeare':
        trainloader, testloader = get_LEAF_dataloader(dataset=args.dataset,
                                                      client_id=args.rank)
    else:
        raise ValueError("Invalid dataset:", args.dataset)

    return trainloader, testloader
```


After that, when the dataset interface of leaf is called (dataloader.py), the corresponding vocab will be obtained and passed to the PickleDataset instantiation object of the corresponding dataset for vocabulary processing.
The relevant example is given below (located in fedlab_benchmarks/leaf/dataloader.py) :

```python
# Need to run leaf/gen_pickle_dataset.sh to generate pickle files for this dataset firstly
pdataset = PickleDataset(dataset_name=dataset, data_root=data_root, pickle_root=pickle_root)
try:
    trainset = pdataset.get_dataset_pickle(dataset_type="train", client_id=client_id)
    testset = pdataset.get_dataset_pickle(dataset_type="test", client_id=client_id)
except FileNotFoundError:
    print(f"No dataset pickle files for {dataset} in {pdataset.pickle_root.resolve()}")
```

### Run experiment

The current experiment of LEAF data set is the **single-machine multi-process** scenario under FedAvg's Cross machine implement, and the tests of femnist, shakespeare, sent140, celeba datasets have been completed.

Run `fedlab_benchmarks/fedavg/cross_machine/LEAF_test.sh' to quickly execute the simulation experiment of fedavg under leaf data set.