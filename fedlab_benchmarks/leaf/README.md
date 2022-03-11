# The usage of the benchmark datasets LEAF - FedLab and PyTorch version

**FedLab migrates the TensorFlow version of LEAF dataset to the PyTorch framework, and provides the implementation of dataloader for the corresponding dataset. The unified interface is in `fedlab_benchmarks/leaf/dataloader.py`**

This markdown file introduces the process of using LEAF dataset in FedLab.

Read this in another language: [简体中文](./README_zh_cn.md).

### description of Leaf datasets

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
bash preprocess.sh -s niid --sf 0.05 -k 0 -t sample

cd fedlab_benchmarks/datasets/data/shakespeare
bash preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8

cd fedlab_benchmarks/datasets/data/sent140
bash ./preprocess.sh -s niid --sf 0.05 -k 3 -t sample

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

At present, FedLab's Leaf experiment need provided training data and test data, so **we needs to provide related data training set-test set splitting parameter for `preprocess.sh`** to carry out the experiment, default is 0.9.

**If you need to obtain or split data again, make sure to delete `data` folder in the dataset directory before re-running `preprocess.sh` to download and preprocess data.**

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
bash gen_pickle_dataset.sh "shakespeare" "../datasets" "./pickle_datasets"
```
And parameters 1, 2, and 3 correspond to dataset_name, data_root, and pickle_root respectively.


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


---
**Supplementary instruction of the vocabulary used by the nlp dataset:**

For NLP task, most of the current methods for the construction of user vocabulary are to obtain all users' training data centrally for generation, which destroys the principle and privacy of non-availability of original data and privacy of federated learning.

Currently, FedLab uses a way to saple some clients and use their data to build vocabulary for the NLP tasks that needs a user-built vocabulary, which is a simple and not private strictly method, but it can maintain the unavailability of federated user data to a certain extent than centrally use all clients' data.
At present, our team has been researching this problem in the federated NLP.

**For nlp tasks that need to build a vocabulary, you need to run `build_vocab.sh` to generate vocab (located in `fedlab_benchmarks/leaf/nlp_utils`) before instantiating the `PickleDataset` object. The script usage example is as follows:**
```shell
cd fedlab_benchmarks/leaf/nlp_utils
bash build_vocab.sh "../../datasets" "shakespeare" 0.25 30000 "./dataset_vocab"
```
Parameter Description:
1. Parameter 1: data_root, represents the root directory of the user's original data storage, the default in FedLab framework is 'fedlab_benchmarks/datasets', which stores the original data of various datasets
2. Parameter 2: dataset, represents the name of the dataset corresponding to the nlp task.
3. Parameter 3: data_select_ratio, represents the proportion of sampling clients participating in vocabulary building
4. Parameter 4: vocab_limit_size, represents the maximum amount of the vocabulary, which limits the size of the vocabulary
5. Parameter 5: save_root, represents the directory location where the built vocabulary is stored. If you need to use the built vocabulary, you should call `get_built_vocab(save_root,dataset)` (located in fedlab_benchmarks/leaf/nlp_utils/sample_build_vocab.py) to provide the path to obtain

After that, when the dataset interface of leaf is called (dataloader.py), the corresponding vocab will be obtained and passed to the PickleDataset instantiation object of the corresponding dataset for vocabulary processing.
The relevant example is given below (located in fedlab_benchmarks/leaf/dataloader.py) :

```python
pdataset = PickleDataset(pickle_root="./pickle_datasets", dataset_name=dataset)
trainset = pdataset.get_dataset_pickle(dataset_type="train", client_id=client_id)
testset = pdataset.get_dataset_pickle(dataset_type="test", client_id=client_id)

# get vocab and index data
if dataset == 'sent140':
    vocab = get_built_vocab(dataset)
    trainset.token2seq(vocab, maxlen=300)
    testset.token2seq(vocab, maxlen=300)
```


### Run experiment

The current experiment of LEAF data set is the **single-machine multi-process** scenario under FedAvg's Cross machine implement, and the tests of femnist and Shakespeare data sets have been completed.

Run `fedlab_benchmarks/fedavg/cross_machine/LEAF_test.sh' to quickly execute the simulation experiment of fedavg under leaf data set.