## 【FedLab】联邦数据集benchmark leaf的使用

FedLab将TensorFlow版本的LEAF数据集迁移到了PyTorch框架下，并提供了相应数据集的dataloader的实现脚本，统一的接口在`fedlab_benchmarks/dataset/leaf_data_process/dataloader.py`

本文介绍在FedLab中leaf数据集的使用流程。

### leaf数据集说明

LEAF是一个模块化的基准测试框架，用于联邦设置的学习，详情可参见：

- **Homepage:** [leaf.cmu.edu](https://leaf.cmu.edu/)
- **Paper:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)
- **Code:** [TalwalkarLab](https://github.com/TalwalkarLab)/**[leaf](https://github.com/TalwalkarLab/leaf)**

LEAF benchmark 包含了celeba, femnist, reddit, sent140, shakespeare, synthetic 六类数据集的联邦设置。参考[leaf - README.md](https://github.com/TalwalkarLab/leaf) ，以下给出六类数据集的简介、总用户数和对应任务类别。

1. FEMNIST

- **概述：** 图像数据集
- **详情：** 共有62个不同类别（10个数字，26个小写字母，26个大写字母）； 每张图像是28 * 28像素（可选择全部处理为128 * 128像素）； 共有3500位用户。
- **任务：** 图像分类

2. Sentiment140

- **概述：** 推特推文文本数据集
- **详情：** 共660120位用户
- **任务：** 情感分析

3. Shakespeare

- **概述：** 莎士比亚对话文本数据集
- **详情：** 共1129位用户（后续根据序列长度减少到660位，详情查看[bug](https://github.com/TalwalkarLab/leaf/issues/19) ）
- **任务：** 下一字符预测

4. Celeba

- **概述：** 基于 [大规模名人面孔属性数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 的图像数据集
- **详情：** 共9343位用户（排除了样本数小于等于5的名人）
- **任务：** 图像识别（微笑检测）

5. Synthetic Dataset

- **概述：** 提出了一个生成具有挑战性的合成联合数据集的过程，高级目标是创建真实模型依赖于各设备的设备。可参阅论文["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097) 查看整个生成过程的描述。
- **详情：** 用户可以自定义设备数量、类别数量和维度数量等
- **任务：** 分类

6. Reddit

- **概述：** 对[pushshift.io](https://files.pushshift.io/reddit/) 发布的2017年12月的Reddit数据进行了预处理
- **详情：** 共1,660,820位用户，总评论56,587,343条。
- **任务：** 下一单词预测

### 使用leaf下载数据集

为方便用户使用leaf，fedlab将leaf六类数据集的下载、处理脚本整合到`fedlab_benchmarks/datasets/data`中，该文件夹存储各类数据集的下载脚本。

leaf数据集文件夹常用结构：

```
/FedLab-benchmarks/fedlab_benchmarks/datasets/{leaf_dataset_name}

   ├── {other_useful_preprocess_util}
   ├── prerpocess.sh
   ├── stats.sh
   └── README.md
```

- `preprocess.sh`：对数据集进行下载和处理
- `create_datasets_and_save.sh`：封装了`preprocess.sh`的使用，并将各用户数据处理为对应的Dataset，以pickle文件的形式存储
- `stats.sh`：对`preprocess.sh`处理后所有数据（存储于`./data/all_data/all_data.json`）进行信息统计
- `README.md`：对该数据集的下载和处理过程进行了详细说明，包含了参数说明和注意事项。

**通过运行preprocess.sh可获取相应数据集，preprocess.sh脚本使用样例如下：**

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

通过对`preprocess.sh`设定参数，实现对原始数据的采样、划分等处理，**各数据集文件夹下的README.md均提供了脚本参数示例和解释，常见参数有：**

1. `-s`表示采样方式，取值有'iid'和'niid'两种选择，表示是否使用i.i.d方式进行采样；
2. `--sf`表示采样数据比例，取值为小数，默认为0.1；
3. `-k` 表示采样时所要求的用户最少样本数目，筛选掉拥有过少样本的用户，若取值为0表示不进行样本数目的筛选。
4. `-t`表示划分训练集测试集的方式，取值为'user'则划分用户到训练-测试集合，取值为'sample'则划分每个用户的数据到训练-测试集合中；
5. `--tf` 表示训练集的数据占比，取值为小数，默认为0.9，表示训练集:测试集=9:1。

目前FedLab的Leaf实验需要提供训练数据和测试数据，因此**需要对`preprocess.sh`提供相关的数据训练集-测试集划分参数，默认划分比例为0.9**

**若需要重新获取数据或划分数据，需要先删除各数据集下的`data`文件夹再运行相关脚本进行数据下载和处理。**

### pickle文件存储DataSet

为加速用户读取数据，fedlab提供了将原始数据处理为DataSet并存储为pickle文件的方法。通过读取数据处理后的pickle文件可获得各客户端对应数据的DataSet。

**设定参数并实例化PickleDataset对象（位于pickle_dataset.py），使用样例如下：**

```python
from leaf.pickle_dataset import PickleDataset
pdataset = PickleDataset(pickle_root="pickle_datasets", dataset_name="shakespeare")
# create responding dataset in pickle file form
pdataset.create_pickle_dataset(data_root="../datasets")
# read saved pickle dataset file and get responding dataset
train_dataset = pdataset.get_dataset_pickle(dataset_type="train", client_id="0")
test_dataset = pdataset.get_dataset_pickle(dataset_type="test", client_id="2")
```

参数说明：

1. `data_root`：存储leaf数据集的root路径，该路径包含leaf各数据集；若使用fedlab所提供的`fedlab_benchmarks/datasets/`下载leaf数据，则`data_root`可设置为该路径，示例给出了该路径的相对地址。
2. `pickle_root`：存储处理后DataSet的pickle文件地址，各数据集DataSet将另存为`{pickle_root}/{dataset_name}/{train,test}`；示例则在当前路径下创建`pickle_datasets`文件夹存储所有的pickle dataset文件。
3. `dataset_name`：指定要处理的leaf数据集名称，有{feminist, Shakespeare, celeba, sent140, synthetic, reddit}六种选择。

> 此外，可直接运行`gen_pickle_dataset.sh`脚本（位于`fedlab_benchmarks/leaf`）实现数据集实例化相应的PickleDataset对象并存储为pickle文件形式。
```shell
bash gen_pickle_dataset.sh "shakespeare" "../datasets" "./pickle_datasets"
```
其中参数1、2、3分别对应于上述的dataset_name、data_root、pickle_root。


### dataloader加载数据集

leaf数据集由`dataloader.py`加载(位于`fedlab_benchmarks/leaf/dataloader.py`)，所有返回数据类型均为pytorch [Dataloader](https://pytorch.org/docs/stable/data.html) 。

通过调用该接口并指明数据集名称，即可获得相应的Dataloader。

**使用示例：**

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
**nlp数据集使用的词表补充说明：**

对nlp任务而言，当前大多数方法用户词表的构建采用的是集中获取所有用户训练数据进行生成，破坏了联邦学习的原始数据不可得的原则和隐私性。

目前FedLab中对于需要构建词表的nlp任务采用的是抽样部分客户端，用这些客户端数据进行词表构建，是一种简单但能一定程度维持联邦用户数据不可得特性的方法。目前本团队已在对联邦nlp中该问题进行研究。

**对于需要构建词表的nlp任务，在使用所存储的PickleDataset对象之前需要先运行`build_vocab.sh`生成vocab便于后续处理（位于`fedlab_benchmarks/leaf/nlp_utils`），脚本使用样例如下：**
```shell
cd fedlab_benchmarks/leaf/nlp_utils
bash build_vocab.sh "../../datasets" "shakespeare" 0.25 30000 "./dataset_vocab"
```

参数说明：
1. 参数1: data_root, 表示用户原始数据存储的根目录，本框架中默认采用的是'fedlab_benchmarks/datasets'存储各类数据集的原始数据
2. 参数2：dataset，表示nlp任务对应的数据集名称
3. 参数3：data_select_ratio，表示抽样客户端参与词表构建的比例
4. 参数4：vocab_limit_size，表示词表的最大数量额度，限制词表大小
5. 参数5：save_root，表示所构建词表存储的目录位置，如需使用所构建的词表，应调用`get_built_vocab(save_root,dataset)`（位于fedlab_benchmarks/leaf/nlp_utils/sample_build_vocab.py）提供该路径获取

之后**将在leaf的数据集接口（dataloader.py）被调用时获取相应的vocab传递给对应数据集的PickleDataset实例化对象进行词表处理**，以下给出相关示例（位于fedlab_benchmarks/leaf/dataloader.py）：
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

### 运行实验

当前LEAF数据集所进行的实验为FedAvg的cross machine下的**单机多进程**场景，目前已完成femnist和shakespeare两类数据集的测试。

通过运行`fedlab_benchmarks/fedavg/cross_machine/LEAF_test.sh`可快速执行LEAF数据集下FedAvg的模拟实验。

