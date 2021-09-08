"""


    Read data for `../leaf_data` directory processed json files.
"""
import json
import pickle
from pathlib import Path
from collections import defaultdict
from torch.utils.data import ConcatDataset


def get_dataset_pickle(dataset_name: str, client_id: int, dataset_type: str, pickle_root: str):
    """load pickle dataset file for `dataset_name` `dataset_type` data based on client with client_id
        Args:
            dataset_name (str): string of dataset name to read pickle file
            client_id (int): client id
            dataset_type (str): Dataset type {train, test}
            pickle_root (str): path for leaf pickle datasets saving
        Raises:
            FileNotFoundError: No such file or directory {pickle_root}/{dataset_name}/{dataset_type}/{dataset_type}_{client_id}.pickle
        Returns:
            if there is no pickle file for `dataset`, throw FileNotFoundError, else return responding dataset
        """
    pickle_file = Path(pickle_root) / dataset_name / dataset_type / "{}_{}.pickle".format(dataset_type, client_id)
    dataset = pickle.load(open(pickle_file, 'rb'))
    return dataset


def get_all_dataset_pickle(dataset_name: str, dataset_type: str, pickle_root: str):
    """load all pickle dataset files for `dataset_name` with `dataset_type` data

    Args:
        dataset_name (str): string of dataset name to read pickle file
        dataset_type (str): Dataset type {train, test}
        pickle_root (str): string path for leaf pickle datasets saving
    Returns:
        ConcatDataset for dataset saved in each pickle file
    """
    pickle_files_path = Path(pickle_root) / dataset_name / dataset_type / dataset_type
    dataset_list = []
    for file in list(pickle_files_path.glob("**/*.pickle")):
        dataset_list.append(pickle.load(open(file, 'rb')))
    all_dataset = ConcatDataset(dataset_list)
    return all_dataset


def get_data_json(data_root: Path, dataset_name: str, dataset_type: str):
    """ Read .json file from ``data_dir``
    This is modified by [LEAF/models/utils/model_utils.py]
    https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py

    Args:
        data_root (Path): path for data saving root
        dataset_name (str): string of dataset name to read pickle file
        dataset_type (str): Dataset type {train, test}
    Returns:
        clients name dict mapping keys to id, groups list for each clients, a dict data mapping keys to client
    Examples:
        get_data_json(Path("../datasets/data"), "shakespeare", "train")
    """
    groups = []
    client_name2data = defaultdict(lambda: None)

    data_dir = data_root / dataset_name / "data" / dataset_type
    files = list(data_dir.glob("**/*.json"))
    for f in files:
        with open(f, 'r') as inf:
            cdata = json.load(inf)
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        client_name2data.update(cdata['user_data'])

    # generate clients_id_str - client_id_index map
    clients_name = list(sorted(client_name2data.keys()))
    clients_id = list(range(len(clients_name)))
    client_id2name = dict(zip(clients_id, clients_name))

    return client_id2name, groups, client_name2data
