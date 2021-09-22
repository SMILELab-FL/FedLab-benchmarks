"""
    Read leaf json data from `fedlab_benchmarks/datasets/data/{leaf_dataset}`
    Read leaf pickle datasets from `fedlab_benchmarks/leaf/process_data`
"""
import json
import pickle
from pathlib import Path
from collections import defaultdict
from torch.utils.data import ConcatDataset


def get_dataset_pickle(dataset_name: str, client_id: int, dataset_type: str, pickle_root: Path):
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
    pickle_root = Path(__file__).parent.resolve() / pickle_root
    pickle_file = pickle_root / dataset_name / dataset_type / f"{dataset_type}_{client_id}.pkl"
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
    pickle_root = Path(__file__).parent.resolve() / pickle_root
    pickle_files_path = pickle_root / dataset_name / dataset_type
    dataset_list = []
    for file in list(pickle_files_path.glob("**/*.pkl")):
        dataset_list.append(pickle.load(open(file, 'rb')))
    all_dataset = ConcatDataset(dataset_list)
    return all_dataset


def get_data_json(data_root: str, dataset_name: str, dataset_type: str):
    """ Read .json file from ``data_dir``
    This is modified by [LEAF/models/utils/model_utils.py]
    https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py

    Args:
        data_root (str): path for data saving root
        dataset_name (str): string of dataset name to read pickle file
        dataset_type (str): Dataset type {train, test}
    Returns:
        clients name dict mapping keys to id, groups list for each clients, a dict data mapping keys to client
    """
    groups = []
    client_name2data = defaultdict(lambda: None)

    data_dir = Path(data_root) / dataset_name / "data" / dataset_type
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


if __name__ == '__main__':
    client_id2name, groups, client_name2data = get_data_json("../datasets", "celeba", "train")
    from PIL import Image
    import torch

    IMAGE_SIZE = 84
    IMAGE_DIR = Path(__file__).parent.resolve() / "../datasets/celeba/data/raw/img_align_celeba"

    for (key, value) in client_name2data.items():
        input = value['x']
        output = value['y']
        from torchvision import transforms
        image_transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])

        data = []
        targets = []
        for index in range(len(input)):
            image_name = input[index]
            label = output[index]

            image = Image.open(IMAGE_DIR / image_name).resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
            image_tensor = image_transform(image)
            data.append(image_tensor)
            targets.append(torch.tensor(label, dtype=torch.long))
        print("{} is completed".format(key))