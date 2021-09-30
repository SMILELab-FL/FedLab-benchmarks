# Modify from Flower's spilt_json_data.py for shakespeare data in leaf
# [https://github.com/adap/flower/blob/main/baselines/flwr_baselines/scripts/leaf/shakespeare/split_json_data.py]

"""Splits LEAF generated datasets and creates individual client partitions."""
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
from torchvision import transforms
from torch.utils.data.dataset import Dataset

# for pickle 
from dataset import FemnistDataset, ShakespeareDataset, CelebADataset, Sent140Dataset

def save_dataset_pickle(save_root: Path, dataset_name: str, user_idx: int, dataset_type: str, dataset: Dataset):
    """Saves partition for specific client
    Args:
        save_root (Path): Root folder where to save partition
        user_idx (int): User ID
        dataset_name (str): name of dataset
        dataset_type (str): Dataset type {train, test}
        dataset (Dataset): Dataset {train, test}
    """
    save_dir = save_root / dataset_name / dataset_type
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{dataset_type}_{str(user_idx)}.pkl", "wb") as save_file:
        pickle.dump(dataset, save_file)


def process_user(
        json_file: Dict[str, Any],
        user_idx: str,
        user_str: str,
        dataset_type: str,
        save_root: Path,
        dataset_name: str,
):
    """Creates and saves partition for user
    Args:
        json_file (Dict[str, Any]): JSON file containing user data
        user_idx (str): User ID (counter) in string format
        user_str (str): Original User ID
        dataset_type (str): Dataset type {train, test}
        save_root (Path): Root folder where to save the partition
        dataset_name (str): name of dataset
    """
    data = json_file["user_data"][user_str]["x"]
    label = json_file["user_data"][user_str]["y"]
    if dataset_name == "femnist":
        dataset = FemnistDataset(client_id=user_idx,
                                 client_str=user_str,
                                 data=data,
                                 targets=label)
    elif dataset_name == "shakespeare":
        dataset = ShakespeareDataset(client_id=user_idx,
                                     client_str=user_str,
                                     data=data,
                                     targets=label)
    elif dataset_name == "celeba":
        image_size = 64
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        dataset = CelebADataset(client_id=user_idx,
                                client_str=user_str,
                                data=data,
                                targets=label,
                                image_root="../datasets/celeba/data/raw/img_align_celeba",
                                transform=image_transform)
    elif dataset_name == "sent140":
        dataset = Sent140Dataset(client_id=user_idx,
                                 client_str=user_str,
                                 data=data,
                                 targets=label)

    else:
        raise ValueError("Invalid dataset:", dataset_name)
    save_dataset_pickle(save_root, dataset_name, user_idx, dataset_type, dataset)


def split_json_and_save(
        dataset_type: str,
        paths_to_json: List[Path],
        save_root: Path,
        dataset_name: str,
):
    """Splits LEAF generated datasets and creates individual client partitions.
    Args:
        dataset_type (str): Dataset type {train, test}
        paths_to_json (PathLike): Path to LEAF JSON files containing dataset.
        save_root (Path): Root directory where to save the individual client
            partition files.
        dataset_name (str): name of dataset
    """
    user_count = 0
    # check leaf data downloaded
    if len(paths_to_json) == 0:
        print("there is no leaf json file for {} {} data, please run leaf in `fedlab_benchmarks/datasets` firstly"
              .format(dataset_name, dataset_type))
        return

    print("processing {} {} data to dataset in pickle file".format(dataset_name, dataset_type))

    for path_to_json in paths_to_json:
        with open(path_to_json, "r") as json_file:
            json_file = json.load(json_file)
            users_list = sorted(json_file["users"])
            num_users = len(users_list)
            for user_idx, user_str in enumerate(users_list):
                process_user(
                    json_file, user_count + user_idx, user_str, dataset_type, save_root, dataset_name
                )
        user_count += num_users
    print("complete processing {} {} data to dataset in pickle file! "
          "all users number is {}".format(dataset_name, dataset_type, user_count))


# Example:
# python create_pickle_dataset.py --data_root "../datasets" --save_root "./pickle_dataset" --dataset_name "shakespeare"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""download and process a LEAF Shakespeare train/test dataset,
        save each client's train/test dataset in their respective folder in a form of pickle."""
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="""Root folder which contains many datasets downloading scripts and their data, including leaf dataset
                example in fedlab_benchmarks: '../../datasets' """,
    )
    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
        help="""Root folder where dataset will be save as
                {save_root}/{dataset_name}/{train,test}/{client_id}.pkl""",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="""processed dataset name""",
    )

    args = parser.parse_args()

    # save train dataset
    train_path = Path(args.data_root) / args.dataset_name / "data/train"
    original_train_datasets = sorted(
        list(train_path.glob("**/*.json"))
    )
    split_json_and_save(
        dataset_type="train",
        paths_to_json=original_train_datasets,
        save_root=Path(args.save_root),
        dataset_name=args.dataset_name,
    )

    # Split and save the test files
    test_path = Path(args.data_root) / args.dataset_name / "data/test"
    original_test_datasets = sorted(
        list(test_path.glob("**/*.json"))
    )
    split_json_and_save(
        dataset_type="test",
        paths_to_json=original_test_datasets,
        save_root=Path(args.save_root),
        dataset_name=args.dataset_name,
    )
