import argparse
import glob
import os

import pandas as pd


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="train a network for indoor dataset")
    parser.add_argument(
        "--dataset_dir", type=str, default="./dataset", help="path to a dataset dirctory"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./csv", help="a directory where csv files will be saved"
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    # img や label を保存するリスト
    train_img_paths = []
    val_img_paths = []
    test_img_paths = []

    train_cls_ids = []
    val_cls_ids = []
    test_cls_ids = []

    train_cls_labels = []
    val_cls_labels = []
    test_cls_labels = []

    class_dirs = glob.glob(os.path.join(args.dataset_dir, "indoorCVPR_09/Images", "*"))
    classes = [os.path.basename(d) for d in class_dirs if os.path.isdir(d)]
    classes.sort()
    class2id = {classes[i]: i for i in range(len(classes))}

    trainval_file_names = sorted(
        set(open(os.path.join(args.dataset_dir, "TrainImages.txt")).read().splitlines())
    )
    test_file_names = set(
        open(os.path.join(args.dataset_dir, "TestImages.txt")).read().splitlines()
    )

    # 各ディレクトリから画像のパスを指定
    # Train: Val = 4: 1
    for i, name in enumerate(trainval_file_names):
        path = os.path.join(args.dataset_dir, "indoorCVPR_09/Images", name)
        label = os.path.dirname(name)
        cls_id = class2id[label]

        if i % 5 == 4:
            # for validation
            val_img_paths.append(path)
            val_cls_ids.append(cls_id)
            val_cls_labels.append(label)
        else:
            # for training
            train_img_paths.append(path)
            train_cls_ids.append(cls_id)
            train_cls_labels.append(label)

    # for test
    for name in test_file_names:
        path = os.path.join(args.dataset_dir, "indoorCVPR_09/Images", name)
        label = os.path.dirname(name)
        cls_id = class2id[label]

        test_img_paths.append(path)
        test_cls_ids.append(cls_id)
        test_cls_labels.append(label)

    # list を DataFrame に変換
    train_df = pd.DataFrame(
        {"image_path": train_img_paths, "class_id": train_cls_ids, "label": train_cls_labels},
        columns=["image_path", "class_id", "label"],
    )

    val_df = pd.DataFrame(
        {"image_path": val_img_paths, "class_id": val_cls_ids, "label": val_cls_labels},
        columns=["image_path", "class_id", "label"],
    )

    test_df = pd.DataFrame(
        {"image_path": test_img_paths, "class_id": test_cls_ids, "label": test_cls_labels},
        columns=["image_path", "class_id", "label"],
    )

    # 保存ディレクトリがなければ，作成
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # 保存
    train_df.to_csv(os.path.join(args.save_dir, "train.csv"), index=None)
    val_df.to_csv(os.path.join(args.save_dir, "val.csv"), index=None)
    test_df.to_csv(os.path.join(args.save_dir, "test.csv"), index=None)

    print("Done")


if __name__ == "__main__":
    main()
