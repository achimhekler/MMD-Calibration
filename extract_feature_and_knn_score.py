"""
To save outputs (features or OOD scores) output by a classifier.
"""
from functools import partial
import multiprocessing
from multiprocessing import Pool
import os
import glob
from sklearn import metrics
import sys
import numpy as np
import pickle
from tqdm import tqdm
import time
import copy
import gc
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

import faiss
import torchvision.transforms as transforms
from utils.io_utils import save_pickle, load_pickle
from utils.get_models import FeatureExtractor
from density_aware_calib import KNNScorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features_and_save_ood_scores(
    args,
    split,
    loader,
    feature_extractor,
    to_save_features=False,
    ood_scorer=None,
    save_batch_interval=50,
):
    """
    1. For training data, ood_scorer should be None, to_save_features should be True.
        Then, extracted features and labels are saved.
    2. For test data, ood_scorer should be given, to_save_features should be False.
        Then, extracted features are not saved, but the OOD scores are saved.
    
    Inputs:
        args: 
            arguments
        split: 
            "train" or "test" or other data name
        loader: 
            data loader
        feature_extractor:  
            the classifier we want to calibrate. (create_feature_extractor())
        save_batch_interval: 
            In order to save memory, we save the outputs in batches.
            Decrease this number if memory error occurs.
    """
    print("Split: ", split)
    # save them
    save_d = os.path.join(args.save_outputs_dir, split)
    save_outputs_d = os.path.join(save_d, "outputs")
    save_labels_d = os.path.join(save_d, "labels")
    save_features_d = os.path.join(save_d, "features")
    os.makedirs(save_d, exist_ok=True)
    os.makedirs(save_outputs_d, exist_ok=True)
    os.makedirs(save_labels_d, exist_ok=True)
    if to_save_features:
        os.makedirs(save_features_d, exist_ok=True)

    if not args.force_save:
        if os.path.exists(os.path.join(save_d, f"outputs.pickle")):
            print(f"Already extracted features: ", split)
            return

    # eval mode
    feature_extractor.eval()

    ##############################
    # get features
    batch_sum = len(loader)
    print("total batch size:", batch_sum)

    save_idx = 0
    data = {
        "features": {},
        "outputs": [],
        "labels": [],
    }
    ood_score_dict = {}
    with torch.no_grad():
        for batch_idx, (inputs, labels) in tqdm(enumerate(loader)):
            inputs, labels = inputs.to(device), labels.to(torch.long)

            # get outputs
            features_dict = feature_extractor(inputs)

            for k, feat in features_dict.items():
                feat = feat.detach().cpu().numpy()
                data["features"].setdefault(k, []).append(feat)
                assert len(feat.shape) == 2, feat.shape
                if batch_idx == 0:
                    print(k, feat.shape)

            outputs = features_dict[list(features_dict.keys())[-1]].detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            data["outputs"].append(outputs)
            data["labels"].append(labels)

            del inputs, outputs, labels, feat
            gc.collect()

            to_end = batch_sum * args.train_data_ratio < batch_idx if split == "train" else False
            if (
                (len(data["outputs"]) >= save_batch_interval)
                or (batch_sum == batch_idx + 1)
                or to_end
            ):
                # concat numpy array
                data["outputs"] = np.concatenate(data["outputs"], 0)
                data["labels"] = np.concatenate(data["labels"], 0)
                for k in data["features"]:
                    data["features"][k] = np.concatenate(data["features"][k], 0)

                # calc OOD score
                if ood_scorer is not None:
                    test_feats = [data["features"][k] for k in data["features"]]
                    ood_scores = ood_scorer.get_score(test_feats)
                    for i, k in enumerate(data["features"]):
                        ood_score_dict.setdefault(k, []).append(ood_scores[i])

                # save them
                p = os.path.join(save_outputs_d, f"batch{save_idx}.pickle")
                save_pickle(p, data["outputs"])

                p = os.path.join(save_labels_d, f"batch{save_idx}.pickle")
                save_pickle(p, data["labels"])

                if to_save_features:
                    for k in data["features"]:
                        this_d = os.path.join(save_features_d, k)
                        os.makedirs(this_d, exist_ok=True)
                        p = os.path.join(this_d, f"batch{save_idx}.pickle")
                        save_pickle(p, data["features"][k])

                # reset
                del data
                gc.collect()
                data = {
                    "features": {},
                    "outputs": [],
                    "labels": [],
                }
                print(f"batch: {batch_idx}, save_idx: {save_idx}")
                save_idx += 1

            if to_end:
                print(f"End at data ratio={args.train_data_ratio}. batch_idx={batch_idx}")
                break

    # combine saved files
    combine_batch_data(args, split)

    if ood_scorer is not None:
        ood_score_dict = {k: np.concatenate(v, 0) for k, v in ood_score_dict.items()}
        save_ood_values(args, split, ood_score_dict)

def extract_features_for_mmd_scores(
    args,
    split,
    loader,
    feature_extractor,
    n_tta=9,
):
    """
    1. For training data, ood_scorer should be None, to_save_features should be True.
        Then, extracted features and labels are saved.
    2. For test data, ood_scorer should be given, to_save_features should be False.
        Then, extracted features are not saved, but the OOD scores are saved.
    
    Inputs:
        args: 
            arguments
        split: 
            "train" or "test" or other data name
        loader: 
            data loader
        feature_extractor:  
            the classifier we want to calibrate. (create_feature_extractor())
        save_batch_interval: 
            In order to save memory, we save the outputs in batches.
            Decrease this number if memory error occurs.
    """
    print("Split: ", split)
    # save them
    save_d = os.path.join(args.save_outputs_dir, split, "mmd")
    save_outputs_d = os.path.join(save_d, "outputs")
    save_labels_d = os.path.join(save_d,"labels")
    save_features_d = os.path.join(save_d,"features")
    os.makedirs(save_d, exist_ok=True)
    os.makedirs(save_outputs_d, exist_ok=True)
    os.makedirs(save_labels_d, exist_ok=True)
    os.makedirs(save_features_d, exist_ok=True)

    if not args.force_save:
        if os.path.exists(os.path.join(save_outputs_d, f"outputs.pickle")):
            print(f"Already extracted features: ", split)
            return

    # eval mode
    feature_extractor.eval()

    ##############################
    # get features
    batch_sum = len(loader)
    print("total batch size:", batch_sum)

    save_idx = 0
    data = {
        "features": {},
        "outputs": [],
        "labels": [],
    }

    tta_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    with torch.no_grad():
        for batch_idx, (inputs, labels) in tqdm(enumerate(loader)):
            inputs, labels = inputs.to(device), labels.to(torch.long)

            tta_inputs = []
            for i in inputs:
                tta_inputs.append(i.cpu().numpy())
                for _ in range(n_tta):
                    tta_inputs.append(tta_transforms(i).cpu().numpy())

            tta_inputs = torch.tensor(tta_inputs).to(device)

            # get outputs
            features_dict = feature_extractor(tta_inputs)

            for k, feat in features_dict.items():
                feat = feat.detach().cpu().numpy()
                data["features"].setdefault(k, []).append(feat)
                assert len(feat.shape) == 2, feat.shape
                if batch_idx == 0:
                    print(k, feat.shape)

            outputs = features_dict[list(features_dict.keys())[-1]].detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            data["outputs"].append(outputs)
            data["labels"].append(labels)

            del inputs, outputs, labels, feat, tta_inputs
            gc.collect()

        # concat numpy array
        data["outputs"] = np.concatenate(data["outputs"], 0)
        data["outputs"] = data["outputs"][::n_tta+1]

        data["labels"] = np.concatenate(data["labels"], 0)

        for k in data["features"]:
            data["features"][k] = np.concatenate(data["features"][k], 0)

        # save them
        p = os.path.join(save_outputs_d, f"outputs.pickle")
        save_pickle(p, data["outputs"])

        p = os.path.join(save_labels_d, f"labels.pickle")
        save_pickle(p, data["labels"])

        for k in data["features"]:
            this_d = os.path.join(save_features_d, k)
            os.makedirs(this_d, exist_ok=True)
            p = os.path.join(this_d, f"features.pickle")
            save_pickle(p, data["features"][k])
        
        # reset
        del data
        gc.collect()
        data = {
            "features": {},
            "outputs": [],
            "labels": [],
        }
        print(f"batch: {batch_idx}, save_idx: {save_idx}")

def combine_batch_data(args, split):
    """
    Combine saved batch data into one file.
    """
    print("Split: ", split)
    # save them
    save_d = os.path.join(args.save_outputs_dir, split)
    os.makedirs(save_d, exist_ok=True)
    save_outputs_d = os.path.join(save_d, "outputs")
    save_labels_d = os.path.join(save_d, "labels")
    save_features_d = os.path.join(save_d, "features")

    if not args.force_save and os.path.exists(os.path.join(save_d, "outputs.pickle")):
        print("outputs.pickle already exist.")
    else:
        outputs = []
        i = 0
        while True:
            if not os.path.exists(os.path.join(save_outputs_d, f"batch{i}.pickle")):
                break
            this_outputs = load_pickle(os.path.join(save_outputs_d, f"batch{i}.pickle"))
            outputs.append(this_outputs)
            i += 1
        outputs = np.concatenate(outputs, 0)
        save_pickle(os.path.join(save_d, "outputs.pickle"), outputs)
        del outputs, this_outputs
        gc.collect()
    shutil.rmtree(save_outputs_d, ignore_errors=True)

    if not args.force_save and os.path.exists(os.path.join(save_d, "labels.pickle")):
        print("labels.pickle already exist.")
    else:
        labels = []
        i = 0
        while True:
            if not os.path.exists(os.path.join(save_labels_d, f"batch{i}.pickle")):
                break
            this_labels = load_pickle(os.path.join(save_labels_d, f"batch{i}.pickle"))
            labels.append(this_labels)
            i += 1
        labels = np.concatenate(labels, 0)
        save_pickle(os.path.join(save_d, "labels.pickle"), labels)
        del labels, this_labels
        gc.collect()
    shutil.rmtree(save_labels_d, ignore_errors=True)

    outputs = load_pickle(os.path.join(save_d, "outputs.pickle"))
    labels = load_pickle(os.path.join(save_d, "labels.pickle"))
    if len(labels.shape) == 2:
        labels = np.argmax(labels, 1)
    correct = np.argmax(outputs, 1) == labels
    acc = np.array(correct).mean()
    print(f"Acc ({split}): ", acc)
    if split in ["train", "val", "test"] and acc < 0.50:
        print("Something wrong with accuracy!")
        print(outputs)
        print(np.argmax(outputs, 1))
        print(labels)
        exit()

    if not args.force_save and ((not args.save_only_ood_scores) or (split == "train")):
        feat_list = glob.glob(os.path.join(save_features_d, "*"))
        feat_list = [f.split("/")[-1].replace(".pickle", "") for f in feat_list]
        for k in feat_list:
            this_save_feats_d = os.path.join(save_features_d, k)
            this_feat_save_path = os.path.join(save_features_d, f"{k}.pickle")
            if os.path.exists(this_feat_save_path):
                print(f"{this_feat_save_path} already exist.")
            else:
                feats = []
                i = 0
                while True:
                    this_path = os.path.join(this_save_feats_d, f"batch{i}.pickle")
                    if not os.path.exists(this_path):
                        break
                    this_feats = load_pickle(this_path)
                    feats.append(this_feats)
                    i += 1
                feats = np.concatenate(feats, 0)
                save_pickle(this_feat_save_path, feats)
                del feats, this_feats
                gc.collect()
            shutil.rmtree(this_save_feats_d, ignore_errors=True)

def save_ood_values(args, split, data):
    name = "ood_score"
    if args.save_dist_arr:
        name = "ood_score_arr"

    save_d = os.path.join(args.save_outputs_dir, split, name)
    os.makedirs(save_d, exist_ok=True)

    for k in data:
        p = os.path.join(save_d, f"{k}.pickle")
        save_pickle(p, data[k])
        print(f"Saved {k} to {p}: {data[k].shape}")
    print("Saved ood values: ", save_d)


def str2bool(s):
    if s.lower() in ["t", "true"]:
        return True
    elif s.lower() in ["f", "false"]:
        return False
    else:
        raise ValueError


def MMD(X, Y, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=None)):
    XX = kernel(X, X)
    YY = kernel(Y, Y)
    XY = kernel(X, Y)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def mmd_calc(tensor, layer):
    idx = np.random.randint(0, len(layer), 100)

    mmds_t = []
    for i in idx:
        mmds_t.append(MMD(tensor, layer[i]))

    return np.array(mmds_t).mean()

def calculate_mmd(feats, valid, args, p):
    if args.use_multithreading:
        res = np.array(list(tqdm(p.imap(partial(mmd_calc, layer=valid), feats), total=len(feats))))
        return res
    else:
        raise NotImplementedError("Use multithreading for faster calculation.")

def calculate_mmds(args, split, n_tta=9):
    """
    Calculate MMD scores.
    """
    print("Split: ", split)
    # save them
    save_v = os.path.join(args.save_outputs_dir, "val")
    save_d = os.path.join(args.save_outputs_dir, split)
    save_outputs_d = os.path.join(save_d, "mmd", "outputs")
    save_labels_d = os.path.join(save_d, "mmd", "labels")
    save_features_d = os.path.join(save_d, "mmd", "features")
    save_features_v = os.path.join(save_v, "mmd", "features")
    save_mmd_d = os.path.join(save_d, "mmd", "mmds")
    os.makedirs(save_mmd_d, exist_ok=True)

    if not args.force_save and os.path.exists(os.path.join(save_mmd_d, "mmd.pickle")):
        print("mmd.pickle already exist.")
    else:
        outputs = load_pickle(os.path.join(save_outputs_d, "outputs.pickle"))
        labels = load_pickle(os.path.join(save_labels_d, "labels.pickle"))
        if len(labels.shape) == 2:
            labels = np.argmax(labels, 1)

        mmd_dict = {}
        with Pool(multiprocessing.cpu_count()) as p:
            for k in os.listdir(save_features_d):
                if k == "mmd":
                    continue
                this_save_features_d = os.path.join(save_features_d, k)
                feats = load_pickle(os.path.join(this_save_features_d, "features.pickle"))     

                this_save_features_v = os.path.join(save_features_v, k)
                feats_valid = load_pickle(os.path.join(this_save_features_v, "features.pickle"))

                # destructure the shape of feats
                feats_res = torch.tensor(feats.reshape(-1, n_tta+1, feats.shape[-1]))
                feats_valid_res = torch.tensor(feats_valid.reshape(-1, n_tta+1, feats_valid.shape[-1]))

                del feats, feats_valid
                gc.collect()

                mmd = calculate_mmd(feats_res, feats_valid_res, args, p)
                # TODO Performance
                mmd_dict[k] = mmd

                del feats_res, feats_valid_res
                gc.collect()

        save_pickle(os.path.join(save_mmd_d, "mmd.pickle"), mmd_dict)

if __name__ == "__main__":
    import sys
    import argparse
    from utils.dataset import get_loaders
    from constants.layer_names import get_layers_name
    from constants.dac_hyperparams import hyperparams
    from utils.get_models import get_model

    parser = argparse.ArgumentParser(description="")
    # dataset
    parser.add_argument("--data_root", type=str, default="../")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10")
    # model
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="")
    # save args
    parser.add_argument("--force_save", type=str2bool, default=False)
    parser.add_argument("--no_state_dict", type=str2bool, default=False)
    parser.add_argument("--save_outputs_root_dir", type=str, default="outputs")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--number_tta", type=int, default=9)
    parser.add_argument("--is_all_layers", type=str2bool, default=False)
    parser.add_argument("--save_only_ood_scores", type=str2bool, default=True)
    parser.add_argument("--save_dist_arr", type=str2bool, default=False)
    parser.add_argument("--use_multithreading", type=str2bool, default=False)
    parser.add_argument("--train_data_ratio", type=float, default=1.0)
    parser.add_argument("--save_batch_interval", type=int, default=100)
    parser.add_argument("--exp_id", type=string, default='')
    parser.add_argument(
        "--test_data_type", type=str, nargs="*", 
        default=[],
        choices=[
            "natural",
            "gaussian_noise",
            "shot_noise",
            "speckle_noise",
            "impulse_noise",
            "defocus_blur",
            "gaussian_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "spatter",
            "saturate",
            "frost",
        ]
    )

    args = parser.parse_args()

    # load DAC hyperparams
    args.num_classes = hyperparams[args.dataset]["num_classes"]
    args.top_k = hyperparams[args.dataset]["knn_k"]
    args.train_data_ratio = hyperparams[args.dataset]["train_data_ratio"]

    print("\n===============")
    print("Dataset: ", args.dataset)
    print("Model: ", args.model_name)
    print("Top k for KNN score: ", args.top_k)
    print("Ratio of train set to use: ", args.train_data_ratio)
    print("===============\n")


    ###### Get dataloader ########
    train_loader, val_loader, test_loader = get_loaders(
        name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_no_aug=True,  # important
    )
    root_dir = args.save_outputs_root_dir + args.exp_id
    args.save_outputs_dir = os.path.join(root_dir, f"{args.dataset}/{args.model_name}")
    os.makedirs(args.save_outputs_dir, exist_ok=True)

    ###### load classifier ########
    model = get_model(args.model_name, args.num_classes)
    if not args.no_state_dict:
        model.load_state_dict(torch.load(args.model_path)["state_dict"])
    else:
        model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    print("loaded model weights: ", args.model_path)

    ###### Feature Extractor ########
    if args.is_all_layers:
        return_nodes = get_layers_name(
            args.model_name, model=model, get_all=True, add_logits=True
        )
        feature_extractor = FeatureExtractor(model, return_nodes)
    else:
        return_nodes = get_layers_name(args.model_name, model=model, add_logits=True)
        feature_extractor = FeatureExtractor(model, return_nodes)
    layers_name = [v for k, v in return_nodes.items()] if isinstance(return_nodes, dict) else return_nodes

    if torch.cuda.device_count() >= 2:
        feature_extractor = torch.nn.DataParallel(feature_extractor).cuda()

    ###### Extract features and save KNN scores ########
    # for train set
    start_time = time.time()
    split = "train"
    extract_features_and_save_ood_scores(
        args,
        split,
        train_loader,
        feature_extractor,
        to_save_features=True, # need to save features for train set
        ood_scorer=None,
        save_batch_interval=args.save_batch_interval,
    )
    end_time = time.time()
    print(f"- Train feature extraction: {end_time - start_time} seconds")

    # set ood_scorer
    train_save_d = os.path.join(args.save_outputs_dir, "train")
    train_labels = load_pickle(os.path.join(train_save_d, "labels.pickle"))
    ood_scorer = KNNScorer(top_k=args.top_k, return_dist_arr=args.save_dist_arr, gpu=True)
    for layer in layers_name:
        path = os.path.join(train_save_d, "features", f"{layer}.pickle")
        train_feat = load_pickle(path)
        print("Set train features to ood_scorer: {}, {}".format(layer, train_feat.shape))
        ood_scorer.set_train_feat(train_feat, train_labels, args.num_classes)

    # for val, test set
    for split, loader in zip(["val", "test"], [val_loader, test_loader]):
        print("Save OOD scores: ", split)
        ood_score_dict = extract_features_and_save_ood_scores(
            args,
            split,
            loader,
            feature_extractor,
            to_save_features=False,
            ood_scorer=ood_scorer,
            save_batch_interval=args.save_batch_interval,
        )

        print("Calculate MMD Scores: ", split)
        # for mmd scores
        extract_features_for_mmd_scores(
            args,
            split,
            loader,
            feature_extractor,
            n_tta=args.number_tta,
        )
        calculate_mmds(args, split, args.number_tta)


    # for corruption data
    assert args.dataset in ["cifar10", "cifar100", "imagenet"]

    test_corruptions = args.test_data_type
    severities = [1, 2, 3, 4, 5]

    for ci, cname in enumerate(test_corruptions):
        for severity in severities:
            cname_s = f"{cname}_{severity}"
            print("Save OOD scores: ", cname_s)

            loader = get_loaders(
                f"{args.dataset}c",
                cname=cname,
                data_root = args.data_root,
                batch_size=args.batch_size,
                severity=severity,
            )

            ood_score_dict = extract_features_and_save_ood_scores(
                args,
                cname_s,
                loader,
                feature_extractor,
                to_save_features=False,
                ood_scorer=ood_scorer,
                save_batch_interval=args.save_batch_interval,
            )
            
            extract_features_for_mmd_scores(
                args,
                cname_s,
                loader,
                feature_extractor,
                n_tta=args.number_tta,
            )
            calculate_mmds(args, cname_s, args.number_tta)