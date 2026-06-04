import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from src.baselines.b1.b1 import ImageLevelModel
from src.data_helper import get_root_dirs
from src.data_helper import b1_load

import pickle
import torch
from torch.utils.data import DataLoader
from src.volleyball_data_loader import VolleyBallDataSet
from src.volleyball_data_loader import get_preprocess
import os

def loss_acc_plot(train_losses, train_accuracies, val_losses, val_accuracies):

    # plt.subplots(1, 2, 1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Train and Val Loss through Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.show()


if __name__ == '__main__':
    root, root_dataset, root_videos, root_output = get_root_dirs()

    fpath = root_output+'/b1-params'
    model = ImageLevelModel(8)

    backbone_model_state_path = os.path.join(fpath, 'backbone_model_state_dict.pth')
    classifier_model_state_path = os.path.join(fpath, 'classifier_state_dict.pth')
    with open(backbone_model_state_path, 'rb') as backbone, open(classifier_model_state_path, 'rb') as classifier:
        model.load_state_dicts(backbone, classifier)

    with open(os.path.join(root_output,"structured-data/volleyball-annotations/test-target-annot.pickle"), "rb") as f:
        test_data = pickle.load(f)

    batch_size = 8
    _, test_preprocess = get_preprocess()
    test_loader = DataLoader(VolleyBallDataSet(root_videos, test_data, preprocess=test_preprocess, shuffle=False), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    test_acc, preds = model.test_model(test_loader, device)
    with open(root_output+"/b1/test-predicts.pickle", "wb") as preds_file:
        pickle.dump(preds, preds_file)

    print(test_acc)

    loss_acc_filepath = root_output+"/b1/loss_acc.pickle"
    with open(loss_acc_filepath, 'rb') as file:
        loss_acc = pickle.load(file)

        train_losses = loss_acc["train_losses"]
        val_losses = loss_acc["val_losses"]
        train_accuracies = loss_acc["train_acc"]
        val_accuracies = loss_acc["val_acc"]

    loss_acc_plot(train_losses, train_accuracies, val_losses, val_accuracies)
