import math
import numpy as np
import torch
from torch import nn
import torchvision.models as models
import torch.optim as optim

import os
import pickle
from src.volleyball_data_loader import VolleyBallDataSet
from src.volleyball_data_loader import get_preprocess
from src.data_helper import get_root_dirs
from torch.utils.data import DataLoader
from src.utils import EarlyStopping


class ImageLevelModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageLevelModel, self).__init__()
        self.backbone_model = None
        self.classifier = None
        self.num_classes = num_classes

        self.optimizer = None
        self.criterion = None
        self.accuracy = None
        self.save_interval = None
        self.early_stopping = None

        self._prepare_model()

    def _prepare_model(self):
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*(list(model.children())[:-1]))

        fc_layers = nn.Sequential(
            nn.Dropout(0.5, inplace=False),
            nn.Linear(2048, 16),
            nn.BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(16, self.num_classes)
        )
        self.backbone_model = model
        self.classifier = fc_layers

    def model_summary(self):
        print(f'backbone model')
        print(self.backbone_model)

        print(f'classifier')
        print(self.classifier)

    def _optimizers(self, optim):
        optims = dict(
            Adam=torch.optim.Adam([{'params': self.backbone_model.parameters()},
                                   {'params': self.classifier.parameters()}],
                                  lr=optim['lr'],
                                  weight_decay=optim['weight_decay']),
            SGD=torch.optim.SGD([{'params': self.backbone_model.parameters()},
                                 {'params': self.classifier.parameters()}],
                                lr=optim['lr'],
                                weight_decay=optim['weight_decay'])
        )
        return optims[optim['optimizer']]

    def set_metrics(self, optimizer, criterion, accuracy, save_interval=5, early_stopping=None):
        self.optimizer = self._optimizers(optimizer)
        self.criterion = criterion
        self.accuracy = accuracy
        self.save_interval = save_interval
        self.early_stopping = early_stopping

    def state_dicts(self):
        model_state_dcts = {
            "backbone_model": self.backbone_model.state_dict,
            "classifier_model": self.classifier.state_dict
        }
        return model_state_dcts

    def train_model(self, trainLoader, backbone_model, classifier, optimizer, device):
        backbone_model.train()
        classifier.train()

        criterion = self.criterion
        running_loss = 0
        total_correct_predictions = 0

        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = backbone_model(data)
            output = output.view(output.size(0), -1)
            output = classifier(output)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

            prediction = torch.argmax(output, dim=1)
            correct_predictions = sum(pred == tar for pred, tar in zip(prediction, target)).item()

            total_correct_predictions += correct_predictions

        total_loss = running_loss / len(trainLoader.dataset)
        total_accuracy = total_correct_predictions / len(trainLoader.dataset)

        return backbone_model, classifier, optimizer, total_loss, total_accuracy

    def eval_model(self, valLoader, backbone_model, classifier, device):
        backbone_model = backbone_model
        classifier = classifier
        criterion = self.criterion

        running_loss = 0
        total_correct_predictions = 0

        with torch.no_grad():
            backbone_model.eval(), classifier.eval()

            for batch_idx, (data, target) in enumerate(valLoader):
                data, target = data.to(device), target.to(device)

                output = backbone_model(data)
                output = output.view(output.size(0), -1)
                output = classifier(output)

                loss = criterion(output, target)
                running_loss += loss.item() * data.size(0)

                prediction = torch.argmax(output, dim=1)
                correct_predictions = sum(pred == tar for pred, tar in zip(prediction, target)).item()

                total_correct_predictions += correct_predictions

        total_loss = running_loss / len(valLoader.dataset)
        total_acc = total_correct_predictions / len(valLoader.dataset)

        return total_loss, total_acc

    def forward(self, trainLoader, valLoader, epochs, output_path, device):
        # print(self.backbone_model)

        backbone_model = self.backbone_model
        classifier = self.classifier
        
        if torch.cuda.is_available():
            backbone_model.cuda()
            classifier.cuda()

        save_interval = self.save_interval
        optimizer = self.optimizer

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        train_dataset_length = len(trainLoader.dataset)
        num_of_steps = 0
        backbone_model_best_weights = None
        classifier_model_best_weights = None

        for epoch in range(epochs):
            num_of_steps += train_dataset_length
            print(f'epoch: {epoch + 1}/{epochs}, steps: {num_of_steps}/{train_dataset_length * epochs}')
            backbone_model, classifier, optimizer, train_loss, train_accuracy = self.train_model(trainLoader,
                                                                                                 backbone_model,
                                                                                                 classifier,
                                                                                                 optimizer,
                                                                                                 device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_loss, val_accuracy = self.eval_model(valLoader, backbone_model, classifier, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f'\ttrain loss: {train_loss:.4f} - train accuracy: {(train_accuracy * 100):.2f}%,'
                  f' val loss: {val_loss: .4f} - val accuracy: {(val_accuracy * 100): .2f} % ')

            if epochs - epoch <= 5:
                checkpoint = {
                    "epoch": epoch + 1,
                    "backbone_model_state_dict": self.backbone_model.state_dict(),
                    "classifier_state_dict": self.classifier.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                checkpoint_filename = f'{output_path}/volleyball_checkpoint_on_epoch_{epoch + 1}.pth'
                torch.save(checkpoint, checkpoint_filename)

            self.early_stopping(val_loss, self)
            if self.early_stopping.best_model:
                backbone_model_best_weights = self.backbone_model.state_dict
                classifier_model_best_weights = self.classifier.state_dict

            if self.early_stopping.early_stop:
                break
        backbone_filename = f'{output_path}/backbone_model_state_dict.pth'
        classifier_filename = f'{output_path}/classifier_state_dict.pth'
        torch.save(backbone_model_best_weights, backbone_filename)
        torch.save(classifier_model_best_weights, classifier_filename)

        loss_acc_epochs = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_acc": train_accuracies,
            "val_acc": val_accuracies
        }



        with open(f'{output_path}/loss_acc.pickle', 'wb') as file:
            pickle.dump(loss_acc_epochs, file)

    def test_model(self, testLoader, device):
        self.backbone_model.eval(), self.classifier.eval()
        total_correct_predictions = 0
        all_predictions = []
        for batch_idx, (data, target) in enumerate(testLoader):
            data, target = data.to(device), target.to(device)

            output = self.backbone_model(data)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)

            prediction = torch.argmax(output, dim=1)
            correct_predictions = sum(pred == tar for pred, tar in zip(prediction, target)).item()

            total_correct_predictions += correct_predictions
            all_predictions.append(prediction.numpy())

        total_acc = total_correct_predictions / len(testLoader.dataset)
        return total_acc, np.reshape(all_predictions, -1)

    def load_state_dicts(self, backbone_st, classifier_st):
        self.backbone_model.load_state_dict(torch.load(backbone_st, map_location='cpu'))
        self.classifier.load_state_dict(torch.load(classifier_st, map_location='cpu'))


if __name__ == '__main__':
    root, root_dataset, root_videos, root_output = get_root_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo = models.resnet50(pretrained=True)
    # print(modelo)

    train_annot_dct = root_dataset + "structured-data/volleyball-annotations/train-target-annot.pickle"
    val_annot_dct = root_dataset + "structured-data/volleyball-annotations/val-target-annot.pickle"

    preprocess = get_preprocess()

    with open(train_annot_dct, 'rb') as tr, open(val_annot_dct, 'rb') as vl:
        train_data = pickle.load(tr)
        val_data = pickle.load(vl)

    batch_size = 32
    train_loader = DataLoader(VolleyBallDataSet(root_videos, train_data, preprocess=preprocess),
                              batch_size=batch_size)
    val_loader = DataLoader(VolleyBallDataSet(root_videos, val_data, preprocess=preprocess),
                            batch_size=batch_size)

    num_classes = 8
    my_model = ImageLevelModel(num_classes)

    optim_params = {
        "optimizer": "Adam",
        "lr": 1e-2,
        "weight_decay": 1e-3
    }
    criterion = torch.nn.CrossEntropyLoss()
    acc = "accuracy"
    save_interval = 10
    early_stopping = EarlyStopping()
    my_model.set_metrics(optimizer=optim_params,
                         criterion=criterion,
                         accuracy=acc,
                         save_interval=save_interval,
                         early_stopping=early_stopping)

    epochs = 50
    # my_model.forward(train_loader, val_loader, epochs, output_path=root_output, device=device)
