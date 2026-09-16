import numpy as np
import torch
from torch import nn
import torchvision.models as models

# from

# Baseline 1 is working on the image level with spatial model. >> No temporal <<
# The model is based on fine-tuning pretrained resnet50 on fc7 layer
#
class GroupActivityClassifier(nn.Module):
    def __init__(self, person_feature_extraction_model, num_classes):
        super(GroupActivityClassifier, self).__init__()
        self.num_classes = num_classes

        model = nn.Sequential(*(list(person_feature_extraction_model.children())[:-1]))


        fc_layers = nn.Sequential(
            nn.Dropout(0.5, inplace=False),
            nn.Linear(2048, self.num_classes)
        )
        self.backbone_model = model
        self.classifier = fc_layers

        self.optimizer = None
        self.criterion = None
        self.accuracy = None
        self.save_interval = None
        self.early_stopping = None


    def model_summary(self):
        print(f'backbone model\n {self.backbone_model}')

        print(f'classifier')
        print(self.classifier)

    def _optimizers(self, optim):
        optims = dict(
            Adam=torch.optim.Adam([{'params': self.backbone_model.parameters()},
                                   {'params': self.classifier.parameters()}],
                                  lr=optim['lr'],
                                  weight_decay=optim['weight_decay']),
            AdamW=torch.optim.AdamW([{'params': self.backbone_model.parameters()},
                                   {'params': self.classifier.parameters()}],
                                  lr=optim['lr'],
                                  weight_decay=optim['weight_decay']),
            SGD=torch.optim.SGD([{'params': self.backbone_model.parameters()},
                                 {'params': self.classifier.parameters()}],
                                lr=optim['lr'],
                                weight_decay=optim['weight_decay'])
        )
        return optims[optim['optimizer']]

    def metrics(self, optimizer, criterion, accuracy, save_interval=5, early_stopping=None):
        self.optimizer = self._optimizers(optimizer)
        self.criterion = criterion
        self.accuracy = accuracy
        self.save_interval = save_interval
        self.early_stopping = early_stopping

    def forward(self, x):
        x = self.backbone_model(x)
        # output.shape == [B * 12, 2048, 1, 1]
        # print(f'B * 12 : {x.size(0)}')  # B * 12
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

