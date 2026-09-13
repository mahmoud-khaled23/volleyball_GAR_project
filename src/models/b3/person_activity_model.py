import numpy as np
import torch
from torch import nn
import torchvision.models as models

# Baseline 1 is working on the image level with spatial model. >> No temporal <<
# The model is based on fine-tuning pretrained resnet50 on fc7 layer
#
class PersonActivityClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PersonActivityClassifier, self).__init__()
        self.num_classes = num_classes

        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*(list(model.children())[:-1]))

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

    # def state_dicts(self):
    #     model_state_dcts = {
    #         "backbone_state_dict": self.backbone_model.state_dict,
    #         "classifier_state_dict": self.classifier.state_dict
    #     }
    #     return model_state_dcts

    def forward(self, x):
        x = self.backbone_model(x)
        # output.shape == [B * 12, 2048, 1, 1]
        # print(f'B * 12 : {x.size(0)}')  # B * 12
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # def test_model(self, testLoader, device):
    #     self.backbone_model.eval(), self.classifier.eval()
    #     total_correct_predictions = 0
    #     all_predictions = []
    #     for batch_idx, (data, target) in enumerate(testLoader):
    #         data, target = data.to(device), target.to(device)
    #
    #         output = self.backbone_model(data)
    #         output = output.view(output.size(0), -1)
    #         output = self.classifier(output)
    #
    #         prediction = torch.argmax(output, dim=1)
    #         # correct_predictions = sum(pred == tar for pred, tar in zip(prediction, target)).item()
    #         correct_predictions = prediction.eq(target).sum().item()
    #         total_correct_predictions += correct_predictions
    #         all_predictions.append(prediction.numpy())
    #
    #         # if the model or prediction is on gpu, better to use .cpu() as below
    #         # all_predictions.append(prediction.cpu().numpy())
    #
    #     total_acc = total_correct_predictions / len(testLoader.dataset)
    #     return total_acc, np.reshape(all_predictions, -1)

    # def load_state_dict(self, backbone_st, classifier_st):
    #     self.backbone_model.load_state_dict(torch.load(backbone_st, map_location='cpu'))
    #     self.classifier.load_state_dict(torch.load(classifier_st, map_location='cpu'))


