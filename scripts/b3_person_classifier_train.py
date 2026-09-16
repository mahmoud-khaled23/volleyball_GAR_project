import torch
import pickle

from torch.utils.data import DataLoader
from src.utils.utils import EarlyStopping

from src.data_utils.volleyball_dataset import VolleyBallPersonDataset
from src.utils.preprocessors import person_preprocessor
from src.models.b3.person_activity_model import PersonActivityClassifier

from src.utils.utils import load_config
from src.paths import CONFIGS_DIR, OUTPUT_DIR, PERSON_ANNOTATIONS_DIR

from src.utils.utils import save_checkpoint



def train_epoch(model, trainLoader, optimizer, criterion, device):
    model.train()

    running_loss = 0
    epoch_correct_predictions = 0

    for batch_idx, (data, target) in enumerate(trainLoader):
        # The input shape is x: [B, 12, 3, 224, 224]
        B, P, C, H, W = data.shape

        data = data.view(B * P, C, H, W)
        target = target.view(B * P)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

        prediction = torch.argmax(output, dim=1)
        batch_correct_predictions = (prediction == target).sum().item()

        epoch_correct_predictions += batch_correct_predictions

    epoch_loss = running_loss / trainLoader.dataset.len_12_players()
    epoch_accuracy = epoch_correct_predictions / trainLoader.dataset.len_12_players()

    return model, optimizer, epoch_loss, epoch_accuracy


def eval_model(model, valLoader, criterion, device):
    model.eval()
    running_loss = 0
    epoch_correct_predictions = 0

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(valLoader):
            B, P, C, H, W = data.shape
            data = data.view(B * P, C, H, W)
            target = target.view(B * P)

            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)

            prediction = torch.argmax(output, dim=1)
            batch_correct_predictions = (prediction == target).sum().item()

            epoch_correct_predictions += batch_correct_predictions

    epoch_loss = running_loss / valLoader.dataset.len_12_players()
    epoch_accuracy = epoch_correct_predictions / valLoader.dataset.len_12_players()

    return epoch_loss, epoch_accuracy


def fit(model, trainLoader, valLoader, epochs, optimizer, criterion, output_dir, device):
    model.to(device)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    train_dataset_length = len(trainLoader.dataset)
    num_of_steps = 0

    for epoch in range(epochs):
        num_of_steps += train_dataset_length
        print(f'epoch: {epoch + 1}/{epochs}, steps: {num_of_steps}/{train_dataset_length * epochs}')
        model, optimizer, train_loss, train_accuracy = train_epoch(model,
                                                                   trainLoader,
                                                                   optimizer,
                                                                   criterion,
                                                                   device)

        val_loss, val_accuracy = eval_model(model, valLoader, criterion, device)

        print(f'\ttrain loss: {train_loss:.4f} - train accuracy: {(train_accuracy * 100):.2f}%,'
              f' val loss: {val_loss:.4f} - val accuracy: {(val_accuracy * 100):.2f}% ')
        # todo
        if epoch % model.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, output_dir)

        # todo
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

from src.data_utils.volleyball_dataset import DummyPlayerDataset

def train_model(model_configs, train_configs, output_checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_path = PERSON_ANNOTATIONS_DIR / 'train_players_crops.pickle'
    val_data_path = PERSON_ANNOTATIONS_DIR / 'val_players_crops.pickle'

    train_preprocess, val_preprocessor = person_preprocessor()

    with open(train_data_path, 'rb') as tr, open(val_data_path, 'rb') as vl:
        train_data = pickle.load(tr)
        val_data = pickle.load(vl)

    dataset = DummyPlayerDataset()

    dummies_dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )
    batch_size = train_configs['batch_size']
    num_workers = train_configs['num_workers']

    train_loader = DataLoader(VolleyBallPersonDataset(train_data, preprocess=train_preprocess),
                              batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(VolleyBallPersonDataset(val_data, preprocess=val_preprocessor),
                            batch_size=batch_size, num_workers=num_workers)

    num_classes = model_configs['num_classes']
    model = PersonActivityClassifier(num_classes)

    optim_params = {
        "optimizer": train_configs['optimizer'],
        "lr": train_configs['learning_rate'],
        "weight_decay": train_configs['weight_decay']
    }

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    accuracy = "accuracy"

    save_interval = 5
    if train_configs['early_stopping']:
        early_stopping = EarlyStopping()
    else:
        early_stopping = None

    model.metrics(optimizer=optim_params,
                  criterion=criterion,
                  accuracy=accuracy,
                  save_interval=save_interval,
                  early_stopping=early_stopping)

    epochs = train_configs['epochs']

    model.model_summary()
    model.fit(train_loader, val_loader, epochs, output_dir=output_checkpoint_dir, device=device)

if __name__ == '__main__':
    b3_config_path = CONFIGS_DIR / 'b3_person_train_config.yaml'
    b3_config = load_config(b3_config_path)

    model_configs = b3_config.model
    train_configs = b3_config.training
    b3_person_checkpoint_dir = OUTPUT_DIR / 'b3' / 'person_activity'

    # train_model(model_configs, train_configs, b3_person_checkpoint_dir)
    print(train_configs)



