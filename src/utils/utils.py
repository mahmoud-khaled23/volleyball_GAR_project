import yaml

class Config:
    def __init__(self, config_dict):
        self.paths = config_dict.get("paths", {})
        self.dataset = config_dict.get("dataset", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})
        self.experiment = config_dict.get("experiment", {})

    def __repr__(self):
        return f"Config(model={self.model}, training={self.training}, data={self.dataset}, experiment={self.experiment})"


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)
    return config


import torch

def save_checkpoint(model, optimizer, epoch, val_loss, save_path):
    # Todo later
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(checkpoint, save_path / 'checkpoint.pth')


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, best_model_path: str = 'best_checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_model_path = best_model_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.best_model_path)


def seed(seed):
    # Todo, The code below is generated
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True




