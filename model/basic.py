import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torchvision import models


class MLPModel(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(84 * 84, 256)
        self.l2 = torch.nn.Linear(256, 4)

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_f1 = torchmetrics.F1Score(average='micro')

        self.learning_rate = lr

    def forward(self, x):
        x = x.view(-1, 84 * 84)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {"loss": loss}
        
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.valid_f1(y_hat, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=False)
        self.log('val_f1', self.valid_f1, on_step=True, on_epoch=False)
        self.log('val_loss', loss)
        return {"val_loss": loss}
    
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"test_loss": loss}
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class CNNModel(pl.LightningModule):
    def __init__(self, lr=5e-4):
        super().__init__()
        self.save_hyperparameters()

        # 3 convolutional layers (sees 84x84) ending with mlp with 4 classes
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(64 * 21 * 21, 256)
        self.fc2 = torch.nn.Linear(256, 4)

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_f1 = torchmetrics.F1Score(average='micro')
        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1Score(average='micro')

        self.learning_rate = lr

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 21 * 21)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.valid_f1(y_hat, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=False)
        self.log('val_f1', self.valid_f1, on_step=True, on_epoch=False)
        self.log('val_loss', loss)
        return {"val_loss": loss}

    
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        metrics = {"test_acc": self.test_f1, "test_loss": loss, "test_f1": self.test_f1}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Resnet model
class ResNetModel(pl.LightningModule):
    def __init__(self, lr=5e-4):
        super().__init__()
        self.save_hyperparameters()

        self.resnet = models.resnet18(pretrained=False)
        linear_size = list(self.resnet.children())[-1].in_features
        self.resnet.fc = torch.nn.Linear(linear_size, 4)

        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_f1 = torchmetrics.F1Score(average='micro')
        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1Score(average='micro')

        self.learning_rate = lr

    def forward(self, x):
        return self.resnet(x)
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.valid_f1(y_hat, y)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=False)
        self.log('val_f1', self.valid_f1, on_step=True, on_epoch=False)
        self.log('val_loss', loss)
        return {"val_loss": loss}
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        metrics = {"test_acc": self.test_f1, "test_loss": loss, "test_f1": self.test_f1}
        self.log_dict(metrics)
        return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Simple LSTM with 8 84x84 vectors as input
class CnnLSTMModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # to black and white
        linear_size = list(self.resnet.children())[-1].in_features
        self.resnet.fc = torch.nn.Linear(linear_size, 300)


        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 4)

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.valid_f1 = torchmetrics.F1Score(average='micro')
        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1Score(average='micro')

        self.learning_rate = lr

    def forward(self, x):
        hidden = None
        for t in range(x.size(1)):
            x_t = x[:, t, :, :]
            with torch.no_grad():
                x_t = x_t[None, :]
                x_t = torch.permute(x_t, (1, 0, 2, 3))
                x_t = self.resnet(x_t)
            x_t = x_t.view(x_t.size(0), -1)
            out, hidden = self.lstm(x_t.unsqueeze(0), hidden)
        x = self.fc1(out.squeeze(0))
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.valid_f1(y_hat, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=False)
        self.log('val_f1', self.valid_f1, on_step=True, on_epoch=False)
        self.log('val_loss', loss)
        return {"val_loss": loss}
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        metrics = {"test_acc": self.test_f1, "test_loss": loss, "test_f1": self.test_f1}
        self.log_dict(metrics)
        return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
