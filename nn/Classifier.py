import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class ChessClassifier(L.LightningModule):
    def __init__(self, in_features, labels):
        super(ChessClassifier, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(in_features, 1048),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1048, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(50, labels),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.nn(x)

    def training_step(self, batch, batch_idx):
        self.nn.train()

        x, t = batch
        y = self.forward(x)

        loss = F.cross_entropy(y, t)
        acc = torch.mean((torch.argmax(y, dim=1) == torch.argmax(t, dim=1)).float())

        self.log_dict({'train_loss' : loss, 'train_acc' : acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.nn.eval()

        with torch.no_grad():
            features, label = batch
            y = self.forward(features)

            loss = F.cross_entropy(y, label)
            acc = torch.mean((torch.argmax(y, dim=1) == torch.argmax(label, dim=1)).float())

        self.log_dict({'val_loss' : loss, 'val_acc' : acc}, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        self.nn.eval()

        with torch.no_grad():
            features, label = batch
            y = self.forward(features)

            loss = F.cross_entropy(y, label)
            acc = torch.mean((torch.argmax(y, dim=1) == torch.argmax(label, dim=1)).float())

        self.log_dict({'test_loss' : loss, 'test_acc' : acc}, prog_bar=True)

        return loss

    def predict(self, x):
        self.nn.eval()
        
        with torch.no_grad():
            x = x.view((1, -1))
            y = self.forward(x)

        return y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)