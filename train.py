import argparse
import os
import sys

import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from lightning import LightningDataModule, LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


class Cfg:
    num_workers = 4
    batch_size = 512

    max_epochs = 10

    path_train = "../data/sign_mnist_train.csv"
    path_val = "../data/sign_mnist_test.csv"


class SignLanguageDataset(data.Dataset):

    def __init__(self, df, transform=None):

        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        label = self.df.iloc[index, 0]

        img = self.df.iloc[index, 1:].values.reshape(28, 28)
        img = torch.Tensor(img).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


transforms4train = transforms.Compose(
    [
        # transforms.Normalize(159, 40),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180))], p=0.2),
    ]
)


class SignData(LightningDataModule):
    def __init__(self):
        # Сохраняем переменные для дальнейшей работы
        super().__init__()

    def prepare_data(self):
        assert os.path.exists(Cfg.path_train), "Файл sign_mnist_train.csv не найден"
        assert os.path.exists(Cfg.path_val), "Файл sign_mnist_test.csv не найден"

        self.train = pd.read_csv(Cfg.path_train)
        self.val = pd.read_csv(Cfg.path_val)

    def _make_dataloader(self, dataset):
        # Общий метод для создания DataLoader
        return DataLoader(
            dataset,
            batch_size=Cfg.batch_size,
            num_workers=Cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self):
        train_dataset = SignLanguageDataset(self.train, transform=transforms4train)
        return self._make_dataloader(train_dataset)

    def val_dataloader(self):
        val_dataset = SignLanguageDataset(self.val)
        return self._make_dataloader(val_dataset)

    def teardown(self, stage: str):
        # Функция, которая выполняется после создания Dataloader - здесь можно удалить ненужные компоненты
        del self.train, self.val


class MyConvNet(nn.Module):

    def __init__(self, stride=1, dilation=1, n_classes=25):

        super().__init__()

        self.stride = stride
        self.dilation = dilation
        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            # input=(batch, 1, 28, 28)
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=self.stride, dilation=self.dilation
            ),
            nn.BatchNorm2d(8),
            # (batch, 8, 28, 28)
            nn.AvgPool2d(2),
            # (batch, 8, 14, 14)
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=self.stride, dilation=self.dilation
            ),
            nn.BatchNorm2d(16),
            # (batch, 16, 14, 14)
            nn.AvgPool2d(2),
            # (batch, 16, 7, 7)
            nn.ReLU(),
        )

        self.lin1 = nn.Linear(in_features=16 * 7 * 7, out_features=100)
        # (batch, 100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(100, self.n_classes)
        # (batch, 25)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)

        return x


class LightningModel(LightningModule):  # Элемент от Lightning
    def __init__(self):
        super().__init__()

        self.mlp = MyConvNet()
        self.classification_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.mlp.forward(x)

    def basic_step(self, batch, batch_idx, step: str):
        # Получаем данные
        x, y = batch

        pred_clas = self(x)

        loss_class = self.classification_criterion(pred_clas, y)

        loss_dict = {
            f"{step}/loss": loss_class,
        }

        self.log_dict(loss_dict, prog_bar=True)

        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.basic_step(batch, batch_idx, "train")
        return loss_dict["train/loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.basic_step(batch, batch_idx, "valid")
        return loss_dict["valid/loss"]

    def test_step(self, batch, batch_idx):
        loss_dict = self.basic_step(batch, batch_idx, "test")
        return loss_dict["test/loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "epoch",
                "reduce_on_plateau": True,
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", type=bool, default=False, help="Run quick test loop")

    args = parser.parse_args()

    trainer = Trainer(
        profiler="simple",
        max_epochs=Cfg.max_epochs,
        log_every_n_steps=1,
        fast_dev_run=args.fast_dev_run,
    )
    model = LightningModel()

    dataset = SignData()
    dataset.setup("fit")

    if args.fast_dev_run:
        try:
            trainer.fit(model, datamodule=dataset)
            print("Тестовый прогон успешно пройден. Далее работа скрипта продолжается в обычном режиме")
        except Exception as e:
            print("Тестовый прогон завершился с ошибкой")
            sys.exit(1)

    trainer.fit(model, datamodule=dataset)
