import math
import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import hydra
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torchmetrics.functional import auroc


class CustumAttentionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_length: int,
        text_column_name: str,
        label_column_name: str,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_columm_name = text_column_name
        self.label_column_name = label_column_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        df_row = self.df.iloc[index]
        text = df_row[self.text_columm_name]
        labels = df_row[self.label_column_name].astype(int)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors="pt",
        )

        return encoding["input_ids"].flatten(), torch.tensor(labels)


class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        batch_size,
        max_length,
        text_column_name: str = "text",
        label_column_name: str = "label",
        pretrained_model="cl-tohoku/bert-base-japanese-whole-word-masking",
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.text_columm_name = text_column_name
        self.label_column_name = label_column_name

    def setup(self):
        self.train_dataset = CustumAttentionDataset(
            self.train_df,
            self.tokenizer,
            self.max_length,
            self.text_columm_name,
            self.label_column_name,
        )
        self.vaild_dataset = CustumAttentionDataset(
            self.valid_df,
            self.tokenizer,
            self.max_length,
            self.text_columm_name,
            self.label_column_name,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))

                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret


class CustumAttention(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        d_model: int,
        nhead: int,
        ntimes: int,
        learning_rate: float,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        max_length: int,
        batch_first: bool = True,
        pretrained_model="cl-tohoku/bert-base-japanese-whole-word-masking",
    ):
        super().__init__()

        # モデルの構造
        bert_vec = BertModel.from_pretrained(
            pretrained_model
        ).embeddings.word_embeddings.weight
        self.emb = nn.Embedding.from_pretrained(bert_vec)
        self.posi = PositionalEncoder(d_model, max_length)
        self.multi_atten_list = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    activation,
                    layer_norm_eps,
                    batch_first,
                )
                for _ in range(ntimes)
            ]
        )
        self.classifier = nn.Linear(d_model, n_classes)
        self.lr = learning_rate
        self.ntimes = ntimes
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes

        for param in self.emb.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        outputs = self.emb(inputs)
        outputs = self.posi(outputs)
        for multi_atten in self.multi_atten_list:
            outputs = multi_atten(outputs)
        preds = self.classifier(outputs[:, 0, :])
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(inputs=x)
        loss = self.criterioncriterion(y_hat, y)
        return {"loss": loss, "batch_preds": y_hat, "batch_labels": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(inputs=x)
        loss = self.criterioncriterion(y_hat, y)
        return {"loss": loss, "batch_preds": y_hat, "batch_labels": y}

    def training_epoch_end(self, outputs, mode="train"):
        epoch_y_hats = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_y_hats, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss)

        _, epoch_preds = torch.max(epoch_y_hats, 1)
        epoch_accuracy = accuracy(epoch_preds, epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy)

        epoch_auroc = auroc(epoch_y_hats, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_y_hats = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_y_hats, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss)

        _, epoch_preds = torch.max(epoch_y_hats, 1)
        epoch_accuracy = accuracy(epoch_preds, epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy)

        epoch_auroc = auroc(epoch_y_hats, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def make_callbacks(min_delta, patience, checkpoint_path):
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    return [early_stop_callback, checkpoint_callback]


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    checkpoint_path = os.path.join(cwd, cfg.path.checkpoint_path)
    wandb_logger = WandbLogger(
        name=("exp_" + str(cfg.wandb.exp_num)),
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True,
    )
    wandb_logger.log_hyperparams(cfg)
    df = pd.read_csv(cfg.path.data_file_name).dropna().reset_index(drop=True)
    train, test = train_test_split(df, test_size=cfg.training.test_size, shuffle=True)
    data_module = CreateDataModule(
        train, test, cfg.training.batch_size, cfg.training.max_length, "tweet"
    )
    data_module.setup()

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )
    model = CustumAttention(
        n_classes=cfg.model.n_classes,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        ntimes=cfg.model.ntimes,
        learning_rate=cfg.training.learning_rate,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        layer_norm_eps=cfg.model.layer_norm_eps,
        max_length=cfg.model.max_length,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        gpus=1,
        progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
    )
    trainer.fit(model, data_module)
    wandb_logger.finalize()


if __name__ == "__main__":
    main()
