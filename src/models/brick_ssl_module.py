from typing import Any, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
import torch.nn.functional as F


class BrickSSLModule(LightningModule):
    """`BrickSSLModule` for training an encoder with InfoNCE Loss.

    This module is designed for contrastive learning tasks where each batch consists
    of pairs of augmented views of the same samples. The InfoNCE loss encourages
    the model to bring positive pairs closer in the embedding space while pushing
    negative pairs apart.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        temperature: float = 0.07,
    ) -> None:
        """Initialize the BrickSSLModule.

        :param net: The encoder network to train.
        :param optimizer: The optimizer class to use (e.g., torch.optim.Adam).
        :param scheduler: The learning rate scheduler class to use (optional).
        :param compile: Whether to compile the model using torch.compile (optional).
        :param temperature: Temperature parameter for InfoNCE loss.
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        self.net = net

        # Temperature parameter for InfoNCE
        self.temperature = temperature

        # Initialize metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the encoder.

        :param x: A tensor of inputs.
        :return: A tensor of normalized embeddings.
        """
        embeddings = self.net(x)  # Shape: (batch_size, embedding_dim)
        embeddings = F.normalize(embeddings, dim=1)  # Normalize embeddings
        return embeddings

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # Reset metrics at the start of training
        self.train_loss.reset()
        self.val_loss.reset()

    def info_nce_loss(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """Compute the InfoNCE loss between two sets of embeddings.

        :param embeddings1: Embeddings from view 1 (batch_size, embedding_dim).
        :param embeddings2: Embeddings from view 2 (batch_size, embedding_dim).
        :return: A scalar tensor representing the InfoNCE loss.
        """
        batch_size = embeddings1.size(0)

        # Compute similarity matrix
        # Shape: (batch_size, batch_size)
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # For numerical stability
        similarity_matrix = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True).values.detach()

        # Labels for contrastive prediction task
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()

        # The diagonal elements are the positive samples
        loss = loss_fn(similarity_matrix, labels)

        if torch.isnan(loss).any():
            print(loss)
            print("Loss nan")
            exit(0)
        return loss

    def training_step(
        self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        :param batch: A batch of data containing two augmented views.
                      Expected format: ((batch_features1, batch_features2), labels)
                      Labels are ignored for contrastive learning.
        :param batch_idx: The index of the current batch.
        :return: The computed InfoNCE loss.
        """
        (batch_features1, batch_features2), _ = batch  # Ignore labels

        # Forward pass through the encoder
        embeddings1 = self.forward(batch_features1)  # Shape: (batch_size, embedding_dim)
        embeddings2 = self.forward(batch_features2)  # Shape: (batch_size, embedding_dim)

        # Compute InfoNCE loss
        loss = self.info_nce_loss(embeddings1, embeddings2)

        # Update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data.

        :param batch: A batch of data containing two augmented views.
                      Expected format: ((batch_features1, batch_features2), labels)
                      Labels are ignored for contrastive learning.
        :param batch_idx: The index of the current batch.
        """
        (batch_features1, batch_features2), _ = batch  # Ignore labels

        # Forward pass through the encoder
        embeddings1 = self.forward(batch_features1)  # Shape: (batch_size, embedding_dim)
        embeddings2 = self.forward(batch_features2)  # Shape: (batch_size, embedding_dim)

        # Compute InfoNCE loss
        loss = self.info_nce_loss(embeddings1, embeddings2)

        # Update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data.

        :param batch: A batch of data containing two augmented views.
                      Expected format: ((batch_features1, batch_features2), labels)
                      Labels are ignored for contrastive learning.
        :param batch_idx: The index of the current batch.
        """
        (batch_features1, batch_features2), _ = batch  # Ignore labels

        # Forward pass through the encoder
        embeddings1 = self.forward(batch_features1)  # Shape: (batch_size, embedding_dim)
        embeddings2 = self.forward(batch_features2)  # Shape: (batch_size, embedding_dim)

        # Compute InfoNCE loss
        loss = self.info_nce_loss(embeddings1, embeddings2)

        # Update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit, validate, test, or predict.

        :param stage: Either "fit", "validate", "test", or "predict".
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        :return: A dict containing the configured optimizers and learning rate schedulers.
        """
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}