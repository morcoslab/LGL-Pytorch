import torch
from .model import VAE


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        # one hot data in, return flattened vectors for input
        self.data = data.flatten(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EarlyStopping:
    def __init__(self, min_delta=0.0, patience=0) -> None:
        self.min_delta = min_delta
        self.patience = patience
        self.best = float("inf")
        self.wait = 0
        self.done = False

    def __call__(self, current) -> bool:
        self.wait += 1

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        elif self.wait >= self.patience:
            self.done = True

        return self.done


class Trainer:
    def __init__(
        self,
        learning_rate: float = 1e-4,
        early_stop_patience: int = 10,
        batch_size: int = 16,
        regularization: float = 1e-2,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.earlystopping = EarlyStopping(patience=early_stop_patience)
        self.training_log = {"loss": list(), "reconstruction": list(), "kld": list()}

    def createDataLoader(
        self,
        data: torch.Tensor,
        batch_size: int = 0,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        if batch_size == 0:
            batch_size = self.batch_size

        dataset = Dataset(data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        return dataloader

    def createOptimizer(self, model: VAE) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(), self.learning_rate, weight_decay=self.regularization
        )

    def train(self, model: VAE, dataset: torch.Tensor, num_epochs: int = 10_000) -> VAE:
        dataloader = self.createDataLoader(dataset)
        optimizer = self.createOptimizer(model)

        for epoch in range(num_epochs):
            epoch_losses = {"loss": 0, "reconstruction": 0, "kld": 0}
            for batch in dataloader:
                output = model.compute_elbo(batch)
                optimizer.zero_grad()
                output["loss"].backward()
                optimizer.step()
                for term in output.items():
                    epoch_losses[term[0]] += term[1].item()

            # Log epoch averages
            for term in epoch_losses.keys():
                epoch_losses[term] /= len(dataset)
                self.training_log[term].append(epoch_losses[term])

            print(
                f"Epoch {epoch+1}: "
                + ", ".join([f"{r[0]}:{round(r[1],3)}" for r in epoch_losses.items()])
            )

            # early stopping
            if self.earlystopping(epoch_losses["loss"]):
                print("Early stopping, training has completed.")
                return model
        return model

    def getLandscapeBounds(
        self, model: VAE, dataset: torch.Tensor, batch_size: int = 10_000
    ) -> tuple:
        """For the landscape, checks all training data and finds min/max z0, z1 values."""
        bounds = [0, 0, 0, 0]  # z0min,z0max,z1min,z1max
        dataloader = self.createDataLoader(dataset, batch_size=batch_size)
        for batch in dataloader:
            hidden_output = model.encoder_base(batch)
            mu_ouptut = model.encoder_mu(hidden_output)

            bounds[0] = min(bounds[0], mu_ouptut[:, 0].min().item())
            bounds[2] = min(bounds[2], mu_ouptut[:, 1].min().item())
            bounds[1] = max(bounds[1], mu_ouptut[:, 0].max().item())
            bounds[3] = max(bounds[3], mu_ouptut[:, 1].max().item())

        # pad things out, even them up
        for idx, bound in enumerate(bounds):
            if bound < 0:
                bounds[idx] = -(round(abs(bound)) + 1)
            else:
                bounds[idx] = round(bound) + 1

        return (min(bounds), max(bounds))
