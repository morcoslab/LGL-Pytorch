import os
import pickle
from typing import List, Union

import numpy as np
import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from dca.dca_class import dca
from dca.dca_functions import return_Hamiltonian
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from .model import VAE
from .seq_utils import one_hot_encode_fasta, seq_code
from .train_utils import Trainer


class LGLVAE:
    def __init__(self, fasta_fn: str, alphabet: dict = seq_code) -> None:
        """Use createVAE() to train the VAE model using the fasta_fn and expected alphabet dictionary.
        Use createDCA() to create a DCA model using the fasta_fn.
        Use createLGL() to create the landscape grid data.
        Use plotSequences(fasta_fn) to create coordinates with the VAE.
        Use generateSequences(coordinates) to create sequences with the VAE.
        """
        self.fasta = fasta_fn
        self.alphabet = alphabet
        self.VAETrainer = Trainer()

    def createVAE(self, output_fn: str = "", device: str = "detect") -> None:
        """Loads data, and trains the VAE model according to our default parameters.
        If output_fn is provided, it will save a pickle of the LGLVAE class.
        (You should really save the output of this)"""

        # Use user settings, or detect and use GPU->MPS->CPU
        if device != "detect":
            device = device
        elif hasattr(torch, "cuda") and torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Training using device: {device}")

        one_hot_data = one_hot_encode_fasta(
            self.fasta, self.alphabet, device=device
        )
        input_dim = (
            one_hot_data.shape[1] * one_hot_data.shape[2]
        )  # size when flattened
        hidden_units = one_hot_data.shape[1]  # sequence length

        model = VAE(input_dim, hidden_units, num_aa=one_hot_data.shape[2])

        # move data to device
        model.to(device)
        comp_model = torch.compile(model)

        # Train model
        comp_model = self.VAETrainer.train(comp_model, one_hot_data)

        # Move model to CPU for regular usage
        model.to("cpu")

        # assign saved model as new variable (comp_model does not pickle, save original)
        self.VAE = model

        # Save LGLVAE class with filename (if provided)
        self.save(output_fn)

    def createDCA(
        self, output_fn: str = "", cdist_batch_size: int = 50_000
    ) -> None:
        """If you run out of memory while running this function (sorry),
        lower the value of cdist_batch_size (at the cost of some speed).
        If output_fn is provided, it will save a pickle of the LGLVAE class.
        (You should really save the output of this)."""

        self.DCA = dca(self.fasta, stype=self.alphabet)
        self.DCA.mean_field(cdist_batch_size=cdist_batch_size)

        # Save LGLVAE class with filename (if provided)
        self.save(output_fn)

    def createLGL(
        self,
        output_fn: str = "",
        batch_size: int = 10_000,
        resolution: int = 500,
    ) -> None:
        """Creates the latent generative landscape.
        If output_fn is provided, it will save a pickle of the LGLVAE class.
        (You should really save the output of this)"""

        if not os.path.exists(self.fasta):
            raise FileNotFoundError(
                "Training Fasta file not found, required for determining bounds."
            )
        if not hasattr(self, "VAE"):
            raise AttributeError(
                "Trained VAE not found, run createVAE() first."
            )
        if not hasattr(self, "DCA"):
            raise AttributeError("DCA model not found, run createDCA() first.")

        # Load training, get size of box we will create for landscape
        sequences = one_hot_encode_fasta(self.fasta, self.alphabet)
        bounds = self.VAETrainer.getLandscapeBounds(
            self.VAE,
            sequences,
            batch_size=batch_size,
        )

        # build coordinate grids
        grid_points = np.linspace(bounds[0], bounds[1], resolution)
        grid = np.meshgrid(grid_points, grid_points)
        coordinates = np.vstack(np.array(grid).transpose())

        # Create dataloader, iterate through it and compute Hamiltonians
        coordinate_loader = self.VAETrainer.createDataLoader(
            torch.Tensor(coordinates), batch_size=batch_size, shuffle=False
        )

        coordinate_hamiltonians = np.zeros(len(coordinates))

        for idx, batch in enumerate(coordinate_loader):
            decoded_sequences = self.VAE.decoder(batch)
            softmax_sequences = decoded_sequences.reshape(
                len(batch), sequences.shape[1], sequences.shape[2]
            ).softmax(-1)
            argmax_sequences = softmax_sequences.argmax(-1).numpy()
            hamiltonians = return_Hamiltonian(
                argmax_sequences, self.DCA.couplings, self.DCA.localfields
            )

            coordinate_hamiltonians[
                idx * len(batch) : (idx * len(batch)) + len(batch)
            ] = hamiltonians

        # Save to new class variable
        self.LGL = np.hstack((coordinates, coordinate_hamiltonians[:, None]))

        # Save LGLVAE class with filename (if provided)
        self.save(output_fn)

    def save(self, save_fn: str) -> None:
        if save_fn == "":
            print("No output filename given, skipping save.")
        else:
            pickle.dump(self, open(save_fn, "wb"))

    @staticmethod
    def load(load_fn: str) -> "LGLVAE":
        """Load a pickled LGLVAE object from file."""
        if not os.path.exists(load_fn):
            raise FileNotFoundError(f"File {load_fn} does not exist.")
        with open(load_fn, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, LGLVAE):
            raise TypeError("Loaded object is not an instance of LGLVAE.")
        return obj

    def plot_landscape(
        self, axes: Axes, contour_levels: int = 1_000, colormap: str = "viridis"
    ) -> AxesImage:
        """Takes matplotlib axis and plots the landscape on it, returns image for colorbar."""
        if not hasattr(self, "LGL"):
            raise AttributeError("LGL not found, run createLGL() first.")

        x_coords = np.unique(self.LGL[:, 0])
        y_coords = np.unique(self.LGL[:, 1])

        # Create meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)

        # Reshape Z values to match grid
        Z = self.LGL[:, 2].reshape(len(y_coords), len(x_coords))

        image = axes.contourf(Y, X, Z, cmap=colormap, levels=contour_levels)

        return image

    def encode_sequences(
        self, fasta_fn: str, batch_size: int = 10_000
    ) -> np.ndarray:
        """Load sequences and encode as mu coordinates with encoder."""

        if not hasattr(self, "VAE"):
            raise AttributeError(
                "Trained VAE not found, run createVAE() first."
            )
        sequences = one_hot_encode_fasta(fasta_fn, self.alphabet)
        dataloader = self.VAETrainer.createDataLoader(
            sequences, batch_size=batch_size, shuffle=False
        )

        sequence_coordinates = np.zeros((len(sequences), 2))
        point = 0
        for idx, batch in enumerate(dataloader):
            hidden_out = self.VAE.encoder_base(batch)
            mu_coordinates = self.VAE.encoder_mu(hidden_out)
            next_point = point + len(batch)
            sequence_coordinates[point:next_point] = mu_coordinates.detach().numpy()
            point = next_point

        return sequence_coordinates

    def generate_sequences(
        self,
        coordinates: Union[np.array, torch.tensor],
        argmax_sequence: bool = True,
    ) -> List[SeqRecord]:
        """Takes numpy/torch arrays as input, and gives sequence strings as output.
        Gives either the maximum probability sequence or a sampled sequence."""
        if not hasattr(self, "VAE"):
            raise AttributeError(
                "Trained VAE not found, run createVAE() first."
            )
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates[None]

        decoded_distributions = self.VAE.decoder(coordinates)
        softmax_distribution = decoded_distributions.reshape(
            decoded_distributions.shape[0],
            decoded_distributions.shape[1] // self.VAE.num_aa,
            self.VAE.num_aa,
        ).softmax(-1)

        if argmax_sequence:
            numeric_sequences = softmax_distribution.argmax(-1).detach().numpy()
        else:
            numeric_sequences = np.array(
                [
                    [
                        np.random.choice(self.VAE.num_aa, p=site_probs)
                        for site_probs in sequence_probs
                    ]
                    for sequence_probs in softmax_distribution.detach().numpy()
                ]
            )

        reverse_alphabet = {v: k for k, v in list(self.alphabet.items())[:21]}
        string_sequences = [
            "".join([reverse_alphabet[int(number)] for number in sequence])
            for sequence in numeric_sequences
        ]

        record_sequences = [
            SeqRecord(
                Seq(seq),
                id="",
                description=", ".join([str(n.item()) for n in coord]),
            )
            for coord, seq in zip(coordinates, string_sequences)
        ]

        return record_sequences
