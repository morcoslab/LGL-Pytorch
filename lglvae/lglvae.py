import os
import pickle
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
    def __init__(
            self,
            fasta_fn: str,
            alphabet: dict[str, int] = seq_code,
            _lr: float = 1e-3) -> None:
        """Use createVAE() to train the VAE model using the fasta_fn and expected alphabet dictionary.
        Use createDCA() to create a DCA model using the fasta_fn.
        Use createLGL() to create the landscape grid data.
        Use plotSequences(fasta_fn) to create coordinates with the VAE.
        Use generateSequences(coordinates) to create sequences with the VAE.
        """
        self.fasta = fasta_fn
        self.alphabet = alphabet
        self.VAETrainer = Trainer(learning_rate=_lr)

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

        # Shape is (batch, num_aa, seq_len) - TensorFlow convention
        num_aa = one_hot_data.shape[1]
        seq_len = one_hot_data.shape[2]
        input_dim = num_aa * seq_len  # size when flattened
        hidden_units = 3 * seq_len  # 3*sequence length

        model = VAE(input_dim=input_dim,
                    hidden_u=hidden_units,
                    num_aa=num_aa,
                    latent_dim=2,
                    l2_reg=self.VAETrainer.regularization)

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

        # Ensure model is in eval mode and disable gradients for inference
        self.VAE.eval()
        with torch.no_grad():
            for idx, batch in enumerate(coordinate_loader):
                decoded_sequences = self.VAE.decoder(batch)
                # Shape: (batch, num_aa, seq_len) - TensorFlow convention
                # sequences.shape[1] = num_aa, sequences.shape[2] = seq_len
                softmax_sequences = decoded_sequences.reshape(
                    len(batch), sequences.shape[1], sequences.shape[2]
                ).softmax(dim=1)  # softmax over amino acids (dim=1)
                argmax_sequences = softmax_sequences.argmax(dim=1).numpy()  # argmax over amino acids

                # Validate sequences are in valid range before passing to DCA
                num_aa = sequences.shape[1]
                if argmax_sequences.max() >= num_aa:
                    raise ValueError(
                        f"Invalid amino acid index {argmax_sequences.max()} >= {num_aa}"
                    )
                if argmax_sequences.min() < 0:
                    raise ValueError(
                        f"Negative amino acid index: {argmax_sequences.min()}"
                    )

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
        coordinates: np.ndarray | torch.Tensor,
        argmax_sequence: bool = True,
    ) -> list[SeqRecord]:
        """Takes numpy/torch arrays as input, and gives sequence strings as output.
        Gives either the maximum probability sequence or a sampled sequence."""
        if not hasattr(self, "VAE"):
            raise AttributeError(
                "Trained VAE not found, run createVAE() first."
            )
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=torch.float32)
        elif coordinates.dtype != torch.float32:
            coordinates = coordinates.float()
        if coordinates.ndim == 1:
            coordinates = coordinates[None]

        decoded_distributions = self.VAE.decoder(coordinates)

        # Reshape to (batch, num_aa, seq_len) - TensorFlow convention
        seq_len = decoded_distributions.shape[1] // self.VAE.num_aa
        softmax_distribution = decoded_distributions.reshape(
            decoded_distributions.shape[0],
            self.VAE.num_aa,
            seq_len,
        ).softmax(dim=1)  # softmax over amino acids (dim=1)


        if argmax_sequence:
            # argmax over amino acids (dim=1), result shape: (batch, seq_len)
            numeric_sequences = softmax_distribution.argmax(dim=1).detach().numpy()
        else:
            # For sampling, transpose to (batch, seq_len, num_aa) for easier iteration
            # Each position needs amino acid probabilities
            transposed = softmax_distribution.permute(0, 2, 1).detach().numpy()
            numeric_sequences = np.array(
                [
                    [
                        np.random.choice(self.VAE.num_aa, p=site_probs)
                        for site_probs in sequence_probs  # iterates over seq_len positions
                    ]
                    for sequence_probs in transposed  # iterates over batch
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
