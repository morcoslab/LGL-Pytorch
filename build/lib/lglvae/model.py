from typing import Dict
import torch.nn.functional as F
import torch


class VAE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_u: int,
        num_aa: int = 21,
        latent_dim: int = 2,
    ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_u = hidden_u
        self.input_dim = input_dim
        self.num_aa = num_aa

        # encoder module#
        self.encoder_base = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_u), torch.nn.ReLU()
        )
        self.encoder_mu = torch.nn.Linear(hidden_u, latent_dim)
        self.encoder_logvar = torch.nn.Linear(hidden_u, latent_dim)

        # decoder module
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_u),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_u, input_dim),
        )

    def forward(self, data: torch.Tensor) -> Dict:
        """Encode, sample, decode. Returns:
        'mu', 'logsigma', 'zed', 'recon_data'
        'zed' is the sample after reparameterizing mu and logsigma.
        'recon_data' is the post-softmax output"""
        encoder_projection = self.encoder_base(data)
        mu = self.encoder_mu(encoder_projection)
        logvar = self.encoder_logvar(encoder_projection)

        # reparameterization: z = mu + exp(0.5 * logvar) * eps
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(mu)
        zed = mu + eps * std

        # decode
        recon_logits = self.decoder(zed)

        # reshape, softmax on aa dimension (like axis=1 in TF code)
        softmax_data = recon_logits.reshape(
            data.shape[0], data.shape[1] // self.num_aa, self.num_aa
        ).softmax(-1)

        return {"mu": mu, "logvar": logvar, "zed": zed, "recon_data": softmax_data}

    def kld(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kld = 1 + logvar - mu.pow(2) - logvar.exp()
        kld = -0.5 * kld.sum(dim=1)  # (batch,)
        return kld

    def compute_elbo(self, data: torch.Tensor) -> Dict:
        output = self.forward(data)

        # flatten decoder output back to (batch, D)
        recon_flat = output["recon_data"].flatten(1)  # (batch, D)
        D = data.size(1)

        # BCE per element, then mean over features, then * D (to get sum over features)
        bce_elem = F.binary_cross_entropy(recon_flat, data, reduction="none")
        recon_loss_per_sample = bce_elem.mean(dim=1) * D  # (batch,)

        kl_loss = self.kld(output["mu"], output["logvar"])  # (batch,)

        # TF: vae_loss = mean(reconstruction_loss + kl_loss)
        loss = (recon_loss_per_sample + kl_loss).mean()

        return {
            "loss": loss,
            "reconstruction": recon_loss_per_sample.mean(),
            "kld": kl_loss.mean(),
        }