from typing import Dict

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
        # reconstruction loss function
        self.recon_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, data: torch.Tensor) -> Dict:
        """Encode, sample, decode. Returns:
        'mu', 'logsigma', 'zed', 'recon_data'
        'zed' is the sample after reparameterizing mu and logsigma.
        'recon_data' is the post-softmax output"""
        encoder_projection = self.encoder_base(data)
        mu = self.encoder_mu(encoder_projection)
        logsigma = self.encoder_logvar(encoder_projection)
        # sample
        zed = torch.randn_like(mu) * logsigma.exp() + mu
        # decode
        recon_data = self.decoder(zed)
        # reshape, softmax on aa dimension
        softmax_data = recon_data.reshape(
            data.shape[0], data.shape[1] // self.num_aa, self.num_aa
        ).softmax(-1)

        return {"mu": mu, "logsigma": logsigma, "zed": zed, "recon_data": softmax_data}

    def kld(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        kld = logsigma.mul(2).add(1) - mu.pow(2) - logsigma.exp().pow(2)
        kld.mul_(-0.5)
        return kld

    def compute_elbo(self, data: torch.Tensor) -> Dict:
        output = self.forward(data)
        recon_loss = self.recon_loss(data, output["recon_data"].flatten(1))
        kld = self.kld(output["mu"], output["logsigma"]).mean()
        loss = recon_loss + kld
        return {"loss": loss, "reconstruction": recon_loss, "kld": kld}
