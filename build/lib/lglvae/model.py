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
        l2_reg: float = 1e-4,
    ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_u = hidden_u
        self.input_dim = input_dim
        self.num_aa = num_aa
        self.l2_reg = float(l2_reg)

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

        # Match TF Dense defaults more closely (Glorot/Xavier uniform, bias zeros)
        self.apply(self._init_tf_dense_like)

    @staticmethod
    def _init_tf_dense_like(m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def _tf_style_l2_penalty(self) -> torch.Tensor:
        """
        Emulate TF kernel_regularizer=L2(reg) used only on the hidden Dense layers:
          - encoder hidden: Linear(input_dim -> hidden_u)
          - decoder hidden: Linear(latent_dim -> hidden_u)
        Keras L2(reg) corresponds to: reg * sum(w^2).
        """
        if self.l2_reg <= 0.0:
            return torch.zeros((), device=self.encoder_base[0].weight.device)

        enc_w = self.encoder_base[0].weight  # hidden encoder kernel
        dec_w = self.decoder[0].weight       # hidden decoder kernel
        return self.l2_reg * (enc_w.pow(2).sum() + dec_w.pow(2).sum())


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
        kld = logvar.add(1) - mu.pow(2) - logvar.exp()
        kld = -0.5 * kld.sum(dim=1)  # (batch,)
        return kld

    def compute_elbo(self, data: torch.Tensor) -> Dict:
        output = self.forward(data)

        # flatten decoder output back to (batch, D)
        recon_flat = output["recon_data"].flatten(1)  # (batch, D)

        recon_loss = F.binary_cross_entropy(recon_flat,
            data,reduction="none").sum(dim=1)
        kl_loss = self.kld(output["mu"], output["logvar"])  # (batch,)

        # TF: vae_loss = mean(reconstruction_loss + kl_loss)
        base = (recon_loss + kl_loss).mean()
        reg = self._tf_style_l2_penalty()
        loss = base + reg

        return {
            "loss": loss,
            "reconstruction": recon_loss.mean(),
            "kld": kl_loss.mean(),
        }