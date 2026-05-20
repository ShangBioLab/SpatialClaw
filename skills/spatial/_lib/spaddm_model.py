"""Minimal SpaDDM model adapter used by the SPATIALCLAW multi-omics skill.

This implementation is adapted from the local SpaDDM repository bundled in the
workspace. The original project structures the model across ``Multi_Diffusion``
modules; here we keep the core two-modality network in one internal file so the
skill can depend on a stable, package-local backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv


@dataclass
class SpaDDMGraph:
    """Lightweight graph container for SpaDDM that is easy to move onto GPU."""

    edge_index: torch.Tensor
    feat: torch.Tensor
    num_nodes: int

    def to(self, device: torch.device) -> "SpaDDMGraph":
        return SpaDDMGraph(
            edge_index=self.edge_index.to(device),
            feat=self.feat.to(device),
            num_nodes=self.num_nodes,
        )


def _get_beta_schedule(
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
) -> torch.Tensor:
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start,
            beta_end,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps,
            1,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)

    return torch.from_numpy(betas)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    out = torch.gather(values, index=timesteps, dim=0).float()
    return out.view([timesteps.shape[0]] + [1] * (len(x_shape) - 1))


class Residual(nn.Module):
    def __init__(self, fnc: nn.Module):
        super().__init__()
        self.fnc = fnc

    def forward(self, x, *args, **kwargs):
        return self.fnc(x, *args, **kwargs) + x


def create_activation(name: str | None) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "prelu":
        return nn.PReLU()
    if name == "elu":
        return nn.ELU()
    if name is None:
        return nn.Identity()
    raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name: str):
    if name == "layernorm":
        return nn.LayerNorm
    if name == "batchnorm":
        return nn.BatchNorm1d
    return nn.Identity


class MlpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        norm: str = "layernorm",
        activation: str = "prelu",
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_mlp = Residual(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                create_norm(norm)(hidden_dim),
                create_activation(activation),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = create_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        return self.act(x)


class DenoisingUNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        out_dim: int,
        num_layers: int,
        nhead: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        norm: str,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.mlp_in_t = MlpBlock(
            in_dim=in_dim,
            hidden_dim=num_hidden * 2,
            out_dim=num_hidden,
            norm=norm,
            activation=activation,
        )
        self.mlp_middle = MlpBlock(
            num_hidden,
            num_hidden,
            num_hidden,
            norm=norm,
            activation=activation,
        )
        self.mlp_out = MlpBlock(
            num_hidden,
            out_dim,
            out_dim,
            norm=norm,
            activation=activation,
        )

        self.down_layers.append(
            GATConv(
                num_hidden,
                num_hidden // nhead,
                heads=nhead,
                concat=True,
                dropout=attn_drop,
                negative_slope=negative_slope,
                add_self_loops=False,
            )
        )
        self.up_layers.append(
            GATConv(
                num_hidden,
                num_hidden,
                heads=1,
                concat=True,
                dropout=attn_drop,
                negative_slope=negative_slope,
                add_self_loops=False,
            )
        )

        for _ in range(1, num_layers):
            self.down_layers.append(
                GATConv(
                    num_hidden,
                    num_hidden // nhead,
                    heads=nhead,
                    concat=True,
                    dropout=attn_drop,
                    negative_slope=negative_slope,
                    add_self_loops=False,
                )
            )
            self.up_layers.append(
                GATConv(
                    num_hidden,
                    num_hidden // nhead,
                    heads=nhead,
                    concat=True,
                    dropout=attn_drop,
                    negative_slope=negative_slope,
                    add_self_loops=False,
                )
            )
        self.up_layers = self.up_layers[::-1]

    def forward(self, graph, x_t: torch.Tensor, time_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_t = self.mlp_in_t(x_t)
        down_hidden: list[torch.Tensor] = []
        edge_index = graph.edge_index

        for layer in self.down_layers:
            h_t = h_t + time_embed
            h_t = F.dropout(h_t, p=self.feat_drop, training=self.training)
            h_t = layer(h_t, edge_index)
            down_hidden.append(h_t)

        h_t = self.mlp_middle(h_t)
        up_hidden: list[torch.Tensor] = []
        for i, layer in enumerate(self.up_layers):
            h_t = h_t + down_hidden[self.num_layers - i - 1]
            h_t = h_t + time_embed
            h_t = F.dropout(h_t, p=self.feat_drop, training=self.training)
            h_t = layer(h_t, edge_index)
            up_hidden.append(h_t)

        out = self.mlp_out(h_t)
        return out, torch.cat(up_hidden, dim=-1)


class AttentionLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int):
        super().__init__()
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.cat(
            [
                torch.unsqueeze(torch.squeeze(emb1), dim=1),
                torch.unsqueeze(torch.squeeze(emb2), dim=1),
            ],
            dim=1,
        )
        v = torch.tanh(torch.matmul(emb, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(torch.squeeze(vu) + 1e-6, dim=-1)
        combined = torch.matmul(torch.transpose(emb, 1, 2), torch.unsqueeze(alpha, -1))
        return torch.squeeze(combined), alpha


class CrossDecoder(nn.Module):
    def __init__(self, in_feat: int, out_feat: int):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        decoded = torch.mm(feat, self.weight)
        if getattr(adj, "is_sparse", False):
            return torch.sparse.mm(adj, decoded)
        return torch.mm(adj, decoded)


def graph_to_sparse_adj(graph) -> torch.Tensor:
    """Build a torch sparse COO adjacency matrix from the stored edge index."""
    indices = graph.edge_index.long()
    num_nodes = graph.num_nodes
    values = torch.ones(indices.shape[1], device=indices.device, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def _loss_fn(x: torch.Tensor, y: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return (1 - (x * y).sum(dim=-1)).pow_(alpha).mean()


class DDM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        nhead: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        norm: str,
        alpha_l: float = 2,
        beta_schedule: str = "linear",
        beta_1: float = 0.0001,
        beta_T: float = 0.02,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.timesteps = timesteps
        beta = _get_beta_schedule(beta_schedule, beta_1, beta_T, timesteps)
        self.register_buffer("betas", beta)

        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

        self.alpha_l = alpha_l
        self.net = DenoisingUNet(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=in_dim,
            num_layers=num_layers,
            nhead=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=0.2,
            norm=norm,
        )
        self.time_embedding = nn.Embedding(timesteps, num_hidden)
        self.norm_x = nn.LayerNorm(in_dim, elementwise_affine=False)

    def sample_q(self, timesteps: torch.Tensor, x: torch.Tensor, graph) -> tuple[torch.Tensor, torch.Tensor, object]:
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))

        mean = x.mean(dim=0)
        std = x.std(dim=0)
        noise = torch.randn_like(x, device=x.device)
        noise = noise * std + mean
        noise = self.norm_x(noise)
        noise = torch.sign(x) * torch.abs(noise)

        x_t = (
            _extract(self.sqrt_alphas_bar, timesteps, x.shape) * x
            + _extract(self.sqrt_one_minus_alphas_bar, timesteps, x.shape) * noise
        )
        return x_t, self.time_embedding(timesteps), graph

    def node_denoising(self, x: torch.Tensor, x_t: torch.Tensor, time_embed: torch.Tensor, graph) -> torch.Tensor:
        out, _ = self.net(graph, x_t=x_t, time_embed=time_embed)
        return _loss_fn(out, x, self.alpha_l)

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(self.timesteps, size=(x.shape[0],), device=x.device)
        x_t, time_embed, graph = self.sample_q(timesteps, x, graph)
        return self.node_denoising(x, x_t, time_embed, graph)

    def embed(self, graph, x: torch.Tensor, timestep: int) -> torch.Tensor:
        t = torch.full((1,), timestep, device=x.device)
        x_t, time_embed, graph = self.sample_q(t, x, graph)
        _, hidden = self.net(graph, x_t=x_t, time_embed=time_embed)
        return hidden


class MultiOmicsDDM(nn.Module):
    def __init__(
        self,
        in_dim_omics_1: int,
        in_dim_omics_2: int,
        num_hidden_omics_1: int,
        num_hidden_omics_2: int,
        num_layers: int = 1,
        nhead: int = 2,
        activation: str = "prelu",
        feat_drop: float = 0.2,
        attn_drop: float = 0.2,
        norm: str = "layernorm",
        alpha_l: float = 2,
        beta_schedule: str = "linear",
        beta_1: float = 0.0001,
        beta_T: float = 0.02,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.ddm_omics_1 = DDM(
            in_dim=in_dim_omics_1,
            num_hidden=num_hidden_omics_1,
            num_layers=num_layers,
            nhead=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            norm=norm,
            alpha_l=alpha_l,
            beta_schedule=beta_schedule,
            beta_1=beta_1,
            beta_T=beta_T,
            timesteps=timesteps,
        )
        self.ddm_omics_2 = DDM(
            in_dim=in_dim_omics_2,
            num_hidden=num_hidden_omics_2,
            num_layers=num_layers,
            nhead=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            norm=norm,
            alpha_l=alpha_l,
            beta_schedule=beta_schedule,
            beta_1=beta_1,
            beta_T=beta_T,
            timesteps=timesteps,
        )
        self.atten_cross = AttentionLayer(num_hidden_omics_1, num_hidden_omics_2)
        self.dec_cross_omics_1 = CrossDecoder(num_hidden_omics_1, in_dim_omics_1)
        self.dec_cross_omics_2 = CrossDecoder(num_hidden_omics_2, in_dim_omics_2)

    def forward(self, graph_omics_1, graph_omics_2) -> dict[str, torch.Tensor]:
        loss_rec_1 = self.ddm_omics_1(graph_omics_1, graph_omics_1.feat)
        loss_rec_2 = self.ddm_omics_2(graph_omics_2, graph_omics_2.feat)
        emb_omics_1 = self.ddm_omics_1.embed(graph_omics_1, graph_omics_1.feat, 100)
        emb_omics_2 = self.ddm_omics_2.embed(graph_omics_2, graph_omics_2.feat, 100)
        emb_combined, alpha = self.atten_cross(emb_omics_1, emb_omics_2)
        adj_omics_1 = graph_to_sparse_adj(graph_omics_1)
        adj_omics_2 = graph_to_sparse_adj(graph_omics_2)
        comb_recon_omics_1 = self.dec_cross_omics_1(emb_combined, adj_omics_1)
        comb_recon_omics_2 = self.dec_cross_omics_2(emb_combined, adj_omics_2)
        return {
            "emb_omics_1": emb_omics_1,
            "emb_omics_2": emb_omics_2,
            "emb_combined": emb_combined,
            "comb_recon_omics_1": comb_recon_omics_1,
            "comb_recon_omics_2": comb_recon_omics_2,
            "loss_rec_omics_1": loss_rec_1,
            "loss_rec_omics_2": loss_rec_2,
            "alpha": alpha,
        }


class SpaDDMTrainer:
    def __init__(
        self,
        graph_omics_1,
        graph_omics_2,
        *,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-7,
        epochs: int = 600,
        latent_dim: int = 64,
        weight_factors: tuple[float, float, float, float] = (0.5, 0.5, 0.1, 0.5),
    ):
        self.graph_omics_1 = graph_omics_1.to(device)
        self.graph_omics_2 = graph_omics_2.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.weight_factors = weight_factors

        self.model = MultiOmicsDDM(
            in_dim_omics_1=self.graph_omics_1.feat.shape[1],
            in_dim_omics_2=self.graph_omics_2.feat.shape[1],
            num_hidden_omics_1=latent_dim,
            num_hidden_omics_2=latent_dim,
        ).to(device)

    def train(self) -> dict[str, np.ndarray]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for _ in range(self.epochs):
            self.model.train()
            results = self.model(self.graph_omics_1, self.graph_omics_2)
            loss = (
                self.weight_factors[0] * results["loss_rec_omics_1"]
                + self.weight_factors[1] * results["loss_rec_omics_2"]
                + self.weight_factors[2]
                * F.mse_loss(self.graph_omics_1.feat, results["comb_recon_omics_1"])
                + self.weight_factors[3]
                * F.mse_loss(self.graph_omics_2.feat, results["comb_recon_omics_2"])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            results = self.model(self.graph_omics_1, self.graph_omics_2)

        return {
            "emb_omics_1": F.normalize(results["emb_omics_1"], p=2, dim=1).cpu().numpy(),
            "emb_omics_2": F.normalize(results["emb_omics_2"], p=2, dim=1).cpu().numpy(),
            "SpatialDDM": F.normalize(results["emb_combined"], p=2, dim=1).cpu().numpy(),
            "rec_omics_1": F.normalize(results["comb_recon_omics_1"], p=2, dim=1).cpu().numpy(),
            "rec_omics_2": F.normalize(results["comb_recon_omics_2"], p=2, dim=1).cpu().numpy(),
            "alpha": results["alpha"].cpu().numpy(),
        }
