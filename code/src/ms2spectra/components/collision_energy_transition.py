import torch as th
import torch.nn as nn


class CELocalTransitionPrior(nn.Module):
    """
    CE-conditioned local fragmentation transition prior.

    Scores each DAG transition edge parent -> child under collision energy.
    Residual + zero-init: initial behavior equals the previous model.
    """

    def __init__(
        self,
        hidden_size: int,
        edge_size: int,
        ce_size: int,
        mlp_hidden_size: int = 256,
        dropout: float = 0.1,
        delta_scale: float = 0.5,
        zero_init: bool = True,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)

        in_size = hidden_size * 3 + edge_size + ce_size

        self.net = nn.Sequential(
            nn.Linear(in_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 1),
        )

        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, parent_h, child_h, edge_h, ce_edge_h):
        if edge_h is None:
            edge_h = parent_h.new_zeros(parent_h.shape[0], 0)

        if parent_h.shape[0] != child_h.shape[0]:
            raise RuntimeError(
                f"parent/child mismatch: {parent_h.shape} vs {child_h.shape}"
            )
        if ce_edge_h.shape[0] != parent_h.shape[0]:
            raise RuntimeError(
                f"CE edge mismatch: {ce_edge_h.shape} vs {parent_h.shape}"
            )

        x = th.cat(
            [
                parent_h,
                child_h,
                parent_h - child_h,
                edge_h,
                ce_edge_h,
            ],
            dim=-1,
        )

        delta = self.net(x).squeeze(-1)
        delta = th.tanh(delta) * self.delta_scale
        return delta
class CEHChannelTransitionPrior(nn.Module):
    """
    CE-conditioned H-channel local transition prior.

    Instead of producing one scalar edge delta, this module produces
    a residual vector over H-transfer channels for each parent -> child edge.

    Output shape:
        [num_edges, num_h_channels]

    This is designed for FraGNNet formula mode:
        frag_joint_logits: [num_nodes, num_h_channels]
    """

    def __init__(
        self,
        hidden_size: int,
        edge_size: int,
        ce_size: int,
        num_h_channels: int,
        mlp_hidden_size: int = 256,
        dropout: float = 0.1,
        delta_scale: float = 0.05,
        zero_init: bool = True,
    ):
        super().__init__()
        self.num_h_channels = int(num_h_channels)
        self.delta_scale = float(delta_scale)

        in_size = hidden_size * 3 + edge_size + ce_size

        self.net = nn.Sequential(
            nn.Linear(in_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, self.num_h_channels),
        )

        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, parent_h, child_h, edge_h, ce_edge_h):
        if edge_h is None:
            edge_h = parent_h.new_zeros(parent_h.shape[0], 0)

        if parent_h.shape[0] != child_h.shape[0]:
            raise RuntimeError(
                f"parent/child mismatch: {parent_h.shape} vs {child_h.shape}"
            )

        if ce_edge_h.shape[0] != parent_h.shape[0]:
            raise RuntimeError(
                f"CE edge mismatch: {ce_edge_h.shape} vs {parent_h.shape}"
            )

        x = th.cat(
            [
                parent_h,
                child_h,
                parent_h - child_h,
                edge_h,
                ce_edge_h,
            ],
            dim=-1,
        )

        delta = self.net(x)
        delta = th.tanh(delta) * self.delta_scale
        return delta
