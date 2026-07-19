import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CEFragmentFiLM(nn.Module):
    """
    Collision-energy-conditioned FiLM gate for fragment node embeddings.

    h'_v = h_v * (1 + gamma(CE, depth_v)) + beta(CE, depth_v)

    设计原则：
    1. 初始接近 identity，不破坏 baseline；
    2. 在 fragment node 层调制，而不是只在最后 MLP concat CE；
    3. 可以在论文图里画成 Energy-aware Fragment Activation。
    """

    def __init__(
        self,
        hidden_size: int,
        ce_size: int,
        gate_hidden_size: int = 128,
        dropout: float = 0.1,
        gamma_scale: float = 0.2,
        use_depth: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ce_size = ce_size
        self.use_depth = use_depth
        self.gamma_scale = gamma_scale

        in_size = ce_size + (1 if use_depth else 0)

        self.net = nn.Sequential(
            nn.Linear(in_size, gate_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, gate_hidden_size),
            nn.SiLU(),
            nn.Linear(gate_hidden_size, hidden_size * 2),
        )

        # 关键：最后一层零初始化，让模型初始等价于原 baseline
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, frag_h, ce_node_embed, depth_value=None):
        if self.use_depth:
            if depth_value is None:
                depth_value = th.zeros(
                    frag_h.shape[0],
                    1,
                    device=frag_h.device,
                    dtype=frag_h.dtype,
                )
            gate_in = th.cat([ce_node_embed, depth_value], dim=-1)
        else:
            gate_in = ce_node_embed
        expected_in = self.net[0].in_features
        if gate_in.shape[-1] != expected_in:
            raise RuntimeError(
                f"CEFragmentFiLM gate input dim mismatch: "
                f"got {gate_in.shape[-1]}, expected {expected_in}. "
                f"frag_h={tuple(frag_h.shape)}, "
                f"ce_node_embed={tuple(ce_node_embed.shape)}, "
                f"depth={None if depth_value is None else tuple(depth_value.shape)}"
            )
        gamma, beta = self.net(gate_in).chunk(2, dim=-1)
        gamma = th.tanh(gamma) * self.gamma_scale

        return frag_h * (1.0 + gamma) + beta