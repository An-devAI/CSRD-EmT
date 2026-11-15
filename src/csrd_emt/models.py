# CBCR License 1.0
#
# Copyright 2022 Centre for Brain Computing Research (CBCR)
#
# Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior
# written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTER-
# RUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

TEMPERATURE = 0.1


class ChebyshevGraphConv(nn.Module):

    def __init__(self, order: int, in_features: int, out_features: int) -> None:
        super().__init__()
        if order < 2:
            raise ValueError("Chebyshev order must be at least 2")
        self.order = order
        self.weight = nn.Parameter(torch.empty(order, 1, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(1, 1, out_features))
        nn.init.normal_(self.weight, 0, 0.1)
        nn.init.normal_(self.bias, 0, 0.1)

    @staticmethod
    def laplacian(adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=1)
        degree_norm = 1.0 / (torch.sqrt(degree) + 1e-05)
        degree_matrix = torch.diag(degree_norm)
        return -(degree_matrix @ adjacency @ degree_matrix)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]):
        x, adjacency = inputs
        laplacian = self.laplacian(adjacency)
        terms = [x, torch.matmul(laplacian, x)]
        for _ in range(2, self.order):
            terms.append(2 * torch.matmul(laplacian, terms[-1]) - terms[-2])
        stacked = torch.stack(terms, dim=0)
        output = torch.matmul(stacked, self.weight).sum(dim=0)
        return (F.relu(output + self.bias), adjacency)


class GraphEncoder(nn.Module):

    def __init__(
        self,
        layers: int,
        nodes: int,
        in_features: int,
        hidden_features: int,
        chebyshev_order: int,
    ) -> None:
        super().__init__()
        modules = []
        for index in range(layers):
            modules.append(
                ChebyshevGraphConv(
                    chebyshev_order,
                    in_features if index == 0 else hidden_features,
                    hidden_features,
                )
            )
        self.layers = nn.ModuleList(modules)
        self.tokenizer = nn.Linear(nodes * hidden_features, hidden_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        state: tuple[torch.Tensor, torch.Tensor] = (x, adjacency)
        for layer in self.layers:
            state = layer(state)
        graph, _ = state
        return self.tokenizer(graph.flatten(start_dim=1))


@dataclass(frozen=True)
class ModelConfig:
    num_channels: int = 62
    num_features: int = 7
    sequence_length: int = 36
    num_classes: int = 3
    graph_layers: tuple[int, int] = (1, 2)
    transformer_layers: int = 8
    num_adjacencies: int = 2
    hidden_graph: int = 32
    chebyshev_order: int = 4
    attention_heads: int = 16
    attention_head_dim: int = 32
    dropout: float = 0.25
    attention_dropout_scale: float = 0.25
    chebyshev_self_loops: bool = False


class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_graph
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = TemporalAttention(
            dim=hidden,
            heads=config.attention_heads,
            head_dim=config.attention_head_dim,
            dropout=config.dropout,
            alpha=config.attention_dropout_scale,
        )
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.feedforward = FeedForward(
            hidden, config.attention_head_dim, config.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class TemporalAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        dropout: float,
        alpha: float,
        anchor: int = 3,
    ) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.scale = head_dim ** (-0.5)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.temporal_filter = nn.Sequential(
            nn.Dropout(alpha * dropout),
            weight_norm(
                nn.Conv2d(
                    heads,
                    heads,
                    kernel_size=(anchor, 1),
                    stride=1,
                    padding=((anchor - 1) // 2, 0),
                )
            ),
        )
        self.to_output = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            item.reshape(item.shape[0], item.shape[1], self.heads, -1).permute(
                0, 2, 1, 3
            )
            for item in (q, k, v)
        )
        attention = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1
        )
        output = self.temporal_filter(torch.matmul(attention, v))
        return self.to_output(
            output.permute(0, 2, 1, 3).reshape(output.shape[0], output.shape[2], -1)
        )


class AdaptedEmT(nn.Module):
    """EmT classification backbone with configurable EEG steps, bands and classes."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.num_adjacencies < 2:
            raise ValueError("The RMPG block requires at least two learned adjacencies")
        self.graph_encoder_shallow = GraphEncoder(
            cfg.graph_layers[0],
            cfg.num_channels,
            cfg.num_features,
            cfg.hidden_graph,
            cfg.chebyshev_order,
        )
        self.graph_encoder_deep = GraphEncoder(
            cfg.graph_layers[1],
            cfg.num_channels,
            cfg.num_features,
            cfg.hidden_graph,
            cfg.chebyshev_order,
        )
        self.adjacencies = nn.Parameter(
            torch.empty(cfg.num_adjacencies, cfg.num_channels, cfg.num_channels)
        )
        nn.init.xavier_uniform_(self.adjacencies)
        self.residual_projection = nn.Linear(
            cfg.num_channels * cfg.num_features, cfg.hidden_graph, bias=False
        )
        self.transformer = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.transformer_layers)]
        )
        self.classifier = nn.Linear(cfg.hidden_graph, cfg.num_classes)

    def learned_adjacencies(self) -> torch.Tensor:
        symmetric = F.relu(self.adjacencies + self.adjacencies.transpose(-1, -2))
        if not self.config.chebyshev_self_loops:
            return symmetric
        identity = torch.eye(
            self.config.num_channels, device=symmetric.device, dtype=symmetric.dtype
        )
        return symmetric + identity

    def encode_stages(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return branch and temporal representations without changing inference."""
        cfg = self.config
        expected = (cfg.sequence_length, cfg.num_channels, cfg.num_features)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected input (B,{expected[0]},{expected[1]},{expected[2]}), got {tuple(x.shape)}"
            )
        batch, steps = x.shape[:2]
        graphs = x.reshape(batch * steps, cfg.num_channels, cfg.num_features)
        adjacency = self.learned_adjacencies()
        residual_flat = self.residual_projection(graphs.flatten(start_dim=1))
        shallow_flat = self.graph_encoder_shallow(graphs, adjacency[0])
        deep_flat = self.graph_encoder_deep(graphs, adjacency[1])
        residual = residual_flat.reshape(batch, steps, -1)
        shallow = shallow_flat.reshape(batch, steps, -1)
        deep = deep_flat.reshape(batch, steps, -1)
        tokens = torch.stack((residual, shallow, deep), dim=1).mean(dim=1)
        contextual = self.transformer(tokens)
        pooled = contextual.mean(dim=1)
        return {
            "residual": residual,
            "shallow": shallow,
            "deep": deep,
            "contextual": contextual,
            "pooled": pooled,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_stages(x)["pooled"])


class ResearchEmT(AdaptedEmT):

    def __init__(self, kind="erm"):
        if kind != "erm":
            raise ValueError(
                "Notebook này dùng backbone EmT gốc, không dùng nhánh vùng."
            )
        super().__init__(ModelConfig(num_features=5, sequence_length=36))

    def features(self, x):
        b, t = x.shape[:2]
        g = x.reshape(b * t, 62, 5)
        a = self.learned_adjacencies()
        tokens = (
            self.residual_projection(g.flatten(1))
            + self.graph_encoder_shallow(g, a[0])
            + self.graph_encoder_deep(g, a[1])
        ) / 3
        tokens = tokens.reshape(b, t, 32)
        result = {}
        pooled = self.transformer(tokens).mean(1)
        z = F.normalize(pooled, dim=-1, eps=1e-06)
        result.update(
            pooled=pooled,
            normalized=z,
            logits=F.linear(z, F.normalize(self.classifier.weight, dim=-1))
            / TEMPERATURE,
        )
        return result

    def forward(self, x, return_features=False):
        out = self.features(x)
        return out if return_features else out["logits"]


class Reverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return (-ctx.scale * g, None)


class MetricEmT(ResearchEmT):

    def __init__(self, subjects, seed=0):
        super().__init__("erm")
        self.subjects = list(sorted(subjects))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 983)
            self.projector = nn.Sequential(
                nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
            )
            self.proxies = nn.Parameter(torch.randn(3, 5, 32) * 0.02)
            self.discriminator = nn.Sequential(
                nn.Linear(32, 64),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.01),
                nn.Linear(32, len(self.subjects)),
            )

    def metric(self, h, y):
        z = self.projector(h)
        weights = torch.softmax(torch.einsum("bd,ckd->bck", z, self.proxies), dim=-1)
        distance = (
            weights
            * (z[:, None, None, :] - self.proxies[None, :, :, :]).square().sum(-1)
        ).sum(-1)
        return semantic_loss(distance, y)


def semantic_loss(d, y):
    assert d.shape == (len(y), 3) and bool(((y >= 0) & (y <= 2)).all())
    own = d.gather(1, y[:, None]).squeeze(1)
    neutral = y == 1
    first = torch.where(
        neutral, F.softplus(d[:, 1] - d[:, 0]), F.softplus(own - d[:, 1])
    )
    opposite = d.gather(1, (2 - y)[:, None]).squeeze(1)
    second = torch.where(
        neutral, F.softplus(d[:, 1] - d[:, 2]), F.softplus(d[:, 1] - opposite)
    )
    return (own + first + second).mean()


class Student(ResearchEmT):

    def __init__(self, subjects, seed, weights):
        super().__init__("erm")
        self.subjects = sorted(subjects)
        self.load_state_dict(
            {k: v for k, v in weights.items() if k in self.state_dict()}, strict=True
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 983)
            self.trunk = nn.Sequential(
                nn.Linear(32, 64),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.01),
            )
            self.heads = nn.ModuleList([nn.Linear(32, len(subjects)) for _ in range(3)])
