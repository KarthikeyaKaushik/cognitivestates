"""Legacy module: BaseRNN plus backward-compatible re-exports.

The projection/decoding functions moved to src/projections.py; they are
re-exported here so older scripts (scripts/run_rnn.py etc.) keep working.
"""
import torch
import torch.nn as nn
import torch.nn.utils as U

# Backward-compatible re-exports (single implementation in src/projections.py)
from src.projections import (  # noqa: F401
    dimensionality_test,
    get_fa_projection,
    get_ica_projection,
    get_lda_projection,
    get_pairwise_decodings,
    get_pca_projection,
    train_lda_from_tensors,
)


class BaseRNN(nn.Module):
    """RNN with in-forward hidden-state normalization (h / (||h|| * l_norm)).

    Used by the legacy scripts/run_rnn.py pipeline. The newer experiments use
    src.training.ConstrainedRNN, which penalizes the norm in the loss instead.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = None,
                 l_norm: float = 0, name: str = 'ConstrainedRNN'):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if seed:
            torch.manual_seed(seed)
        self.cell = nn.RNNCell(input_size=input_dim, hidden_size=hidden_dim)
        self.fc = U.weight_norm(nn.Linear(hidden_dim, output_dim))
        self.l_norm = l_norm

    def forward(self, input_seq: torch.Tensor):
        B, T, _ = input_seq.shape
        h = torch.zeros(B, self.cell.hidden_size, device=input_seq.device)
        h[:, 0] = 1.0

        outputs = []
        hidden_states = []

        for t in range(T):
            x_t = input_seq[:, t, :]
            h = self.cell(x_t, h)
            h = h / (torch.norm(h, p=2.0, dim=-1, keepdim=True) * self.l_norm)
            out_logits = self.fc(h)
            outputs.append(out_logits.unsqueeze(1))
            hidden_states.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        hidden_states = torch.cat(hidden_states, dim=1)
        return outputs, hidden_states
