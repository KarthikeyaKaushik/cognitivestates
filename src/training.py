"""Shared RNN training stack for the cognitive-diagram experiments.

Extracted verbatim from scripts/run_comparison.py (the same classes were
copy-pasted in run_constrained.py and run_probe.py). Computation only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U

from src.cognitive_diagram import CognitiveDiagram


class ConstrainedRNN(nn.Module):
    """Vanilla RNN with a weight-normed linear readout.

    The "constraint" is an L2 activity penalty on the hidden states, weighted
    by ``multiplier = 1 / lnorm`` and added to the loss in ``train_model``.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 seed: int = None, lnorm: float = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if seed:  # NB: seed=0 intentionally leaves the ambient torch seed in place
            torch.manual_seed(seed)
        self.cell = nn.RNNCell(input_size=input_dim, hidden_size=hidden_dim)
        self.fc = U.weight_norm(nn.Linear(hidden_dim, output_dim))
        self.lnorm = lnorm
        self.multiplier = 1 / lnorm if lnorm else 0.0

    def forward(self, input_seq: torch.Tensor):
        B, T, _ = input_seq.shape
        h = torch.zeros(B, self.cell.hidden_size, device=input_seq.device)
        h[:, 0] = 1.0
        outputs = []
        hidden_states = []

        for t in range(T):
            x_t = input_seq[:, t, :]
            h = self.cell(x_t, h)
            out_logits = self.fc(h)
            outputs.append(out_logits.unsqueeze(1))
            hidden_states.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        hidden_states = torch.cat(hidden_states, dim=1)
        return outputs, hidden_states


class DataGenerator:
    """One-hot tensor batches from (input_tokens, output_tokens) pairs.

    The cognitive diagram supplies the vocabularies and the per-step state
    trace, which is attached to each batch as a third (one-hot) tensor.
    """

    def __init__(self, data, cog_diagram: CognitiveDiagram, batch_size: int = 32,
                 train_frac: float = 0.8):
        self.data = data
        self.cog_diagram = cog_diagram
        self.batch_size = batch_size
        self.input_map = {token: idx for idx, token in enumerate(sorted(cog_diagram.X))}
        self.inv_input_map = {v: k for k, v in self.input_map.items()}
        self.output_map = {token: idx for idx, token in enumerate(sorted(cog_diagram.Y))}
        self.inv_output_map = {v: k for k, v in self.output_map.items()}
        self.state_map = {state: idx for idx, state in enumerate(sorted(cog_diagram.states))}
        self.inv_state_map = {v: k for k, v in self.state_map.items()}
        self.tensor_data = self.get_tensor_data()
        num_train = int(len(self.tensor_data) * train_frac)
        indices = torch.randperm(len(self.tensor_data)).tolist()
        self.train = [self.tensor_data[i] for i in indices[:num_train]]
        self.test = [self.tensor_data[i] for i in indices[num_train:]]

    def get_tensor_data(self):
        tensor_data = []
        for inp_seq, out_seq in self.data:
            inp_indices = [self.input_map[token] for token in inp_seq]
            out_indices = [self.output_map[token] for token in out_seq]
            state_seq = [self.state_map[state] for state in self.cog_diagram.trace(inp_seq)]
            inp_onehot = F.one_hot(torch.tensor(inp_indices), num_classes=len(self.input_map)).float()
            out_onehot = F.one_hot(torch.tensor(out_indices), num_classes=len(self.output_map)).float()
            state_onehot = F.one_hot(torch.tensor(state_seq), num_classes=len(self.state_map)).float()
            tensor_data.append((inp_onehot, out_onehot, state_onehot))
        return tensor_data

    def batch_generator(self, train: bool = True):
        data = self.train if train else self.test
        num_examples = len(data)
        indices = torch.randperm(num_examples)
        for i in range(0, num_examples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_inputs = torch.stack([data[idx][0] for idx in batch_indices])
            batch_targets = torch.stack([data[idx][1] for idx in batch_indices])
            batch_states = torch.stack([data[idx][2] for idx in batch_indices])
            yield batch_inputs, batch_targets, batch_states


def readout_weight_norm(model):
    """Frobenius norm of the effective weight-normalized readout, excluding bias."""
    # Each row of the effective weight has norm abs(weight_g). The cached
    # fc.weight is stale immediately after an optimizer step.
    return model.fc.weight_g.detach().norm().item()


def train_model(model, datagenerator, learning_rate, num_epochs,
                loss_tracking_epochs=1):
    """Train with cross-entropy plus the (1/lnorm)-weighted hidden-norm penalty.

    Returns (model, track_loss) with loss, accuracy and post-update readout
    weight norm recorded every loss_tracking_epochs epochs.
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    track_loss = {'loss': [], 'acc': [], 'epochs': [], 'readout_weight_norm': []}
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        for batch_inputs, batch_targets, batch_states in datagenerator.batch_generator(train=True):
            optimizer.zero_grad()
            outputs, hidden_states = model(batch_inputs)
            target_indices = batch_targets.argmax(dim=-1).view(-1)
            predictions = outputs.view(-1, outputs.size(-1))

            loss = criterion(predictions, target_indices).mean()
            loss += torch.norm(hidden_states, p=2.0, dim=-1).mean() * model.multiplier
            loss.backward()
            optimizer.step()

            predicted_classes = predictions.argmax(dim=-1)
            epoch_acc += (predicted_classes == target_indices).float().mean().item()
            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        avg_epoch_acc = epoch_acc / batch_count

        if epoch % loss_tracking_epochs == 0:
            readout_norm = readout_weight_norm(model)
            track_loss['loss'].append(avg_loss)
            track_loss['acc'].append(avg_epoch_acc)
            track_loss['epochs'].append(epoch)
            track_loss['readout_weight_norm'].append(readout_norm)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss : {avg_loss:.4f} - "
                  f"Acc : {avg_epoch_acc:.4f} - Readout weight norm : {readout_norm:.4f}")

    return model, track_loss


def get_hidden_states(model, datagenerator, train=True):
    """Run the model over one split and collect all tensors.

    train=True uses the 80% training split (legacy behavior); train=False the
    held-out test split. Returns (hidden_states, state_truths, inputs,
    outputs), each (N, T, dim).
    """
    model.eval()
    all_hidden_states, all_truths = [], []
    all_inputs, all_outputs = [], []
    for batch_inputs, batch_targets, batch_states in datagenerator.batch_generator(train=train):
        outputs, hidden_states = model(batch_inputs)
        if len(all_hidden_states) == 0:
            all_hidden_states = hidden_states
            all_truths = batch_states
            all_inputs = batch_inputs
            all_outputs = batch_targets
        else:
            all_hidden_states = torch.concat([all_hidden_states, hidden_states], dim=0)
            all_truths = torch.concat([all_truths, batch_states], dim=0)
            all_inputs = torch.concat([all_inputs, batch_inputs], dim=0)
            all_outputs = torch.concat([all_outputs, batch_targets], dim=0)
    return all_hidden_states, all_truths, all_inputs, all_outputs


def flatten_states(hidden_states, truths):
    """(N, T, D) hidden states + one-hot truths -> (N*T, D) predictor and
    (N*T,) integer state targets."""
    predictor = hidden_states.reshape(
        (hidden_states.shape[0] * hidden_states.shape[1], hidden_states.shape[2]))
    target = truths.reshape((truths.shape[0] * truths.shape[1], truths.shape[2]))
    target = torch.argmax(target, axis=-1)
    return predictor, target


def balance_classes(predictor, target):
    """Subsample so every state has as many samples as the rarest one.
    Returns (predictor, target, sampled_indices)."""
    min_indices = target.unique(return_counts=True)[1].min()
    sampled = []
    for state in target.unique():
        state_indices = (target == state).nonzero(as_tuple=True)[0]
        sampled.append(state_indices[torch.randperm(len(state_indices))[:min_indices]])
    sampled = torch.cat(sampled, dim=0)
    return predictor[sampled], target[sampled], sampled
