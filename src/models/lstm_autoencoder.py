"""
lstm_autoencoder.py — LSTM Autoencoder with optional Attention for anomaly detection.

Architecture:
  Encoder: Input → LSTM → Bottleneck (latent)
  Attention (optional): Soft attention over encoder hidden states
  Decoder: Latent → LSTM → Reconstruction
  Anomaly Score = Mean Squared Error(input, reconstruction)

Trained on NORMAL traffic only.  High reconstruction error → anomaly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils import get_logger, timer, get_device, save_torch_model

logger = get_logger("lstm_autoencoder")

# ──────────────────────────────────────────────────────────────────────
# Attention Layer
# ──────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Simple additive attention over encoder hidden states."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(encoder_outputs), dim=1)  # (batch, seq_len, 1)
        context = (weights * encoder_outputs).sum(dim=1)  # (batch, hidden_dim)
        return context, weights.squeeze(-1)

# ──────────────────────────────────────────────────────────────────────
# LSTM Autoencoder
# ──────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for sequence reconstruction.
    Input shape: (batch, seq_len, n_features)
    For tabular data, we reshape each sample as (1, n_features) — a single-step sequence.
    """
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
        seq_len: int = 1,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.use_attention = use_attention

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Attention (optional)
        if use_attention:
            self.attention = Attention(hidden_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size = x.size(0)

        # Encode
        enc_out, (h_n, c_n) = self.encoder_lstm(x)

        if self.use_attention:
            context, attn_weights = self.attention(enc_out)
            latent = self.encoder_fc(context)
        else:
            # Use last hidden state
            latent = self.encoder_fc(h_n[-1])

        # Decode
        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        reconstruction = self.output_layer(dec_out)

        return reconstruction

# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

@timer
def train_lstm_autoencoder(
    X_train_normal: np.ndarray,
    X_val: np.ndarray = None,
    n_features: int = None,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    n_layers: int = 2,
    dropout: float = 0.2,
    use_attention: bool = True,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    progress_callback=None,
):
    """
    Train an LSTM Autoencoder on normal-only traffic.
    Returns (model, train_losses, val_losses, threshold).
    """
    device = get_device()
    n_features = n_features or X_train_normal.shape[1]

    # Reshape for LSTM: (batch, seq_len=1, features)
    X_train_t = torch.FloatTensor(X_train_normal).unsqueeze(1)
    train_ds = TensorDataset(X_train_t, X_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
        val_ds = TensorDataset(X_val_t, X_val_t)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        dropout=dropout,
        use_attention=use_attention,
        seq_len=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # — Training —
        model.train()
        epoch_loss = 0
        for batch_x, batch_target in train_loader:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)

            recon = model(batch_x)
            loss = criterion(recon, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # — Validation —
        val_loss = None
        if val_loader:
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for batch_x, batch_target in val_loader:
                    batch_x = batch_x.to(device)
                    batch_target = batch_target.to(device)
                    recon = model(batch_x)
                    v_loss += criterion(recon, batch_target).item() * batch_x.size(0)
            val_loss = v_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f}"
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:.6f}"
            logger.info(msg)

        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_loss, val_loss)

    # — Compute dynamic threshold —
    model.eval()
    recon_errors = compute_reconstruction_errors(model, X_train_normal, device, batch_size)
    threshold = float(np.mean(recon_errors) + 2 * np.std(recon_errors))
    logger.info(f"Dynamic threshold (mean + 2*std): {threshold:.6f}")

    return model, train_losses, val_losses, threshold

# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

def compute_reconstruction_errors(model, X: np.ndarray, device=None, batch_size=256):
    """Compute per-sample MSE reconstruction error."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    X_t = torch.FloatTensor(X).unsqueeze(1)
    ds = TensorDataset(X_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            mse = ((recon - batch_x) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def predict_lstm_autoencoder(model, X: np.ndarray, threshold: float, device=None, batch_size=256):
    """
    Return binary predictions and anomaly scores (reconstruction errors).
    """
    errors = compute_reconstruction_errors(model, X, device, batch_size)
    preds = (errors > threshold).astype(int)
    return preds, errors


def train_and_save(X_train_normal, X_val=None, **kwargs):
    """Full train + save workflow."""
    model, train_losses, val_losses, threshold = train_lstm_autoencoder(
        X_train_normal, X_val, **kwargs
    )
    metadata = {
        "model_type": "LSTM_Autoencoder",
        "n_features": kwargs.get("n_features", X_train_normal.shape[1]),
        "hidden_dim": kwargs.get("hidden_dim", 64),
        "latent_dim": kwargs.get("latent_dim", 32),
        "n_layers": kwargs.get("n_layers", 2),
        "use_attention": kwargs.get("use_attention", True),
        "threshold": threshold,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs": kwargs.get("epochs", 50),
    }
    save_torch_model(model, "lstm_autoencoder", metadata)
    return model, metadata
