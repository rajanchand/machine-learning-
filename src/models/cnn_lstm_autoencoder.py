"""
cnn_lstm_autoencoder.py — Proposed Hybrid 1D-CNN-LSTM Autoencoder with soft Attention.

Combines:
  - 1D Convolutions: learn local feature dependencies and spatial correlation groups.
  - LSTM: captures sequential and feature-order dependencies.
  - Soft Attention: weighs key representation segments dynamically.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils import get_logger, timer, get_device, save_torch_model

logger = get_logger("cnn_lstm_autoencoder")


class Attention(nn.Module):
    """Soft additive attention mechanism over sequence steps."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        # outputs shape: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(encoder_outputs), dim=1)  # (batch, seq_len, 1)
        context = (weights * encoder_outputs).sum(dim=1)  # (batch, hidden_dim)
        return context, weights.squeeze(-1)


class CNNLSTMAutoencoder(nn.Module):
    """
    Proposed NIDS hybrid network combining 1D CNN + LSTM + Attention.
    """
    def __init__(
        self,
        n_features: int,
        conv_channels: int = 16,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.conv_channels = conv_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention

        # Spatial Feature Extractor (1D-CNN)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        
        # We determine the sequence length for LSTM after spatial convolution.
        # Since we convolute across features, the sequence length is n_features,
        # and the number of features per step is conv_channels.
        self.seq_len = n_features

        # Temporal Sequence Modeler (LSTM Encoder)
        self.encoder_lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Attention layer
        if use_attention:
            self.attention = Attention(hidden_dim)

        # Sequence Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        
        # Flattened reconstruction projection
        self.reconstruct_layer = nn.Linear(self.seq_len * hidden_dim, n_features)

    def forward(self, x):
        # x input shape: (batch, 1, n_features)
        
        # 1. 1D CNN Spatial convolution along the features dimension
        # x_conv shape: (batch, conv_channels, n_features)
        x_conv = self.relu(self.conv1(x))
        
        # 2. Reshape for LSTM sequence: (batch, seq_len=n_features, input_size=conv_channels)
        x_lstm = x_conv.permute(0, 2, 1)
        
        # 3. Encode temporal sequence
        enc_out, (h_n, c_n) = self.encoder_lstm(x_lstm)
        
        # 4. Attention fusion or last hidden step
        if self.use_attention:
            context, attn_weights = self.attention(enc_out)
            latent = self.encoder_fc(context)
        else:
            latent = self.encoder_fc(h_n[-1])
            
        # 5. Decode
        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        
        # 6. Reconstruct the original feature array
        # Flatten decoded sequence steps for direct linear scaling
        dec_flat = dec_out.reshape(dec_out.size(0), -1)
        reconstruction = self.reconstruct_layer(dec_flat)
        
        # Output matches tabular input shape: (batch, 1, n_features)
        return reconstruction.unsqueeze(1)


@timer
def train_cnn_lstm_autoencoder(
    X_train_normal: np.ndarray,
    X_val: np.ndarray = None,
    n_features: int = None,
    conv_channels: int = 16,
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
    Train a 1D-CNN-LSTM Autoencoder on normal-only traffic.
    """
    device = get_device()
    n_features = n_features or X_train_normal.shape[1]

    # Reshape for CNN-1D input: (batch, channels=1, length=n_features)
    X_train_t = torch.FloatTensor(X_train_normal).unsqueeze(1)
    train_ds = TensorDataset(X_train_t, X_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
        val_ds = TensorDataset(X_val_t, X_val_t)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CNNCNNLSTMAutoencoder = CNNLSTMAutoencoder(
        n_features=n_features,
        conv_channels=conv_channels,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        dropout=dropout,
        use_attention=use_attention,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
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

    # Dynamic thresholding: mean + 2 * std of training reconstruction error
    model.eval()
    recon_errors = compute_reconstruction_errors(model, X_train_normal, device, batch_size)
    threshold = float(np.mean(recon_errors) + 2 * np.std(recon_errors))
    logger.info(f"Dynamic threshold calculated: {threshold:.6f}")

    return model, train_losses, val_losses, threshold


def compute_reconstruction_errors(model, X: np.ndarray, device=None, batch_size=256):
    """Compute MSE reconstruction error for the CNN-LSTM Autoencoder."""
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
            # Recon shape: (batch, 1, n_features) -> calculate MSE between x and recon
            mse = ((recon - batch_x) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def predict_cnn_lstm_autoencoder(model, X: np.ndarray, threshold: float, device=None, batch_size=256):
    """Return binary predictions and reconstruction anomaly scores."""
    errors = compute_reconstruction_errors(model, X, device, batch_size)
    preds = (errors > threshold).astype(int)
    return preds, errors


def train_and_save(X_train_normal, X_val=None, **kwargs):
    """Train and securely save the proposed CNN-LSTM model with sidecar SHA-256."""
    model, train_losses, val_losses, threshold = train_cnn_lstm_autoencoder(
        X_train_normal, X_val, **kwargs
    )
    metadata = {
        "model_type": "CNN_LSTM_Autoencoder",
        "n_features": kwargs.get("n_features", X_train_normal.shape[1]),
        "conv_channels": kwargs.get("conv_channels", 16),
        "hidden_dim": kwargs.get("hidden_dim", 64),
        "latent_dim": kwargs.get("latent_dim", 32),
        "n_layers": kwargs.get("n_layers", 2),
        "use_attention": kwargs.get("use_attention", True),
        "threshold": threshold,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs": kwargs.get("epochs", 50),
    }
    save_torch_model(model, "cnn_lstm_autoencoder", metadata)
    return model, metadata
