import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    An improved Encoder with additional convolution blocks, 
    optional dropout, and a two-layer GRU for deeper temporal modeling.
    """
    def __init__(self,
                 input_dim=79,
                 latent_dim=16,
                 latent_seq_len=5,
                 num_gru_layers=2,
                 conv_dropout=0.1,
                 gru_dropout=0.1):
        """
        Args:
            input_dim (int): Number of input features (keys).
            latent_dim (int): Dimension of the latent space.
            latent_seq_len (int): Desired sequence length of the latent representation.
            num_gru_layers (int): Number of GRU layers for deeper temporal modeling.
            conv_dropout (float): Dropout rate in convolutional blocks.
            gru_dropout (float): Dropout rate in GRU (applies between RNN layers if > 1 layer).
        """
        super(Encoder, self).__init__()
        self.latent_seq_len = latent_seq_len
        self.num_gru_layers = num_gru_layers

        # Convolutional blocks
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()
        self.ln1 = nn.LayerNorm(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv2_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(256)

        # Optional extra conv block for deeper capacity
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()
        self.ln3 = nn.LayerNorm(256)

        # Adaptive pooling to reduce sequence length to latent_seq_len
        self.pool = nn.AdaptiveAvgPool1d(latent_seq_len)

        # GRU layer (2-layer, bidirectional can also be used if desired)
        self.gru = nn.GRU(input_size=256, hidden_size=latent_dim,
                          num_layers=num_gru_layers, batch_first=True,
                          dropout=gru_dropout if num_gru_layers > 1 else 0)
        self.ln_gru = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Shape (batch_size, input_dim, sequence_length)

        Returns:
            z (torch.Tensor): Latent representation of shape (batch_size, latent_dim, latent_seq_len)
        """
        # First Conv Block
        x = self.conv1(x)                              # (B, 128, seq_len)
        x = F.relu(x)
        x = self.conv1_dropout(x)
        x = x.transpose(1, 2)                          # (B, seq_len, 128)
        x = self.ln1(x).transpose(1, 2)                # (B, 128, seq_len)

        # Second Conv Block
        x = self.conv2(x)                              # (B, 256, seq_len)
        x = F.relu(x)
        x = self.conv2_dropout(x)
        x = x.transpose(1, 2)                          # (B, seq_len, 256)
        x = self.ln2(x).transpose(1, 2)                # (B, 256, seq_len)

        # Extra Conv Block
        x = self.conv3(x)                              # (B, 256, seq_len)
        x = F.relu(x)
        x = self.conv3_dropout(x)
        x = x.transpose(1, 2)                          # (B, seq_len, 256)
        x = self.ln3(x).transpose(1, 2)                # (B, 256, seq_len)

        # Pool to reduce seq_len -> latent_seq_len
        x = self.pool(x)                               # (B, 256, latent_seq_len)

        # Prepare for GRU
        x = x.transpose(1, 2)                          # (B, latent_seq_len, 256)

        # GRU pass
        z, _ = self.gru(x)                             # (B, latent_seq_len, latent_dim)
        z = self.ln_gru(z)                             # (B, latent_seq_len, latent_dim)

        # Transpose to (B, latent_dim, latent_seq_len)
        z = z.transpose(1, 2)
        return z


class Decoder(nn.Module):
    """
    A corresponding Decoder with a two-layer structure:
    GRU -> Upsample -> Deconvolutions. Also includes optional dropout.
    """
    def __init__(self,
                 latent_dim=16,
                 output_dim=79,
                 original_seq_len=10,
                 num_gru_layers=2,
                 conv_dropout=0.1,
                 gru_dropout=0.1):
        """
        Args:
            latent_dim (int): Dimension of the latent space.
            output_dim (int): Number of output features (keys).
            original_seq_len (int): Original sequence length (10 in your case).
            num_gru_layers (int): Number of GRU layers in decoder.
            conv_dropout (float): Dropout rate in deconv layers.
            gru_dropout (float): Dropout rate in GRU.
        """
        super(Decoder, self).__init__()
        self.original_seq_len = original_seq_len

        # GRU to expand latent back to 256 dim
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=256,
                          num_layers=num_gru_layers, batch_first=True,
                          dropout=gru_dropout if num_gru_layers > 1 else 0)
        self.ln_gru = nn.LayerNorm(256)

        # Upsampling to original sequence length
        self.upsample = nn.Upsample(size=original_seq_len, mode='linear', align_corners=True)

        # Deconvolution blocks
        self.deconv_dropout = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()

        self.deconv1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(128)

        self.deconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=output_dim, kernel_size=3, padding=1)

    def forward(self, z):
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Shape (batch_size, latent_dim, latent_seq_len)

        Returns:
            x_recon (torch.Tensor): Reconstructed shape (batch_size, output_dim, original_seq_len)
        """
        # Prepare for GRU: (B, latent_seq_len, latent_dim)
        z = z.transpose(1, 2)                           # (B, latent_seq_len, latent_dim)

        # GRU
        z, _ = self.gru(z)                              # (B, latent_seq_len, 256)
        z = self.ln_gru(z)                              # (B, latent_seq_len, 256)

        # Transpose to (B, 256, latent_seq_len)
        z = z.transpose(1, 2)                           # (B, 256, latent_seq_len)

        # Upsample to original sequence length
        z = self.upsample(z)                            # (B, 256, original_seq_len)

        # Deconvolution 1
        z = self.deconv1(z)                             # (B, 128, original_seq_len)
        z = F.relu(z)
        z = self.deconv_dropout(z)
        # LayerNorm across channels
        z = z.transpose(1, 2)
        z = self.ln1(z).transpose(1, 2)                 # (B, 128, original_seq_len)

        # Deconvolution 2 -> final output
        x_recon = self.deconv2(z)                       # (B, output_dim, original_seq_len)
        return x_recon


class KeyPressAutoencoder(nn.Module):
    """
    A deterministic autoencoder combining the improved Encoder and Decoder.
    """
    def __init__(self,
                 input_dim=79,
                 latent_dim=16,
                 latent_seq_len=5,
                 original_seq_len=10,
                 num_gru_layers=2,
                 conv_dropout=0.1,
                 gru_dropout=0.1):
        """
        Args:
            input_dim (int): Number of input features (79 keys).
            latent_dim (int): Dimension of the latent space.
            latent_seq_len (int): Compressed sequence length in the latent space.
            original_seq_len (int): Original sequence length (10 time bins).
            num_gru_layers (int): Number of GRU layers in both encoder and decoder.
            conv_dropout (float): Dropout in conv/deconv blocks.
            gru_dropout (float): Dropout in GRU.
        """
        super(KeyPressAutoencoder, self).__init__()
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            latent_seq_len=latent_seq_len,
            num_gru_layers=num_gru_layers,
            conv_dropout=conv_dropout,
            gru_dropout=gru_dropout
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            original_seq_len=original_seq_len,
            num_gru_layers=num_gru_layers,
            conv_dropout=conv_dropout,
            gru_dropout=gru_dropout
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch_size, input_dim, original_seq_len)

        Returns:
            (x_recon, z):
                x_recon: (batch_size, input_dim, original_seq_len)
                z: (batch_size, latent_dim, latent_seq_len)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def get_latent_l1_loss(self, z):
        """
        Optional L1 penalty on the latent representation.
        Returns the mean absolute value of the latent activations.

        Usage in your training loop:
            recon_loss = criterion(x_recon, x)
            latent_l1 = model.get_latent_l1_loss(z)
            loss = recon_loss + alpha * latent_l1
        """
        return z.abs().mean()
