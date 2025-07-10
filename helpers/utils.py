import torch


def pad_sequence_dims(sequences, dims, padding_value=0):
    assert all([len(sequences[0].shape) == len(sequences[i].shape) for i in range(1, len(sequences))])
    assert isinstance(dims, tuple)

    max_shape = []
    for i in range(len(sequences[0].shape)):
        if i in dims:
            max_shape.append(max([s.shape[i] for s in sequences]))
        else:
            static_dim = [s.shape[i] for s in sequences]
            assert min(static_dim) == max(static_dim)
            max_shape.append(sequences[0].shape[i])

    padded_tensor = torch.full([len(sequences)] + max_shape, padding_value,
                               dtype=sequences[0].dtype, device=sequences[0].device)
    for i, s in enumerate(sequences):
        new_shape = tuple([i] + [slice(0, s.shape[j]) for j in range(len(s.shape))])
        padded_tensor[new_shape].copy_(s)
    return padded_tensor


def fourier_filter(x, scale, d_s=0.25):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = torch.fft.fftn(x, dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    H, W = x_freq.shape[-2], x_freq.shape[-1]

    grid_H, grid_W = torch.meshgrid(torch.linspace(-1, 1, H, device=x_freq.device),
                                    torch.linspace(-1, 1, W, device=x_freq.device), indexing='ij')
    d_square = grid_H**2 + grid_W**2
    mask = torch.where(d_square <= 2 * d_s,
                       scale * torch.ones_like(x_freq),
                       torch.ones_like(x_freq))

    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered


def fourier_filter_1d(x, scale, d_s=0.25):
    dtype = x.dtype
    x = x.type(torch.float32)

    # FFT (1D)
    x_freq = torch.fft.fftn(x, dim=(-1,))
    x_freq = torch.fft.fftshift(x_freq, dim=(-1,))

    # Get the length of the signal
    L = x_freq.shape[-1]

    # Create 1D frequency grid
    grid = torch.linspace(-1, 1, L, device=x_freq.device)

    # Calculate distances (in 1D, it's just the squared value of the position)
    d_square = grid**2

    # Create mask (passing frequencies below threshold)
    mask = torch.where(d_square <= 2 * d_s,
                       scale * torch.ones_like(x_freq),
                       torch.ones_like(x_freq))

    # Apply mask
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-1,))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-1,)).real
    x_filtered = x_filtered.type(dtype)

    return x_filtered
