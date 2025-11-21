import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # When imported as part of the package.
    from .canopy_plots import createCHM
except ImportError:
    # Fallback for running as a standalone script (python -m hhdc.forward_model).
    from hhdc.canopy_plots import createCHM


class LidarForwardImagingModel(nn.Module):
    def __init__(
        self,
        input_res_m=(2.0, 2.0),
        output_res_m=(3.0, 6.0),
        footprint_diameter_m=10.0,
        b=0.1,
        eta=0.5,
        ref_altitude=500.0,
        ref_photon_count=20.0,
    ):
        """
        Args:
            input_res_m (tuple): Physical size of input pixels (dy, dx) in meters.
            output_res_m (tuple): Physical size of output pixels (dy, dx) in meters.
            footprint_diameter_m (float): The 1/e^2 beam diameter in meters.
            b (float): Background noise.
            eta (float): Readout noise.
            ref_altitude (float): Reference altitude (km).
            ref_photon_count (float): Target photon count.
        """
        super().__init__()
        self.b = b
        self.eta = eta
        self.ref_altitude = ref_altitude
        self.ref_photon_count = ref_photon_count

        self.input_res_m = input_res_m
        self.output_res_m = output_res_m

        # 1. Calculate Area Scale Factor
        in_area = input_res_m[0] * input_res_m[1]
        out_area = output_res_m[0] * output_res_m[1]
        self.area_scale_factor = out_area / in_area

        # 2. Calculate Sigma in Input Pixels
        # Def: 1/e^2 diameter is 4 * sigma
        sigma_m = footprint_diameter_m / 4.0

        avg_input_res = (input_res_m[0] + input_res_m[1]) / 2.0
        sigma_px = sigma_m / avg_input_res

        # 3. Create Base Kernel
        # We use 6*sigma for the kernel size to capture >99% of energy
        # (Since diameter is 4*sigma, this means kernel is 1.5x the footprint size)
        kernel_size = int(math.ceil(6 * sigma_px))
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.register_buffer("kernel", self._create_gaussian_kernel(kernel_size, sigma_px))

        print(f"Model Initialized: In {input_res_m}m -> Out {output_res_m}m")
        print(f"Footprint (1/e^2): {footprint_diameter_m}m (Sigma: {sigma_m:.2f}m / {sigma_px:.2f} px)")

    def _create_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size).float() - (size - 1) / 2
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, X_h, altitude=1.0):
        if X_h.ndim == 3:
            X_h = X_h.unsqueeze(0)

        batch_size, num_bins, h_in, w_in = X_h.shape

        # --- 1. Dynamic Output Size ---
        fov_h_m = h_in * self.input_res_m[0]
        fov_w_m = w_in * self.input_res_m[1]

        out_h = int(fov_h_m / self.output_res_m[0])
        out_w = int(fov_w_m / self.output_res_m[1])
        output_size = (out_h, out_w)

        # --- 2. Physics Normalization ---
        energy_per_tube = X_h.sum(dim=1, keepdim=True)
        global_mean_energy = energy_per_tube.mean(dim=(2, 3), keepdim=True)
        X_norm = X_h / (global_mean_energy + 1e-8)

        dist_scale = (self.ref_altitude / altitude) ** 2
        target_intensity = (self.ref_photon_count / self.area_scale_factor) * dist_scale
        X_scaled = X_norm * target_intensity

        # --- 3. Spatial Blurring ---
        # Expand single-channel kernel to match height bins
        current_kernel = self.kernel.repeat(num_bins, 1, 1, 1)

        padding = current_kernel.shape[-1] // 2
        X_blurred = F.conv2d(X_scaled, current_kernel, padding=padding, groups=num_bins)

        # --- 4. Downsampling ---
        X_binned = F.interpolate(X_blurred, size=output_size, mode='area')
        X_integrated = X_binned * self.area_scale_factor

        # --- 5. Noise ---
        lambda_val = torch.relu(X_integrated) + self.b
        X_l = torch.poisson(lambda_val)
        gaussian_noise = torch.randn_like(X_l) * self.eta
        Y_l = X_l + gaussian_noise

        if Y_l.shape[0] == 1:
            Y_l = Y_l.squeeze(0)

        return Y_l

# --- Example Usage ---
if __name__ == "__main__":
    try:
        X_h = torch.from_numpy(np.load('TestCube/SERC_2016-07_x001_y020.npz')['cube']).float().transpose(0,-1)
    except:
        print("Warning: File not found. Generating dummy canopy data.")
        X_h = torch.zeros((128, 48, 48))
        X_h[10, :, :] = 0.2 # Ground
        X_h[40:60, 12:36, 12:36] = 0.5 # Tree

    # Instantiate model with 1/e^2 Footprint
    model = LidarForwardImagingModel(
        input_res_m=(2.0, 2.0),
        output_res_m=(3.0, 6.0),
        footprint_diameter_m=10.0,
        b=0.1,
        eta=0.5,
        ref_altitude=500.0,
        ref_photon_count=20.0
    )

    TARGET_ALTITUDE = 500.0
    with torch.no_grad():
        Y_l = model(X_h, altitude=TARGET_ALTITUDE)

    mean_photons = Y_l.sum(0).mean().item()
    print(f"--- Simulation Results ---")
    print(f"Target Altitude: {TARGET_ALTITUDE} km")
    print(f"Actual Mean Count:   {mean_photons:.4f}")
    print(f"Input shape: {X_h.shape} -> Output shape: {Y_l.shape}")

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(12, 12))

    X_vis = X_h / (X_h.max(0, keepdim=True)[0] + 1e-8)
    Y_vis = Y_l / (Y_l.max(0, keepdim=True)[0] + 1e-8)

    axs[0,0].imshow(X_vis.numpy()[::-1, :, 0], cmap='gray_r')
    axs[0,0].set_title("High Res Input Slice")

    axs[0,1].imshow(Y_vis.numpy()[::-1, :, 0], aspect=2/3, cmap='gray_r')
    axs[0,1].set_title(f"Low Res Output Slice")

    axs[0,2].plot(X_vis.numpy()[:, 0, 0], label='X_h')
    axs[0,2].plot(Y_vis.numpy()[:, 0, 0], label='Y_l')
    axs[0,2].legend()
    axs[0,2].set_title("Waveform Comparison")

    chm_in = createCHM(X_h.numpy())[0]
    chm_out = createCHM(Y_l.numpy())[0]

    axs[1,0].imshow(chm_in)
    axs[1,0].set_title("Input CHM")

    axs[1,1].imshow(chm_out, aspect=1.5/3)
    axs[1,1].set_title("Output CHM")

    im = axs[1,2].imshow(Y_l.sum(0).numpy())
    axs[1,2].set_title(f"Total Photons (Mean={mean_photons:.1f})")
    plt.colorbar(im, ax=axs[1,2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
