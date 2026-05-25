# plot_endmembers.py
#
# Save spectral signature plots of learned endmembers.
#
# Usage:
#   python plot_endmembers.py \
#       --ckpt /path/to/ckpt_29999_rank0.pt \
#       --output endmembers_plot.png
#
# Optional:
#   --wavelength_start 450
#   --wavelength_end 650

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="endmembers_plot.png",
                        help="Output image filename")
    parser.add_argument("--wavelength_start", type=float, default=450.0)
    parser.add_argument("--wavelength_end", type=float, default=650.0)

    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if "endmembers" not in ckpt:
        raise ValueError("Checkpoint does not contain 'endmembers'.")

    # Shape: [num_endmembers, num_bands]
    endmembers = ckpt["endmembers"]

 
    endmembers = torch.nn.functional.sigmoid(endmembers)

    endmembers = endmembers.detach().cpu().numpy()

    num_endmembers, num_bands = endmembers.shape

    # Wavelength axis
    wavelengths = np.linspace(
        args.wavelength_start,
        args.wavelength_end,
        num_bands
    )

    # Plot
    plt.figure(figsize=(10, 6))

    for i in range(num_endmembers):
        plt.plot(
            wavelengths,
            endmembers[i],
            linewidth=2,
            label=f"Endmember {i+1}"
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance / Intensity")
    plt.title("Spectral Signatures of Learned Endmembers")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save image
    plt.savefig(args.output, dpi=300)

    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()