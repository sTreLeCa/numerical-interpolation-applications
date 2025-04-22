# Numerical Interpolation Applications

This repository contains Python examples demonstrating numerical interpolation techniques for both 1D audio signals and 2D images.

## Projects

1.  **Audio Interpolation (`CP2 audio.py`):**
    *   Generates speech audio from text using `gTTS`.
    *   Applies 1D interpolation methods (Linear, Piecewise Cubic Hermite/PCHIP, Cubic Spline) using `SciPy` and `NumPy` to the audio signal samples.
    *   Saves the original and interpolated audio signals as `.wav` files.
    *   Plots the original and interpolated waveforms using `Matplotlib`.

2.  **Image Restoration (`CP2 image.py`):**
    *   Loads an input image (`32_example_image.png`) using `Pillow`.
    *   Randomly removes a specified percentage of pixels (simulating data loss).
    *   Restores the image using 2D Radial Basis Function (RBF) interpolation (`scipy.interpolate.Rbf`) with different kernel functions (`gaussian`, `linear`, `multiquadric`).
    *   Saves the modified and restored images as `.png` files.
    *   Displays the original, modified, and restored images using `Matplotlib` for visual comparison across different removal percentages and RBF kernels.
