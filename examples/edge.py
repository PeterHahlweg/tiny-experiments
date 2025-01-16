#!/usr/bin/env python
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
import numpy as np
from typing import Optional
from PIL import Image

class OptimizedCannyEdgeDetector:
    def __init__(self, default_blur_sigma: float = 1.0, default_kernel_size: int = 5):
        # Initialize merged Sobel filter for gradient computation
        # Shape: (2, 1, 3, 3) for X and Y gradients
        sobel_kernels = np.array([
            [[-1, 0, 1],  # X gradient
             [-2, 0, 2],
             [-1, 0, 1]],
            [[-1, -2, -1],  # Y gradient
             [0, 0, 0],
             [1, 2, 1]]
        ], dtype=np.float32)

        # Store default parameters
        self.default_blur_sigma = default_blur_sigma
        self.default_kernel_size = default_kernel_size

        # Initialize single Sobel convolution layer with 2 output channels
        self.sobel_conv = Conv2d(1, 2, 3, padding=1, bias=False)
        # Set up the weights with shape (2, 1, 3, 3)
        self.sobel_conv.weight = Tensor(sobel_kernels.reshape(2, 1, 3, 3))

    def gaussian_blur(self, image: Tensor, kernel_size: Optional[int] = None,
                     sigma: Optional[float] = None) -> Tensor:
        """Apply Gaussian blur using two separate 1D convolutions"""
        sigma = sigma or self.default_blur_sigma

        # If sigma is very close to 0, return original image
        if abs(sigma) < 1e-4:
            return image

        # Calculate appropriate kernel size if not provided
        if kernel_size is None:
            kernel_size = max(3, int(6 * sigma))
            if kernel_size % 2 == 0:
                kernel_size += 1

        if kernel_size < 3:
            kernel_size = 3

        # Create horizontal kernel (row vector)
        center = kernel_size // 2
        x = np.linspace(-center, center, kernel_size)
        gaussian_h = np.exp(-(x**2)/(2*sigma**2)).astype(np.float32)
        gaussian_h = gaussian_h / gaussian_h.sum()
        kernel_h = Tensor(gaussian_h).reshape(1, 1, 1, kernel_size)

        # Create vertical kernel (column vector) - same values, different shape
        kernel_v = Tensor(gaussian_h).reshape(1, 1, kernel_size, 1)

        # Initialize convolution layers
        conv_h = Conv2d(1, 1, (1, kernel_size), padding=(0, kernel_size//2), bias=False)
        conv_v = Conv2d(1, 1, (kernel_size, 1), padding=(kernel_size//2, 0), bias=False)
        conv_h.weight = kernel_h
        conv_v.weight = kernel_v

        # Ensure proper input shape
        if len(image.shape) == 2:
            image = image.reshape(1, 1, *image.shape)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)

        # Apply horizontal convolution followed by vertical convolution
        blurred_h = conv_h(image)
        blurred = conv_v(blurred_h)[0, 0]

        return blurred

    def compute_gradients(self, image: Tensor):
        """Compute gradients using merged Sobel operator"""
        # Ensure proper input shape
        if len(image.shape) == 2:
            image = image.reshape(1, 1, *image.shape)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)

        # Single convolution returns both gradients
        gradients = self.sobel_conv(image)

        # Split channels into X and Y gradients
        grad_x = gradients[:, 0, :, :]  # First channel: X gradient
        grad_y = gradients[:, 1, :, :]  # Second channel: Y gradient

        # Remove batch dimension
        grad_x = grad_x[0]
        grad_y = grad_y[0]

        # Compute magnitude and direction
        magnitude = (grad_x ** 2 + grad_y ** 2).sqrt()
        direction = grad_y.div(grad_x + 1e-10)  # Adding epsilon to avoid division by zero

        return magnitude, direction

    def non_maximum_suppression(self, magnitude: Tensor, direction: Tensor) -> Tensor:
        """Apply non-maximum suppression using available tinygrad operations"""
        # Normalize the direction vector components using direction (dy/dx)
        norm = (direction * direction + 1).sqrt()  # sqrt(dx^2 + dy^2) where dx=1
        dx = (1 / norm)  # normalized dx
        dy = (direction / norm)  # normalized dy

        # Bin the directions into 4 categories based on dx and dy components
        direction_id = ((dy >= dx) * 1 +
                       (dy >= -dx) * 2)  # This gives us 4 direction bins

        # Pad magnitude for neighbor operations
        padded = magnitude.pad(((1,1), (1,1)))
        h, w = magnitude.shape

        # Pre-compute all neighbor values we might need
        n_left = padded[1:h+1, 0:w]
        n_right = padded[1:h+1, 2:w+2]
        n_up = padded[0:h, 1:w+1]
        n_down = padded[2:h+2, 1:w+1]
        n_topleft = padded[0:h, 0:w]
        n_topright = padded[0:h, 2:w+2]
        n_bottomleft = padded[2:h+2, 0:w]
        n_bottomright = padded[2:h+2, 2:w+2]

        # Create masks for each direction
        d0 = (direction_id == 0).float()  # Horizontal
        d1 = (direction_id == 1).float()  # 45 degrees
        d2 = (direction_id == 2).float()  # Vertical
        d3 = (direction_id == 3).float()  # 135 degrees

        # Compare with neighbors based on direction
        suppress0 = (magnitude >= n_left) * (magnitude >= n_right)
        suppress1 = (magnitude >= n_topright) * (magnitude >= n_bottomleft)
        suppress2 = (magnitude >= n_up) * (magnitude >= n_down)
        suppress3 = (magnitude >= n_topleft) * (magnitude >= n_bottomright)

        # Combine results
        result = magnitude * (
            d0 * suppress0 +
            d1 * suppress1 +
            d2 * suppress2 +
            d3 * suppress3
        )

        return result

    def detect_edges(self, image: Tensor, dump_dir: Optional[str] = None) -> Tensor:
        """Canny edge detection with optional intermediate result dumping"""
        # Fixed thresholds
        low_threshold = 0.1
        high_threshold = 0.3

        # 1. Gaussian smoothing
        smoothed = self.gaussian_blur(image)
        if dump_dir:
            save_image(smoothed.numpy(), os.path.join(dump_dir, '1_gaussian_blur.png'))

        # 2. Compute gradients using merged Sobel
        magnitude, direction = self.compute_gradients(smoothed)
        if dump_dir:
            save_image(magnitude.numpy(), os.path.join(dump_dir, '2_gradient_magnitude.png'))

        # 3. Non-maximum suppression
        suppressed = self.non_maximum_suppression(magnitude, direction)
        if dump_dir:
            save_image(suppressed.numpy(), os.path.join(dump_dir, '3_nonmax_suppression.png'))

        # 4. Double thresholding
        high_mask = (suppressed > high_threshold).float()
        low_mask = (suppressed > low_threshold).float()
        if dump_dir:
            save_image(high_mask.numpy(), os.path.join(dump_dir, '4a_high_threshold.png'))
            save_image(low_mask.numpy(), os.path.join(dump_dir, '4b_low_threshold.png'))

        # 5. Edge tracking by hysteresis
        weak_edges = (low_mask - high_mask).relu()
        if dump_dir:
            save_image(weak_edges.numpy(), os.path.join(dump_dir, '5a_weak_edges.png'))

        # Setup convolution for checking neighbors
        edges_4d = high_mask.reshape(1, 1, *high_mask.shape)
        neighbor_kernel = Tensor(np.ones((1, 1, 3, 3), dtype=np.float32))
        neighbor_check = Conv2d(1, 1, 3, padding=1, bias=False)
        neighbor_check.weight = neighbor_kernel

        # Find weak edges connected to strong edges
        neighbor_strong = (neighbor_check(edges_4d)[0, 0] > 0).float()
        if dump_dir:
            save_image(neighbor_strong.numpy(), os.path.join(dump_dir, '5b_strong_neighbors.png'))

        # Combine strong edges with connected weak edges
        final_edges = high_mask + (weak_edges * neighbor_strong)

        return final_edges

def load_image(path):
    """Load an image and convert it to grayscale numpy array."""
    img = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]

def save_image(array, path):
    """Save a numpy array as an image."""
    array = (array * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(array)
    img.save(path)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Optimized Canny Edge Detection using TinyGrad')
    parser.add_argument('--input', '-i', type=str, help='Input image path')
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--use-test-image', '-t', action='store_true',
                       help='Use built-in test image instead of loading from file')
    parser.add_argument('--dump', action='store_true',
                       help='Save intermediate processing results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_test_image:
        print("Creating test image...")
        test_image = np.zeros((1080, 1920), dtype=np.float32)
        test_image[400:600, :] = 1.0  # Create a horizontal band
        image_array = test_image
        save_image(image_array, os.path.join(args.output_dir, 'test_image.png'))
    else:
        if not args.input:
            raise ValueError("Please provide an input image path or use --use-test-image")
        print(f"Loading image from {args.input}...")
        image_array = load_image(args.input)

    # Convert to Tensor
    image = Tensor(image_array)

    print("Initializing optimized Canny edge detector...")
    detector = OptimizedCannyEdgeDetector()

    print("\nDetecting edges...")
    edges = detector.detect_edges(
        image,
        dump_dir=args.output_dir if args.dump else None
    )

    # Save result
    output_path = os.path.join(args.output_dir, 'canny_edges.png')
    save_image(edges.numpy(), output_path)
    print(f"\nEdge detection result saved to: {output_path}")