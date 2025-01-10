#!/usr/bin/env python
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt  # For visualization

class EdgeDetector:
    def __init__(self, default_blur_sigma: float = 1.0, default_kernel_size: int = 5):
        # Initialize filters
        sobel_x_data = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y_data = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        prewitt_x_data = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_y_data = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        laplacian_data = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        
        # Store default parameters
        self.default_blur_sigma = default_blur_sigma
        self.default_kernel_size = default_kernel_size
        
        # Initialize convolution layers and move weights to GPU
        self.sobel_x_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        self.prewitt_x_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        self.prewitt_y_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        self.laplacian_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        
        # Set up the weights on GPU
        self.sobel_x_conv.weight = Tensor(sobel_x_data.reshape(1, 1, 3, 3))
        self.sobel_y_conv.weight = Tensor(sobel_y_data.reshape(1, 1, 3, 3))
        self.prewitt_x_conv.weight = Tensor(prewitt_x_data.reshape(1, 1, 3, 3))
        self.prewitt_y_conv.weight = Tensor(prewitt_y_data.reshape(1, 1, 3, 3))
        self.laplacian_conv.weight = Tensor(laplacian_data.reshape(1, 1, 3, 3))

    def create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        """Create a 2D Gaussian kernel"""
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        center = kernel_size // 2
        x, y = np.meshgrid(np.linspace(-center, center, kernel_size),
                          np.linspace(-center, center, kernel_size))
        
        gaussian = np.exp(-(x**2 + y**2)/(2*sigma**2)).astype(np.float32)
        gaussian = gaussian / gaussian.sum()
        
        return Tensor(gaussian).reshape(1, 1, kernel_size, kernel_size)

    def apply_filter(self, image: Tensor, conv_layer: Conv2d) -> Tensor:
        """Apply a convolution filter to the image"""
        if len(image.shape) == 2:
            image = image.reshape(1, 1, *image.shape)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)
        
        return conv_layer(image)[0, 0]

    def gaussian_blur(self, image: Tensor, kernel_size: Optional[int] = None, 
                     sigma: Optional[float] = None) -> Tensor:
        """Apply Gaussian blur"""
        kernel_size = kernel_size or self.default_kernel_size
        sigma = sigma or self.default_blur_sigma
        
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)
        gaussian_conv = Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        gaussian_conv.weight = gaussian_kernel
        
        return self.apply_filter(image, gaussian_conv)

    def sobel_edges(self, image: Tensor, blur: bool = True,
                    blur_kernel_size: Optional[int] = None,
                    blur_sigma: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply Sobel edge detection"""
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
            
        grad_x = self.apply_filter(image, self.sobel_x_conv)
        grad_y = self.apply_filter(image, self.sobel_y_conv)
        
        magnitude = (grad_x ** 2 + grad_y ** 2).sqrt()
        return magnitude, grad_x, grad_y

    def prewitt_edges(self, image: Tensor, blur: bool = True,
                      blur_kernel_size: Optional[int] = None,
                      blur_sigma: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply Prewitt edge detection"""
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
            
        grad_x = self.apply_filter(image, self.prewitt_x_conv)
        grad_y = self.apply_filter(image, self.prewitt_y_conv)
        
        magnitude = (grad_x ** 2 + grad_y ** 2).sqrt()
        return magnitude, grad_x, grad_y

    def laplacian_edges(self, image: Tensor, blur: bool = True,
                       blur_kernel_size: Optional[int] = None,
                       blur_sigma: Optional[float] = None) -> Tensor:
        """Apply Laplacian edge detection"""
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
        return self.apply_filter(image, self.laplacian_conv)

    def non_maximum_suppression(self, magnitude: Tensor, direction: Tensor) -> Tensor:
        """Apply non-maximum suppression using tensor operations"""
        # Normalize angles to 0-4 range for binning
        normalized = ((direction + 3.14159) * 4 / 3.14159).floor()
        direction = normalized - (normalized.div(4.0).floor() * 4)
        
        # Pad magnitude for neighbor operations
        padded_magnitude = magnitude.pad(((1,1), (1,1)))
        h, w = magnitude.shape
        
        # Pre-compute all neighbor values
        n_left = padded_magnitude[1:h+1, 0:w].reshape(1, h, w)
        n_right = padded_magnitude[1:h+1, 2:w+2].reshape(1, h, w)
        n_up = padded_magnitude[0:h, 1:w+1].reshape(1, h, w)
        n_down = padded_magnitude[2:h+2, 1:w+1].reshape(1, h, w)
        n_upleft = padded_magnitude[0:h, 0:w].reshape(1, h, w)
        n_upright = padded_magnitude[0:h, 2:w+2].reshape(1, h, w)
        n_downleft = padded_magnitude[2:h+2, 0:w].reshape(1, h, w)
        n_downright = padded_magnitude[2:h+2, 2:w+2].reshape(1, h, w)
        
        # Stack neighbors
        neighbors = n_left.cat(
            n_right, n_up, n_down,
            n_upleft, n_upright, n_downleft, n_downright,
            dim=0
        )
        
        # Create masks
        horizontal_mask = (direction == 0).float() + (direction == 2).float()
        vertical_mask = (direction == 1).float() + (direction == 3).float()
        diag45_mask = (direction == 1).float()
        diag135_mask = (direction == 3).float()
        
        # Create direction masks tensor
        direction_masks = horizontal_mask.reshape(1, h, w).cat(
            horizontal_mask.reshape(1, h, w),
            vertical_mask.reshape(1, h, w),
            vertical_mask.reshape(1, h, w),
            diag45_mask.reshape(1, h, w),
            diag135_mask.reshape(1, h, w),
            diag45_mask.reshape(1, h, w),
            diag135_mask.reshape(1, h, w),
            dim=0
        )
        
        # Apply masks and compute max
        masked_neighbors = neighbors * direction_masks
        local_max = masked_neighbors.max()
        
        return magnitude * (magnitude >= local_max).float()

    def canny_edge_detection(self, image: Tensor, low_threshold: float = 0.1,
                           high_threshold: float = 0.3, sigma: float = 1.0) -> Tensor:
        """Canny edge detection maintaining GPU residency"""
        # 1. Apply Gaussian smoothing
        smoothed = self.gaussian_blur(image, kernel_size=int(6*sigma), sigma=sigma)
        
        # 2. Compute gradients using Sobel
        magnitude, grad_x, grad_y = self.sobel_edges(smoothed, blur=False)
        
        # Compute gradient direction
        epsilon = 1e-10
        direction = grad_y.div(grad_x + epsilon)
        
        # 3. Non-maximum suppression
        suppressed = self.non_maximum_suppression(magnitude, direction)
        
        # 4. Double thresholding
        high_mask = (suppressed > high_threshold).float()
        low_mask = (suppressed > low_threshold).float()
        
        # 5. Edge tracking by hysteresis
        neighbor_kernel = Tensor(np.ones((1, 1, 3, 3), dtype=np.float32))
        neighbor_check = Conv2d(1, 1, 3, padding=1, bias=False)
        neighbor_check.weight = neighbor_kernel
        
        weak_edges = (low_mask - high_mask).relu()
        edges_4d = high_mask.reshape(1, 1, *high_mask.shape)
        neighbor_strong = (neighbor_check(edges_4d)[0, 0] > 0).float()
        
        return high_mask + (weak_edges * neighbor_strong)

def load_image(path):
    """Load an image and convert it to grayscale numpy array."""
    img = Image.open(path).convert('L')  # Convert to grayscale
    return np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]

def save_image(array, path):
    """Save a numpy array as an image."""
    # Ensure the array is in the correct range [0, 255]
    array = (array * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(array)
    img.save(path)

def visualize_results(original, canny, sobel):
    """Visualize the original image and detection results."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(canny, cmap='gray')
    ax2.set_title('Canny Edges')
    ax2.axis('off')
    
    ax3.imshow(sobel, cmap='gray')
    ax3.set_title('Sobel Edges')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Edge detection using TinyGrad')
    parser.add_argument('--input', '-i', type=str, help='Input image path')
    parser.add_argument('--output-dir', '-o', type=str, default='output', 
                       help='Output directory for results')
    parser.add_argument('--use-test-image', '-t', action='store_true',
                       help='Use built-in test image instead of loading from file')
    args = parser.parse_args()
    
    if args.use_test_image:
        print("Creating test image...")
        test_image = np.zeros((100, 100), dtype=np.float32)
        test_image[40:60, :] = 1.0  # Create a horizontal band
        image_array = test_image
    else:
        if not args.input:
            raise ValueError("Please provide an input image path or use --use-test-image")
        print(f"Loading image from {args.input}...")
        image_array = load_image(args.input)
    
    # Convert to Tensor
    image = Tensor(image_array)
    
    print("Initializing edge detector...")
    detector = EdgeDetector()
    
    print("\nComputing Canny edges...")
    canny = detector.canny_edge_detection(image)
    print("Canny edge shape:", canny.shape)
    
    print("\nComputing Sobel edges...")
    sobel_magnitude, sobel_x, sobel_y = detector.sobel_edges(image)
    print("Sobel magnitude shape:", sobel_magnitude.shape)
    
    # Force computation and get results
    print("\nComputing final results...")
    canny_result = canny.numpy()
    sobel_result = sobel_magnitude.numpy()
    
    print("\nResults summary:")
    print(f"Canny edges - min: {canny_result.min():.4f}, max: {canny_result.max():.4f}")
    print(f"Sobel edges - min: {sobel_result.min():.4f}, max: {sobel_result.max():.4f}")
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_image(canny_result, os.path.join(args.output_dir, 'canny_edges.png'))
    save_image(sobel_result, os.path.join(args.output_dir, 'sobel_edges.png'))
    
    # Visualize results
    visualize_results(image_array, canny_result, sobel_result)
