#!/usr/bin/env python
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
import numpy as np
from typing import Tuple, Optional

class EdgeDetector:
    def __init__(self, default_blur_sigma: float = 1.0, default_kernel_size: int = 5):
        # Store default parameters
        self.default_blur_sigma = default_blur_sigma
        self.default_kernel_size = default_kernel_size
        
        # Initialize filters as float32
        sobel_x_data = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y_data = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        prewitt_x_data = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_y_data = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        laplacian_data = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        
        # Create convolution layers with fixed weights
        self.sobel_x_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        self.prewitt_x_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        self.prewitt_y_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        self.laplacian_conv = Conv2d(1, 1, 3, padding=1, bias=False)
        
        # Set the weights
        self.sobel_x_conv.weight = Tensor(sobel_x_data.reshape(1, 1, 3, 3))
        self.sobel_y_conv.weight = Tensor(sobel_y_data.reshape(1, 1, 3, 3))
        self.prewitt_x_conv.weight = Tensor(prewitt_x_data.reshape(1, 1, 3, 3))
        self.prewitt_y_conv.weight = Tensor(prewitt_y_data.reshape(1, 1, 3, 3))
        self.laplacian_conv.weight = Tensor(laplacian_data.reshape(1, 1, 3, 3))
    
    def create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        """Create a 2D Gaussian kernel"""
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        center = kernel_size // 2
        x, y = np.meshgrid(np.linspace(-center, center, kernel_size),
                          np.linspace(-center, center, kernel_size))
        
        # 2D Gaussian function with explicit float32
        gaussian = np.exp(-(x**2 + y**2)/(2*sigma**2)).astype(np.float32)
        gaussian = gaussian / gaussian.sum()  # Normalize
        
        # Return as 4D tensor (out_channels, in_channels, height, width)
        return Tensor(gaussian).reshape(1, 1, kernel_size, kernel_size)

    def apply_filter(self, image: Tensor, conv_layer: Conv2d) -> Tensor:
        """Apply a convolution filter to the image using nn.Conv2d"""
        # Convert image to float32 if needed
        if not isinstance(image.numpy().dtype, np.float32):
            image = Tensor(image.numpy().astype(np.float32))
        
        # Make the image 4D (batch, channel, height, width)
        if len(image.shape) == 2:
            image = image.reshape(1, 1, *image.shape)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)
        
        # Apply convolution and remove extra dimensions
        result = conv_layer(image)
        return result[0, 0]
    
    def gaussian_blur(self, image: Tensor, kernel_size: Optional[int] = None, 
                     sigma: Optional[float] = None) -> Tensor:
        """Apply Gaussian blur using Conv2d"""
        kernel_size = kernel_size or self.default_kernel_size
        sigma = sigma or self.default_blur_sigma
        
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create Gaussian kernel
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)
        
        # Create conv layer with Gaussian kernel
        gaussian_conv = Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        gaussian_conv.weight = gaussian_kernel
        
        return self.apply_filter(image, gaussian_conv)

    def sobel_edges(self, image: Tensor, blur: bool = True, 
                    blur_kernel_size: Optional[int] = None,
                    blur_sigma: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply Sobel edge detection with Conv2d"""
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
            
        grad_x = self.apply_filter(image, self.sobel_x_conv)
        grad_y = self.apply_filter(image, self.sobel_y_conv)
        
        magnitude = (grad_x ** 2 + grad_y ** 2).sqrt()
        return magnitude, grad_x, grad_y

    def prewitt_edges(self, image: Tensor, blur: bool = True,
                      blur_kernel_size: Optional[int] = None,
                      blur_sigma: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply Prewitt edge detection with Conv2d"""
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
            
        grad_x = self.apply_filter(image, self.prewitt_x_conv)
        grad_y = self.apply_filter(image, self.prewitt_y_conv)
        
        magnitude = (grad_x ** 2 + grad_y ** 2).sqrt()
        return magnitude, grad_x, grad_y

    def laplacian_edges(self, image: Tensor, blur: bool = True,
                        blur_kernel_size: Optional[int] = None,
                        blur_sigma: Optional[float] = None) -> Tensor:
        """Apply Laplacian edge detection with Conv2d"""
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
        return self.apply_filter(image, self.laplacian_conv)

    def non_maximum_suppression(self, magnitude: Tensor, direction: Tensor) -> Tensor:
        """Apply non-maximum suppression to thin edges using tensor operations"""
        # Normalize angles to 0-4 range for binning
        normalized = ((direction + 3.14159) * 4 / 3.14159).floor()
        direction = normalized - (normalized.div(4.0).floor() * 4)
        
        # Pad magnitude for neighbor operations
        padded_magnitude = magnitude.pad(((1,1), (1,1)))
        h, w = magnitude.shape
        
        # Create neighbor arrays by stacking all possible neighbor positions
        # Left neighbor as base tensor for concatenation
        left = padded_magnitude[1:h+1, 0:w].reshape(1, h, w)
        neighbors = left.cat(
            padded_magnitude[1:h+1, 2:w+2].reshape(1, h, w),    # right
            padded_magnitude[0:h, 1:w+1].reshape(1, h, w),      # up
            padded_magnitude[2:h+2, 1:w+1].reshape(1, h, w),    # down
            padded_magnitude[0:h, 0:w].reshape(1, h, w),        # upleft
            padded_magnitude[0:h, 2:w+2].reshape(1, h, w),      # upright
            padded_magnitude[2:h+2, 0:w].reshape(1, h, w),      # downleft
            padded_magnitude[2:h+2, 2:w+2].reshape(1, h, w),    # downright
            dim=0
        )
        
        # For each direction, create masks for corresponding neighbor pairs
        horizontal_mask = (direction == 0).float() + (direction == 2).float()
        vertical_mask = (direction == 1).float() + (direction == 3).float()
        diag45_mask = (direction == 1).float()
        diag135_mask = (direction == 3).float()

        # Create pairs masks for each direction
        # Start with horizontal mask for base concatenation
        horizontal_base = horizontal_mask.reshape(1, h, w)
        horizontal_pairs = horizontal_base.cat(
            horizontal_mask.reshape(1, h, w),    # right
            Tensor.zeros(1, h, w),               # up
            Tensor.zeros(1, h, w),               # down
            Tensor.zeros(1, h, w),               # upleft
            Tensor.zeros(1, h, w),               # upright
            Tensor.zeros(1, h, w),               # downleft
            Tensor.zeros(1, h, w),               # downright
            dim=0
        )

        # Vertical pairs
        vertical_base = Tensor.zeros(1, h, w)
        vertical_pairs = vertical_base.cat(
            Tensor.zeros(1, h, w),              # right
            vertical_mask.reshape(1, h, w),     # up
            vertical_mask.reshape(1, h, w),     # down
            Tensor.zeros(1, h, w),              # upleft
            Tensor.zeros(1, h, w),              # upright
            Tensor.zeros(1, h, w),              # downleft
            Tensor.zeros(1, h, w),              # downright
            dim=0
        )

        # Diagonal 45 pairs
        diag45_base = Tensor.zeros(1, h, w)
        diag45_pairs = diag45_base.cat(
            Tensor.zeros(1, h, w),              # right
            Tensor.zeros(1, h, w),              # up
            Tensor.zeros(1, h, w),              # down
            diag45_mask.reshape(1, h, w),       # upleft
            Tensor.zeros(1, h, w),              # upright
            Tensor.zeros(1, h, w),              # downleft
            diag45_mask.reshape(1, h, w),       # downright
            dim=0
        )

        # Diagonal 135 pairs
        diag135_base = Tensor.zeros(1, h, w)
        diag135_pairs = diag135_base.cat(
            Tensor.zeros(1, h, w),              # right
            Tensor.zeros(1, h, w),              # up
            Tensor.zeros(1, h, w),              # down
            Tensor.zeros(1, h, w),              # upleft
            diag135_mask.reshape(1, h, w),      # upright
            diag135_mask.reshape(1, h, w),      # downleft
            Tensor.zeros(1, h, w),              # downright
            dim=0
        )
        
        # Combine all pair masks
        all_pairs = horizontal_pairs + vertical_pairs + diag45_pairs + diag135_pairs
        
        # Apply masks and get maximum values along neighbor axis
        masked_neighbors = neighbors * all_pairs
        # Max over first dimension (neighbor dimension)
        local_max = masked_neighbors.max()
        
        # Return only points that are local maxima
        return magnitude * (magnitude >= local_max).float()

    def canny_edge_detection(self, 
                           image: Tensor, 
                           low_threshold: float = 0.1, 
                           high_threshold: float = 0.3,
                           sigma: float = 1.0) -> Tensor:
        """
        Implement Canny edge detection
        
        Args:
            image: Input image tensor
            low_threshold: Lower threshold for hysteresis
            high_threshold: Higher threshold for hysteresis
            sigma: Standard deviation for Gaussian filter
            
        Returns:
            Tensor containing edge map
        """
        # 1. Apply Gaussian smoothing
        smoothed = self.gaussian_blur(image, kernel_size=int(6*sigma), sigma=sigma)
        
        # 2. Compute gradients using Sobel
        magnitude, grad_x, grad_y = self.sobel_edges(smoothed, blur=False)
        # Compute angle of gradient using basic math operations
        # atan(x) = 0.5 * ln((1 + x)/(1 - x))
        x = grad_y.div(grad_x + 1e-10)  # add small epsilon to prevent division by zero
        numer = x.add(1)
        denom = x.sub(1).neg()  # 1 - x = -(x - 1)
        angle = numer.div(denom).log().mul(0.5)
        direction = angle
        
        # 3. Non-maximum suppression
        suppressed = self.non_maximum_suppression(magnitude, direction)
        
        # 4. Double thresholding using tensor operations
        high_mask = (suppressed > high_threshold).float()
        low_mask = (suppressed > low_threshold).float()
        
        # 5. Edge tracking by hysteresis using convolution
        # First, set strong edges
        output = high_mask
        
        # Create kernel for 8-connected neighborhood check
        neighbor_kernel = Tensor(np.ones((1, 1, 3, 3), dtype=np.float32))
        neighbor_check = Conv2d(1, 1, 3, padding=1, bias=False)
        neighbor_check.weight = neighbor_kernel
        
        # Identify weak edges
        weak_edges = (low_mask - high_mask).relu()
        
        # Reshape for convolution
        edges_4d = output.reshape(1, 1, *output.shape)
        
        # Check for strong edges in neighborhood
        neighbor_strong = (neighbor_check(edges_4d)[0, 0] > 0).float()
        
        # Add weak edges that are connected to strong edges
        output = output + (weak_edges * neighbor_strong)
        
        return output

# Example usage:
if __name__ == "__main__":
    # Create a sample image
    image = Tensor(np.random.rand(100, 100).astype(np.float32))
    
    # Initialize edge detector
    detector = EdgeDetector()
    
    # Apply different edge detection methods
    sobel_magnitude, sobel_x, sobel_y = detector.sobel_edges(image)
    prewitt_magnitude, prewitt_x, prewitt_y = detector.prewitt_edges(image)
    laplacian = detector.laplacian_edges(image)
    canny = detector.canny_edge_detection(image)
