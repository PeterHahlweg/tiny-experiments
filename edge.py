#!/usr/bin/env python
from tinygrad.tensor import Tensor
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
        
        # Create tensors with proper shapes
        self.sobel_x = Tensor(sobel_x_data).reshape(1, 1, 3, 3)
        self.sobel_y = Tensor(sobel_y_data).reshape(1, 1, 3, 3)
        self.prewitt_x = Tensor(prewitt_x_data).reshape(1, 1, 3, 3)
        self.prewitt_y = Tensor(prewitt_y_data).reshape(1, 1, 3, 3)
        self.laplacian = Tensor(laplacian_data).reshape(1, 1, 3, 3)
    
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
    
    def gaussian_blur(self, image: Tensor, kernel_size: Optional[int] = None, 
                     sigma: Optional[float] = None) -> Tensor:
        """Apply Gaussian blur to the image"""
        kernel_size = kernel_size or self.default_kernel_size
        sigma = sigma or self.default_blur_sigma
        
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create full 2D Gaussian kernel
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)
        print(f"Created Gaussian kernel with size {kernel_size}x{kernel_size}, sigma={sigma}")
        print(f"Gaussian kernel shape: {gaussian_kernel.shape}")
        
        # Apply convolution
        return self.apply_filter(image, gaussian_kernel)

    def apply_filter(self, image: Tensor, kernel: Tensor) -> Tensor:
        """Apply a convolution filter to the image using basic operations"""
        # Convert image to float32 if needed
        if not isinstance(image.numpy().dtype, np.float32):
            image = Tensor(image.numpy().astype(np.float32))
        
        # Debug initial shapes
        print(f"Input image shape: {image.shape}")
        print(f"Input kernel shape: {kernel.shape}")
        
        # Make the image 4D (batch, channel, height, width)
        if len(image.shape) == 2:
            image = image.reshape(1, 1, *image.shape)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)
        
        # Make sure the kernel is 4D (out_channels, in_channels, height, width)
        if len(kernel.shape) != 4:
            kernel = kernel.reshape(1, 1, kernel.shape[-2], kernel.shape[-1])
        
        # Get dimensions
        bs, c, h, w = image.shape
        _, _, kh, kw = kernel.shape
        
        # Calculate padding size
        h_pad, w_pad = kh//2, kw//2
        
        print(f"After reshape - Image shape: {image.shape}")
        print(f"After reshape - Kernel shape: {kernel.shape}")
        print(f"Padding size: {h_pad}, {w_pad}")
        
        # Pad image
        padded = image.pad(((0,0), (0,0), (h_pad,h_pad), (w_pad,w_pad)))
        print(f"Padded shape: {padded.shape}")
        
        # Create output array using numpy first
        output = np.zeros((h, w), dtype=np.float32)
        
        # Perform convolution
        for i in range(h):
            for j in range(w):
                # Extract patch
                patch = padded[:, :, i:i+kh, j:j+kw]
                # Multiply with kernel and sum
                result = (patch * kernel).sum((-2, -1))
                output[i, j] = result.numpy()[0, 0]
        
        # Convert back to tensor
        return Tensor(output)

    def sobel_edges(self, image: Tensor, blur: bool = True, 
                    blur_kernel_size: Optional[int] = None,
                    blur_sigma: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply Sobel edge detection with optional Gaussian blur"""
        # Ensure input is float32
        if not isinstance(image.numpy().dtype, np.float32):
            image = Tensor(image.numpy().astype(np.float32))
            
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
        grad_x = self.apply_filter(image, self.sobel_x)
        grad_y = self.apply_filter(image, self.sobel_y)
        
        # Compute gradient magnitude
        magnitude = (grad_x ** 2 + grad_y ** 2).sqrt()
        
        return magnitude, grad_x, grad_y

    def prewitt_edges(self, image: Tensor, blur: bool = True,
                      blur_kernel_size: Optional[int] = None,
                      blur_sigma: Optional[float] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply Prewitt edge detection with optional Gaussian blur"""
        # Ensure input is float32
        if not isinstance(image.numpy().dtype, np.float32):
            image = Tensor(image.numpy().astype(np.float32))
            
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
        grad_x = self.apply_filter(image, self.prewitt_x)
        grad_y = self.apply_filter(image, self.prewitt_y)
        
        magnitude = (grad_x ** 2 + grad_y ** 2).sqrt()
        
        return magnitude, grad_x, grad_y

    def laplacian_edges(self, image: Tensor, blur: bool = True,
                        blur_kernel_size: Optional[int] = None,
                        blur_sigma: Optional[float] = None) -> Tensor:
        """Apply Laplacian edge detection with optional Gaussian blur"""
        # Ensure input is float32
        if not isinstance(image.numpy().dtype, np.float32):
            image = Tensor(image.numpy().astype(np.float32))
            
        if blur:
            image = self.gaussian_blur(image, blur_kernel_size, blur_sigma)
        return self.apply_filter(image, self.laplacian)

    def non_maximum_suppression(self, magnitude: Tensor, direction: Tensor) -> Tensor:
        """Apply non-maximum suppression to thin edges"""
        # Convert to numpy for easier processing
        direction = direction.numpy()
        magnitude = magnitude.numpy()
        
        # Quantize directions to 0, 45, 90, 135 degrees
        direction = ((direction + np.pi) * 4 / np.pi + 0.5).astype('int') % 8
        
        # Initialize output as zeros
        output = np.zeros_like(magnitude, dtype=np.float32)
        
        # For each pixel, check if it's a local maximum along gradient direction
        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                if direction[i, j] in [0, 4]:  # Horizontal
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif direction[i, j] in [2, 6]:  # Vertical
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                elif direction[i, j] in [1, 5]:  # Diagonal 1
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                else:  # Diagonal 2
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                
                if magnitude[i, j] >= max(neighbors):
                    output[i, j] = magnitude[i, j]
        
        return Tensor(output)

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
        # Ensure input is float32
        if not isinstance(image.numpy().dtype, np.float32):
            image = Tensor(image.numpy().astype(np.float32))
        
        # 1. Apply Gaussian smoothing
        smoothed = self.gaussian_blur(image, kernel_size=int(6*sigma), sigma=sigma)
        
        # 2. Compute gradients using Sobel
        magnitude, grad_x, grad_y = self.sobel_edges(smoothed, blur=False)
        direction = grad_y.arctan2(grad_x)
        
        # 3. Non-maximum suppression
        suppressed = self.non_maximum_suppression(magnitude, direction)
        
        # Convert to numpy for thresholding operations
        suppressed_np = suppressed.numpy()
        
        # 4. Double thresholding
        high_mask = suppressed_np > high_threshold
        low_mask = suppressed_np > low_threshold
        
        # 5. Edge tracking by hysteresis
        output = np.zeros_like(suppressed_np, dtype=np.float32)
        output[high_mask] = 1
        
        # Iterate through pixels between thresholds
        weak_edges = (low_mask & ~high_mask)
        rows, cols = np.where(weak_edges)
        
        for r, c in zip(rows, cols):
            # Check 8-connected neighbors
            if np.any(output[max(0, r-1):min(r+2, output.shape[0]),
                            max(0, c-1):min(c+2, output.shape[1])] == 1):
                output[r, c] = 1
        
        return Tensor(output)

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
