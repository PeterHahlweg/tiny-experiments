# TinyGrad Experiments

A collection of small experiments to learn and explore [tinygrad](https://docs.tinygrad.org), a simple deep learning framework.

## Current Experiments

### Canny Edge Detection
An implementation of the Canny edge detection algorithm using tinygrad tensors and operations. This experiment demonstrates:
- Basic tensor operations
- Convolution layers
- Image processing fundamentals
- Gradient computations

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the edge detection:
```bash
python edge.py --input your_image.jpg --output-dir output
```

Or try the built-in test image:
```bash
python edge.py --use-test-image --output-dir output
```

## License

MIT
