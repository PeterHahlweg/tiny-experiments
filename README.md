# tinygrad experiments

A collection of small experiments to learn and explore [tinygrad](https://docs.tinygrad.org).

## Getting Started

```bash
# Set up experiments repo
git clone https://github.com/PeterHahlweg/tiny-experiments.git
cd tiny-experiments

# Set up virtual env
uv venv
source .venv/bin/activate.fish

# Install tinygrad
cd ~/github/tinygrad/
set -e PYTHONPATH
set -x PYTHONPATH ~/github/tinygrad/extra $PYTHONPATH
uv pip uninstall tinygrad
uv pip install -e .

# Run test
python examples/edge_detection/edge.py --use-test-image --output-dir output
```

## License

MIT
