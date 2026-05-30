# Kleinberg Topology Experiment

## What changed from the original code?
1. **graph.py**: Added a custom `kleinberg_ws_graph` function. This replaces the standard Watts-Strogatz random rewiring with a distance-biased exponential decay model.
2. **main.py**: Added the `Kleinberg` flag to the command-line arguments to trigger the new graph structure.

## The Goal
To prove that by restricting long-range connections based on spatial distance (a clustering exponent of alpha=2.0), the Spiking Neural Network can achieve high biological plausibility and accuracy while massively reducing the GPU memory (VRAM) footprint compared to purely random (ER/GNM) or Hub (BA) networks.
