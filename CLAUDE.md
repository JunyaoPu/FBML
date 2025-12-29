# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FBML (Flappy Bird Machine Learning) uses genetic algorithms and neural networks to evolve AI-controlled birds that learn to play Flappy Bird. Features a full menu system with model save/load, configurable training parameters, and training visualization.

## Running the Project

```bash
uv run python main_flappy_bird.py
```

## Architecture

### main_flappy_bird.py
Central game file containing pygame loop, menu system, and training logic. Key game states:
- `STATE_MENU` → `STATE_MODEL_SELECT` → `STATE_TRAIN_SETTINGS` → `STATE_TRAIN`
- `STATE_MENU` → `STATE_MODEL_SELECT` → `STATE_RUN_MODEL`
- `STATE_MENU` → `STATE_PLAY` (human play)

Game constants: FPS=600 (training), screen=288x512, pipe gap=100px.

### Genetics/birdnetclass.py
Neural network (BirdNet) for bird decisions:
- **Input**: 3 nodes (dx to pipe, dy to gap center, velocity) normalized to [-1, 1]
- **Hidden**: Configurable layers, default [4]
- **Output**: 1 node with sigmoid, flap if > 0.5
- Weights initialized in [-1, 1]

### Genetics/populationclass.py
Population management and evolution:
- **Selection**: Top N% survive as parents (configurable parent_fraction)
- **Breeding**: Clone parents into children with weighted selection (better parents more likely)
- **Mutation**: Gaussian noise on 10% of weights, rate scales by parent rank (0.5x for best, 1.5x for worst)
- **Elitism**: Best-ever bird preserved unmutated in slot 0

### Training Settings (configurable in-game)
- Population size (default 50)
- Mutation rate (default 0.1)
- Parent fraction (default 30%)
- Network structure (default [4])
- Runs per bird (default 3) - multi-run evaluation for fitness
- Fitness method: avg, min, geo (geometric mean), harm (harmonic mean)

Settings persist to `training_settings.json`.

### Model Storage
- Models saved as `.npz` (weights) + `.json` (settings) in `models/`
- Training outputs (CSV logs, PNG graphs) in `outputs/YYYY-MM-DD/`

## Key Implementation Details

- When continuing training from a saved model, network structure is locked (inferred from weights)
- Population initialization from model: bird 0 = exact copy, birds 1-N = mutated clones
- Fitness = distance traveled (evaluated over multiple runs, aggregated by selected method)
- Window is resizable with dynamic scaling
