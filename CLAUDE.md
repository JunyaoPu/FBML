# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FBML (Flappy Bird Machine Learning) is a Python project that uses genetic algorithms and neural networks to evolve AI-controlled birds that learn to play Flappy Bird.

## Running the Project

```bash
# Run with uv (recommended)
uv run python main_flappy_bird.py

# Or install dependencies and run directly
uv sync
python main_flappy_bird.py
```

The game runs autonomously - no user input needed after start. Console outputs generation statistics (mean/std fitness).

## Architecture

### Core Components

**main_flappy_bird.py** - Pygame game loop with Flappy Bird mechanics. Integrates with the genetic algorithm population. Key constants: FPS=600, pipe gap=100px, screen=288x512.

**Genetics/birdnetclass.py** - Neural network for bird decision-making:
- Input: 2 nodes (relative X/Y to nearest pipe, normalized)
- Hidden: [100, 20, 5] layers
- Output: 2 nodes (flap if output[0] > output[1])
- Weights initialized in [-1, 1]

**Genetics/populationclass.py** - Population management and evolution:
- Population size: 10 birds (default)
- Selection: Top 20% survive as parents
- Crossover: Random weight selection from parents
- Mutation rate: 20%

**Genetics/evolver.py** - Alternative evolution implementation with 10% parent fraction and 10% mutation probability.

**Bird.py** - Legacy simple bird class (deprecated, replaced by BirdNet).

### Game Loop Flow

1. Create population of 10 birds with random neural networks
2. Each generation: run game until all birds crash
3. Fitness = distance traveled
4. Evolve: sort by fitness → select parents → crossover → mutate
5. Print generation stats and repeat

## TODOs in Codebase

- `main_flappy_bird.py:10-11` - Generate bird population outside game loop; find way to restart when all birds crash
- `birdnetclass.py:15-16` - Normalize propagation and use sigmoid function
