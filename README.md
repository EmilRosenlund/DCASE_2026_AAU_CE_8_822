# DCASE Task 2 Project

This repository contains our 8th semester project for DCASE Task 2: noise-aware unsupervised anomalous sound detection for machine condition monitoring.

The code is organized around three complementary model tracks, each trained in a slightly different way, followed by a fusion and ensemble stage that combines their outputs into one final system.

## Project Idea

The goal is not to rely on a single detector. Instead, the system learns several views of the same audio problem and lets them work together at inference time. Some parts focus on representation learning, some on pretrained embeddings, and some on machine-specific structure.

## Three Model Tracks

- Audio Encoder: a 2-channel + residual autoencoder track for learning compact latent structure.
- BEATs: a pretrained embedding track for transfer-style representation learning.
- SSLAM: a stereo embedding track for channel-aware feature extraction.

Each track can be trained independently, tuned separately, and evaluated on its own.

## Training Approaches

Different models in the repository use different training styles:

- Reconstruction-based training for autoencoder-style learning.
- Clustering and pseudo-label training for shaping latent spaces.
- Fine-tuning and embedding extraction for pretrained backbones.
- Variant-specific setups for different machines and input configurations.

This keeps the experiments modular while still letting them contribute to the same overall system.

## Fusion and Ensemble

The final stage combines the model outputs in a higher-level decision layer.

- Fusion merges complementary embedding streams or scores.
- Ensemble combines multiple detectors to improve robustness.

The aim is to keep the individual models simple and independent, then let the final decision benefit from their different strengths.

## Repository Layout

- `audio_encoder/` - audio encoder experiments, model parts, and dataset utilities.
- `src/` - pipeline orchestration, embedding modules, and classification logic.
- `training/` - backbone training code and supporting training utilities.
- `beats/` - BEATs model code and helpers.

## Project Report

The full report for this project is included in `P8.pdf`.
