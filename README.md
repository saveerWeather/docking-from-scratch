# Understanding Protein–Ligand Docking by Ablation

**Implementing and Fitting a Docking Scoring Function from Scratch**

CS279 Final Project — Saveer Jain (saveerja)

## Project Relevance

The project works directly on 3D structures of protein–ligand complexes. I will implement a docking scoring function and a search over the pose of a ligand within a protein binding site.

## Dataset

The [Astex Diverse Set](https://doi.org/10.1021/jm061277y) (ADS) is a well-known benchmark dataset of 85 high-resolution protein–ligand crystal structures used for evaluating docking functions.

## Methods

I will implement a docking program from scratch in Python as well as a Monte Carlo search with local minimization over ligand pose. I will fit the weights of each term in the scoring function by regression against measured affinities on a training split, and then ablate each term (refitting the remainder) to measure how much each contributes. AutoDock Vina will be an external baseline.

Scoring function terms to be implemented and ablated:

- Van der Waals
- Electrostatics
- Hydrogen bonding
- Desolvation
- Torsion effects
