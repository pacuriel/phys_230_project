# Contact Inhibition of Locomotion in Zebrafish Cells

This project simulates and analyzes collective cell migration with and without contact inhibition of locomotion (CIL),
using centroid data extracted from experimental time-lapse images.

## Structure
- `scripts/` contains Python tools to simulate, analyze, and visualize CIL behavior.
- `data/` holds input TIFFs, centroids, and track CSVs.
- `results/` stores all plots, GIFs, and trajectory outputs.

## Quick Start
```bash
conda env create -f environment.yml
conda activate cil-env
python scripts/zebrafish_cil_sim.py --help
```
