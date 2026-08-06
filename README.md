# Classification of Down Syndrome of Mice Protein Dataset

This project reproduces the analysis from the notebook as a modular Python pipeline. It loads the mice protein expression dataset, preprocesses missing values, trains several classification models, and explores feature-selection ideas using a custom genetic selection workflow.

## Project structure

- [pipeline.ipynb](pipeline.ipynb) – original notebook kept intact.
- [run_pipeline.py](run_pipeline.py) – entry point for the end-to-end modular workflow.
- [mice_protein_pipeline/](mice_protein_pipeline) – reusable modules for data loading, preprocessing, modeling, and feature selection.
- [requirements.txt](requirements.txt) – Python dependencies.

## Dataset

The workflow expects the file [Data_Cortex_Nuclear.csv](Data_Cortex_Nuclear.csv) in the project root.

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the modular pipeline:

   ```bash
   python run_pipeline.py
   ```

## What the pipeline does

- Loads the dataset and fills missing values with column means.
- Removes metadata columns and encodes the target label.
- Builds group-specific datasets for the notebook’s analysis scenarios.
- Trains baseline classifiers and reports accuracy.
- Trains a Random Forest model and an XGBoost model for the selected feature set.

## Notes

The notebook remains available as the original exploratory workflow, while the Python scripts make the workflow easier to run, extend, and maintain.
