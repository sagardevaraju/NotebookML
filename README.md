# NotebookOrchestrator

End-to-end template machine learning workflow orchestrated with Jupyter notebooks and [Papermill](https://papermill.readthedocs.io/). The project demonstrates how to modularise a typical Kaggle-style pipeline (Titanic survival prediction) into reusable notebooks that can be parameterised and executed programmatically. The workflow supports switching between Pandas, Modin (Ray), and Dask dataframe engines, and uses Optuna for model hyperparameter optimisation.

## Repository structure

```
notebookml/              # Shared Python utilities (backend abstraction)
notebooks/               # Parameterised workflow notebooks
  01_data_preparation.ipynb
  02_feature_engineering.ipynb
  03_model_building.ipynb
  04_model_evaluation.ipynb
  05_orchestrator.ipynb   # Runs the entire pipeline via Papermill
requirements.txt         # Python dependencies
```

The pipeline stages persist their outputs in the `data/` and `models/` folders. These directories are ignored by default so that artefacts generated during execution are not tracked by Git.

## Getting started

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the orchestrator** to execute the full workflow with Papermill:

   ```bash
   papermill notebooks/05_orchestrator.ipynb notebooks/runs/latest/orchestrator-output.ipynb \
     -p engine pandas \
     -p modin_engine ray \
     -p dataset_url https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv \
     -p n_trials 20
   ```

   The orchestrator will sequentially execute all stage notebooks using the provided parameters. Outputs include:

   - Cleaned dataset (`data/processed.csv`)
   - Feature engineered train/test splits (`data/train_features.csv`, `data/test_features.csv`)
   - Optimised model artefacts (`models/random_forest.pkl`, `models/optuna_trials.csv`, `models/best_params.json`)
   - Evaluation metrics (`models/metrics.json`)

4. **Customise the execution backend** by changing the `engine` parameter to one of `pandas`, `modin`, or `dask`. Additional parameters (e.g. `test_size`, `random_state`, `n_trials`) can also be provided when calling Papermill.

## Notebook overview

| Notebook | Purpose | Key technologies |
| --- | --- | --- |
| `01_data_preparation.ipynb` | Downloads and cleans the Titanic dataset, deriving helper features and summaries. | BackendManager, Pandas/Modin/Dask |
| `02_feature_engineering.ipynb` | Generates model-ready features, train/test splits, and metadata. | Pandas, scikit-learn |
| `03_model_building.ipynb` | Tunes a RandomForestClassifier with Optuna and persists the trained model. | Optuna, scikit-learn |
| `04_model_evaluation.ipynb` | Computes standard classification metrics on the held-out test set. | scikit-learn |
| `05_orchestrator.ipynb` | Orchestrates the pipeline using Papermill. | Papermill |

## Extending the pipeline

- Add new feature engineering transformations in the second notebook and expose new parameters through Papermill.
- Swap the model in `03_model_building.ipynb` for alternative estimators or additional Optuna search spaces.
- Integrate experiment tracking, model registries, or deployment notebooks by extending the orchestrator sequence.

## Licensing

This project is released under the terms of the MIT License. See [LICENSE](LICENSE) for details.
