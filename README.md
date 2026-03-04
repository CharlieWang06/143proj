## Project Structure

```text
|-- pro.ipynb
|-- requirements.txt
|-- README.md
|-- run_pipeline.py
|-- data/
|-- outputs/
`-- diabetes_pipeline/
    |-- __init__.py
    |-- data.py
    |-- preprocess.py
    |-- eda.py
    `-- model.py
```

## Scripts Description

- `data.py`: dataset locating/downloading and CSV loading
- `preprocess.py`: deduplication, BMI casting, feature selection, NearMiss balancing
- `eda.py`: plotting utilities for all notebook EDA figures and confusion matrix
- `model.py`: train-test split, scaling, logistic regression training, evaluation
- `run_pipeline.py`: executable pipeline entrypoint

## How To Run

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline from the project root:

```bash
python run_pipeline.py
```

4. Optional: skip EDA plots (faster run):

```bash
python run_pipeline.py --skip-eda
```

Generated plots are saved in `outputs/` and include:
- `histograms.png`
- `correlation_heatmap.png`
- `target_correlation_bar.png`
- `highbp_highchol_stacked.png`
- `diabetes_pie.png`
- `bmi_countplot.png`
- `genhlth_bar.png`
- `age_bar.png`
- `education_kde.png`
- `income_kde.png`
- `confusion_matrix.png`

Classification report is printed in terminal.

## Third-Party Modules Used

- `kagglehub`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `imbalanced-learn` (`imblearn`)
- `scikit-learn` (`sklearn`)
