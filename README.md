# Insurance Cost Prediction

Predicting health insurance premiums using individual health profiles and demographic data.

## Project Overview

Insurance companies need to move beyond broad actuarial tables toward individualized premium pricing. This project uses machine learning to predict insurance premiums based on personal health and demographic features, enabling more accurate and equitable pricing.

## Project Structure

```
├── notebooks/
│   ├── 01_EDA_and_Visualization.ipynb    # Block 1 & 2: EDA, visualization, hypothesis testing
│   └── 02_ML_Modeling.ipynb              # Block 3: Preprocessing, modeling, evaluation
├── data/
│   └── insurance_data.csv                # Source dataset
├── models/
│   └── best_model.pkl                    # Trained model artifact
├── app/
│   ├── app.py                            # Streamlit web calculator (Block 4)
│   └── requirements.txt                  # App dependencies
├── tableau/
│   └── screenshots/                      # Tableau dashboard screenshots
└── README.md
```

## Dataset

986 records with 11 features including age, health conditions (diabetes, blood pressure, transplants, chronic diseases), body metrics (height, weight), surgical history, and family health history.

**Target variable:** PremiumPrice (continuous, range: 15,000–40,000)

## Approach

1. **Exploratory Data Analysis** — Distribution analysis, correlation mapping, outlier detection
2. **Hypothesis Testing** — Statistical validation of premium differences across health conditions
3. **ML Modeling** — Baseline (Linear Regression) → Random Forest → Gradient Boosting with cross-validation
4. **Deployment** — Streamlit web app for real-time premium estimation

## Key Findings

_To be updated after analysis._

## How to Run

### Notebooks
Open in Google Colab or run locally with Jupyter.

### Streamlit App
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Author

Pallabi Roy Singh
