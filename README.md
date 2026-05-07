# Insurance Cost Prediction

Predicting health insurance premiums using individual health profiles and demographic data.

## Problem Statement

Insurance companies need to move beyond broad actuarial tables toward individualized premium pricing. This project uses machine learning to predict insurance premiums based on personal health and demographic features, enabling more accurate and equitable pricing.

**Target Metric:** R² Score — measuring how well the model explains premium variation across individuals.

## Dataset

- **Records:** 986
- **Features:** 10 input features + 1 target (PremiumPrice)
- **Target Variable:** PremiumPrice (continuous, range: 15,000–40,000)
- **Missing Values:** None

## Project Structure

```
├── notebooks/
│   ├── 01_EDA_and_Visualization.ipynb    # EDA, visualization, hypothesis testing
│   ├── 02_ML_Modeling.ipynb              # Preprocessing, modeling, evaluation
│   └── 03_Post_Model_Analysis.ipynb      # SHAP, error analysis, business impact
├── data/
│   └── insurance.csv                     # Source dataset
├── models/
│   ├── best_model.pkl                    # Trained Random Forest model
│   ├── scaler.pkl                        # StandardScaler
│   └── feature_names.pkl                 # Feature list for input validation
├── streamlit_app/
│   ├── app.py                            # Streamlit web calculator
│   └── requirements.txt                  # App dependencies
├── tableau/
│   └── screenshots/                      # Tableau dashboard screenshots
└── README.md
```

## Approach

### 1. Exploratory Data Analysis

- Premium distribution is multimodal with three peaks around 15,000, 23,000, and 28,000–29,000 — suggesting distinct premium tiers
- Age is the dominant factor: premiums nearly double from youngest group (16,212) to oldest (28,788)
- AnyTransplants shows the largest premium gap among health conditions (+7,866)
- KnownAllergies and Diabetes have negligible impact on premiums

### 2. Hypothesis Testing

- **T-tests:** 5 out of 6 health conditions showed statistically significant premium differences (p < 0.05). KnownAllergies was not significant (p = 0.704).
- **ANOVA:** Number of surgeries significantly affects premiums (F = 26.14, p ≈ 0), with a clear staircase from 0 to 2 surgeries.
- **Key insight:** Diabetes is statistically significant (p = 0.017) but the premium difference (+964) is practically negligible — less than 4% of the mean premium.

### 3. Machine Learning Modeling

Four models were trained and compared:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 3,494 | 2,586 | 0.714 |
| Decision Tree | 4,008 | 1,288 | 0.623 |
| Random Forest | 2,135 | 1,010 | 0.893 |
| Gradient Boosting | 2,383 | 1,522 | 0.867 |

**Best Model:** Tuned Random Forest
- R² = 0.905
- MAE = 1,007
- RMSE = 2,014

**Feature Importance:**
- Age dominates with 67.7% importance
- AnyTransplants is the most important health condition (10.3%)
- Diabetes (0.1%) and KnownAllergies (0.1%) are essentially ignored by the model

### 4. Post-Model Analysis

- **SHAP Values:** Age shifts individual predictions by up to ±10,000. Transplant status always pushes predictions upward.
- **Error Analysis:** Worst predictions occur at age extremes. Model is consistently off by ~2,200 for transplant patients.
- **Learning Curves:** Mild overfitting (gap = 0.097). Validation curve still rising — more data would improve performance.
- **Prediction Intervals:** 95% coverage achieved with average width of 5,455 — well-calibrated uncertainty estimates.
- **Business Impact:** Model reduces pricing error by 81% compared to using the average premium for everyone. For 10,000 policies, total pricing error drops by approximately 43 million.

### 5. Deployment

The trained model is deployed as a Streamlit web application where users can input their health and demographic details to receive a real-time premium prediction.

**Live App:** [Insurance Premium Predictor](https://insurance-cost-prediction-by-pallabi-roy-singh.streamlit.app)

## Key Insights and Recommendations

1. **Age is the primary premium driver** — any pricing model that doesn't heavily weight age is likely underperforming
2. **Transplant history** is the most impactful health condition (+7,866 premium increase), representing a distinct high-risk subgroup
3. **Diabetes and allergies** have minimal impact — insurers using these as major pricing factors may be adding unnecessary complexity
4. **Feature engineering doesn't always help** — BMI performed worse than raw Weight as a predictor
5. **Statistical significance ≠ practical significance** — BloodPressureProblems was significant in testing but nearly useless in the model due to multicollinearity with Age
6. **Collect more data** — learning curves show the model hasn't plateaued
7. **Use prediction intervals** — communicate uncertainty (±2,700 average) instead of false single-point precision

## Links

- **Tableau Dashboard:** [Insurance Cost Prediction Dashboard](https://public.tableau.com/app/profile/pallabi.roy.singh/viz/InsuranceCostPrediction_17765428386570/DemographicInsights)
- **Live App:** [Insurance Premium Predictor](https://insurance-cost-prediction-by-pallabi-roy-singh.streamlit.app)
- **Technical Blog:** https://medium.com/@pallabiroysingh/predicting-insurance-premiums-with-machine-learning-a-complete-walkthrough-10e2f3e0fbf1 

## How to Run

### Notebooks
Open in Google Colab or run locally with Jupyter.

### Streamlit App
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Author

Pallabi Roy Singh
