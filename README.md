# Santander Cycles Demand Forecasting (SVR)

Predicting daily Santander Cycles trip counts across London boroughs from weather and calendar features, using RBF-kernel Support Vector Regression in R.

> Part of the CS5812 Predictive Data Analysis group coursework, MSc AI, Brunel University London. Companion model: [1D CNN](https://github.com/emrekutlusan/SantanderRegression-1D-CNN).

## Highlights

- **Test R^2 0.725** on unseen 2018 data, using weather and calendar features only
- Log-transforming the skewed target raised R² from **0.58 to 0.72**
- Temporal split (train 2014–2017, test 2018) and leakage control (`total_duration` excluded)
- Commuter boroughs (e.g. City of London) proved far more predictable than tourist-heavy ones (e.g. Westminster)

## Data

22,848 daily observations (2014–2018) built from [TfL cycle hire data](https://cycling.data.tfl.gov.uk/), [Open-Meteo](https://open-meteo.com/) weather at borough level, and gov.uk bank holidays.

## Results

| Metric | Train | Test |
|---|---|---|
| MAE | 520.20 | 716.15 |
| RMSE | 802.18 | 1103.31 |
| R² | 0.859 | **0.725** |
| MAPE | – | 31.42% |

**Model:** ε-SVR, RBF kernel, C = 10, γ = 0.1, ε = 0.1, 17 standardised features.

## Run

```r
install.packages(c("tidyverse", "e1071", "caret"))
```

Place `combined_daily_data.csv` in the project root and run `svm_model.R`. The script prints metrics and saves the predictions, borough-level performance and a results summary as CSV files.

## Author

**Emre Kutlusan** · MSc AI, Brunel University London
Group work with Akan Utku Efe and Burak Can Soyak. My part: weather data pipeline, SVR and 1D CNN models.
