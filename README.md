# Used Vehicle Price Prediction

## Project Overview
This project develops a machine-learning workflow to estimate the price of a used vehicle from details such as its brand, model, model year, mileage, fuel type, transmission, engine specifications, exterior/interior colour, accident history, and clean-title status.

The project includes data cleaning, feature engineering, preprocessing, model comparison, hyperparameter tuning, evaluation, and a Streamlit web application for interactive predictions.

## Dataset
- **Source:** Kaggle — [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)
- **Working cleaned dataset:** 4,009 rows and 16 columns.
- **Original fields:** `brand`, `model`, `model_year`, `milage`, `fuel_type`, `engine`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title`, and `price`.
- **Missing values in the original data:** `fuel_type` (170), `accident` (113), and `clean_title` (596).
- **Duplicate rows:** None were found in the initial duplicate check.

The dataset contains a highly right-skewed price distribution. The median price was 31,000, while the mean was 44,553 and the maximum was 2,954,083. A small number of very expensive vehicles can strongly affect error metrics.

## Workflow
1. **Load and inspect the data:** Review columns, data types, missing values, duplicates, and summary statistics.
2. **Clean data:** Convert string-formatted mileage and price values to numeric values and ensure model year is numeric.
3. **Engineer features:** Create `car_age` from model year and extract `engine_liters`, `horsepower`, and `cylinders` from the engine description.
4. **Split the data:** Use an 80/20 train-test split with `random_state=42` (3,207 training rows and 802 test rows).
5. **Preprocess features:** Numeric columns use median imputation and standard scaling. Categorical columns use constant-value imputation (`Missing`) and one-hot encoding with unknown categories ignored.
6. **Train and compare models:** Evaluate Linear Regression, Ridge Regression, Decision Tree, Random Forest, and Gradient Boosting.
7. **Tune a model:** Run `RandomizedSearchCV` with five-fold cross-validation on a Random Forest using a log-transformed target.
8. **Evaluate and save:** Compare MAE, RMSE, and R², and save the selected baseline Random Forest pipeline for use by the app.
9. **Deploy locally:** Use Streamlit to collect vehicle details and display an estimated price.

## Model Selection and Results
The initial model comparison produced these test-set results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 17,192.81 | 133,047.05 | 0.1340 |
| Gradient Boosting | 20,170.42 | 133,351.76 | 0.1300 |
| Decision Tree | 20,455.82 | 135,396.01 | 0.1031 |
| Ridge Regression | 22,114.16 | 135,600.36 | 0.1004 |
| Linear Regression | 22,251.00 | 138,818.00 | 0.0572 |

**Selected model:** The baseline Random Forest was retained because it had the lowest MAE and RMSE and the highest R² among the initial comparison models.

A separate RandomizedSearchCV run tested a log-target Random Forest:
- Best cross-validation RMSE reported: **29,494.39**
- Best parameters: `n_estimators=100`, `min_samples_split=5`, `min_samples_leaf=1`, `max_features=1.0`, `max_depth=20`
- Test-set results: **MAE 17,283.71**, **RMSE 135,127.22**, **R² 0.1067**

The tuned model did not outperform the baseline Random Forest on the reported test metrics, so the baseline model was kept for the Streamlit application. The cross-validation RMSE and test RMSE are from different evaluation procedures and should not be compared as if they were the same measurement.

### Feature Importance
The leading features in the baseline Random Forest included:
- Mileage (`num__milage`): 0.3315
- Engine litres (`num__engine_liters`): 0.1242
- Lamborghini brand indicator: 0.1073
- Rolls-Royce brand indicator: 0.1020
- Horsepower (`num__horsepower`): 0.0800
- Car age (`num__car_age`): 0.0375

These are model feature-importance scores, not causal effects or percentages of price explained.

## Limitations
- The baseline Random Forest has a test R² of 0.1340, so the current model explains only a limited share of price variation.
- The price distribution is highly skewed, with a small number of very expensive vehicles. These outliers can increase RMSE substantially.
- Some useful information may be missing, such as vehicle condition, trim level, service history, location, seller type, and current market demand.
- Engine specifications are extracted from text, so inconsistent descriptions may lead to missing or imperfect engineered values.
- One-hot encoding can produce many features for high-cardinality fields such as brand and model, and unseen categories may not be represented well.
- The model is trained on a particular dataset and may not generalize to different markets, currencies, or time periods.
- The test set was used during model comparison and model selection. Therefore, the reported metrics should be treated as a practical benchmark, not a completely untouched final evaluation.
- The app's displayed currency symbol should be verified against the dataset's original documentation.

## Future Improvements
- Verify and document the dataset's currency, units, source, and collection date.
- Investigate extreme prices and evaluate robust handling of outliers without removing legitimate luxury vehicles blindly.
- Use cross-validation for model selection and reserve a final holdout test set for one-time evaluation.
- Try stronger gradient-boosting approaches such as XGBoost, LightGBM, or CatBoost, if available.
- Test alternative target transformations and report metrics both on the transformed and original price scale where appropriate.
- Improve engine-text parsing and assess the effect of missing engineered features.
- Add more relevant data, such as trim, condition, location, service history, and listing date.
- Add prediction intervals or uncertainty estimates so users can understand that the result is an estimate.
- Improve the Streamlit interface with input validation, clear currency/unit labels, and examples.
- Monitor model performance on newer listings and retrain when the market changes.

## Run the Streamlit App
Keep `app.py`, `used_vehicle_price_model.joblib`, and `requirements.txt` in the same folder.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal, usually `http://localhost:8501`.

## Project Files
A recommended repository structure is:

```text
Used-Vehicle-Price-Prediction/
├── app.py
├── used_vehicle_price_model.joblib
├── requirements.txt
├── README.md
├── Task03.ipynb
├── Task03_Report.pdf
└── Visualizations/
    ├── 1.png
    ├── 2.png
    └── 3.png
```


## Conclusion
This project demonstrates a complete introductory regression workflow, from data preparation and feature engineering through model comparison and a local Streamlit interface. Random Forest performed best among the initial models, but its R² indicates that further data and modelling improvements are needed before the predictions should be relied on for real purchasing decisions.
