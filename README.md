# Project03_46W38  
Wind power forecasting by machine learning  

## Dataset Description

The project uses an hourly dataset of wind and power production from **4 turbine sites**, covering:  
**02.01.2017 → 31.12.2021**,  
with an **open “CC0 1.0 Universal” license**.

### Available Variables
- Time — `YYYY-MM-DD HH:mm:ss`
- temperature_2m — °F @ 2 m
- relativehumidity_2m — %
- dewpoint_2m — °F
- windspeed_10m — m/s @ 10 m
- windspeed_100m — m/s @ 100 m
- winddirection_10m — deg (0–360) @ 10 m
- winddirection_100m — deg (0–360) @ 100 m
- windgusts_10m — m/s
- power — % of Prated (normalized)

---

## ML Pipeline Overview

Steps:

1. **load_data**  
2. **preprocess_data**  
3. **train/test split — 80% train, 20% test**  
4. **train_model**  
5. **test_model**  
6. **evaluation metrics**
   - **Plots**
     - Scatter plot: `y_true vs y_pred`
     - Error distribution
   - **Numerical metrics (regression)**
     - MSE — Mean Squared Error  
     - RMSE — Root Mean Square Error  
     - MAE — Mean Absolute Error  
     - R² — coefficient of determination  
7. **save_model**

---

## Pipeline Diagram (Mermaid)

```mermaid
flowchart TB
    subgraph INPUT["Dataset 2017–2021"]
        A["Wind and Power Observations - Hourly - 4 sites"]
    end

    subgraph PIPELINE["ML Pipeline"]
        B["Load Data"]
        C["Preprocess Data - cleaning and feature engineering"]
        D["Train Test Split 80-20"]
        E["Train Model"]
        F["Test Model"]
    end

    subgraph EVAL["Evaluation"]
        G1["Scatter Plot: y_true vs y_pred"]
        G2["Error Distribution"]
        H1["MSE"]
        H2["RMSE"]
        H3["MAE"]
        H4["R2"]
    end

    J["Save Model"]

    A --> B --> C --> D --> E --> F --> EVAL --> J

