# Project03_46W38
wind power forecasting by machine learning
|-- input: hourly record of wind and power production of 4 turbine sites from 02.01.017 to 31.12.2021, 
|   |-- openly available dataset with "CC0 1.0 Universal" license
|   |-- Time - hour of the day in format YYY-MM-dd HH:mm:ss
|   |-- temperature_2m - °F @ 2m asl
|   |-- relativehumidity_2m - % @ 2m asl
|   |-- dewpoint_2m - °F @ 2 asl
|   |-- windspeed_10m - m/s @ 10 m asl
|   |-- windspeed_100m - m/s @ 100 m asl
|   |-- winddirection_10m - Wind direction in degrees (0-360) at 10 m asl
|   |-- winddirection_100m - deg (0-360) @ 100 m asl
|   |-- windgusts_10m - m/s @ 10 m asl
|   |-- power - % of Prated (normalized power)
|
|-- load_data
|-- preprocess_data
|-- divide_in__80-20_subsets
|
|-- train_model on training data subset
|-- test_model on test data subset
|-- metrics on results
|   |-- plots 
|   |   |-- scatter plot y_true vs y_pred
|   |   |-- distribution of errors y_actual-ypredicted
|   |   |-- confusion matrix (tbd how to plly it here)
|   |-- numerical metrics
|   |   |-- MSE – Mean Squared Error (penalizing larger deviations y_actual-ypredicted)
|   |   |-- RMSE – Root Mean Squared Error (same unit of predicted, here % of Prated)
|   |   |-- MAE – Mean Absolute Error
|   |   |-- R² – how much the model captures
|   |-- classification of the model
|   |   |-- accuracy = % of correct predictions (in this case there is no inherent unbalance in data)
|   |   |-- precision (tbd in this case)
|   |   |-- recall / F1 (tbc if applicable here)
|   |   |-- R² – how much the model captures
|-- save model

flowchart project 3
    subgraph INPUT["Dataset 2017–2021"]
        A["Wind and Power Observations - Hourly - 4 sites"]
    end

    subgraph PIPELINE["ML Pipeline"]
        B["Load Data"]
        C["Preprocess Data - cleaning, feature engineering"]
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
