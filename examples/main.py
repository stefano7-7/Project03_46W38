from project03 import WindForecast, ask_value

prediction = WindForecast()

site_number = ask_value("Choose site number",
                        ["1", "2", "3", "4"], "1")

prediction.load_data(site_number)
prediction.plot_timeseries()
prediction.split()
prediction.train_ml_model()
prediction.test_ml_model()
