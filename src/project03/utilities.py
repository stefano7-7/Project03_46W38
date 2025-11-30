""""SUPPORT FUNCTIONS"""

import tkinter as tk
from tkcalendar import DateEntry

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def ask_value(prompt, choices, default):
    """ask User for a value from a list of choices"""
    text = input(f"{prompt} {choices} (default = {default}): ")
    if text == "":  # user presses ENTER
        return default
    if text in choices:
        return text
    print("value not valid")
    return ask_value(prompt, choices, default)


def choose_from_list(list_to_show):
    """GUI with a list for the User to choose from"""
    # create GUI window
    root = tk.Tk()
    root.title('Select')
    list_showed = tk.Listbox(root, selectmode=tk.MULTIPLE)
    # map(func, list) calls lambda (anoymous func with implicit return) to insert
    # to insert every element c of list_to_show
    # at the end of the list in the GUI
    list(map(lambda c: list_showed.insert(tk.END, c), list_to_show))
    list_showed.pack(padx=10, pady=10)
    chosen = []
    tk.Button(root, text='OK', command=lambda: (chosen.extend(
        list_to_show[i]for i in list_showed.curselection()), root.destroy())).pack(pady=5)
    root.mainloop()
    return chosen


def choose_time_window(start_date, end_date):
    """
    asks User to selcet a time window for further use 
    (e.g. time series plot)
    start_date: first allowed date
    end_date: last allowed date
    """
    print(f'Select a date between {start_date} '
          f'and {end_date}')
    root = tk.Tk()
    root.title('Select time window')
    tk.Label(root, text="Start date").pack(pady=(10, 0))
    start_cal = DateEntry(root,
                          mindate=start_date.date(),
                          maxdate=end_date.date(),
                          date_pattern="yyyy-mm-dd")
    start_cal.pack(pady=5)

    tk.Label(root, text="End date").pack(pady=(10, 0))
    end_cal = DateEntry(root,
                        mindate=start_date.date(),
                        maxdate=end_date.date(),
                        date_pattern="yyyy-mm-dd")
    end_cal.pack(pady=5)

    result = {}

    def on_ok():
        result["start"] = start_cal.get_date()
        result["end"] = end_cal.get_date()
        root.destroy()

    tk.Button(root, text="OK", command=on_ok).pack(pady=15)

    root.mainloop()
    return result.get("start"), result.get("end")


def prediction_metrics(y_known, y_estimated, unit):
    """
    calculate & print metrics
    MSE (emphasis on large error), RMSE (MSE in same units), MAE (outliers)
    """
    mse = mean_squared_error(y_known, y_estimated)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_known, y_estimated)
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f} [{unit}] MAE: {mae:.2f}")
