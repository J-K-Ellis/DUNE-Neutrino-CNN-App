
# Project App

This is a Python-based GUI app using Tkinter, designed to visualize, process, and analyze data for a scientific or detector project. It dynamically loads pages, backends, and helper functions for modular and flexible usage.

---

## Project Structure

```
project_root/
│
├── main.py
├── Imports/
│   └── common_imports.py
│
├── Backend/
│   ├── __init__.py
│   ├── pdg_id_script.py
│   └── ...
│
├── Core_Functions/
│   ├── __init__.py
│   └── ...
│
├── Helpers/
│   ├── __init__.py
│   ├── Frame_Manager_Script.py
│   └── ...
│
└── Pages/
    ├── __init__.py
    ├── StartPage.py
    ├── ...
```

---

## Features

* **Dynamic Windowss/Pages:** Pages are loaded automatically from the `Pages` folder. You can add new ones easily, and they’ll just work.
* **Automatic Module Imports:** The app grabs functions and classes from `Backend`, `Core_Functions`, and `Helpers` and makes them available to use anywhere in the app.
* **Data Directory Selection:** When you run the app, you’re prompted to choose a data directory. The app looks for `.hdf5` files (or similar) and sorts them for you.
* **Plotting:** Uses `matplotlib` with a color-blind-friendly palette and interactive plotting features.
* **Frame Management:** You can switch between pages, refresh them, and even destroy old ones dynamically.
* **Resizable Window:** The app automatically adjusts its window size based on the active page content.

---

## App Class

### Initialization

```python
App(Data_Directory='', input_type='edep', det_complex='2x2')
```

**Parameters:**

* `Data_Directory` – Directory path containing your input data files.
* `input_type` – Type of input, default is `'edep'`.
* `det_complex` – Set the detector complexity, default is `'2x2'`.

### Some Key Methods

* `show_frame(page_identifier)` – Shows the specified page. Can be a class or page name.
* `_destroy_frame(frame_class)` – Destroys a page frame.
* `_reinitialize_frame(frame_or_name)` – Destroys and re-creates a page.
* `_resize_to(frame)` – Dynamically resizes the window to fit the current page.

---

## Page System

Pages are defined in the `Pages` folder. Each page should inherit from `tk.Frame` and expect two arguments in its constructor: the parent container and the controller (which is the `App`).

Example:

```python
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        tk.Label(self, text="Welcome to the Msci Project App!").pack()
```

Pages can implement a `refresh_content()` method to refresh their content when they’re shown.

---

## Running the App

### To run:

```bash
python main.py
```

You’ll be prompted to select a data directory. Once you select it, the app will initialize and start up.

---

## Developer Notes

* The app uses `matplotlib.use('agg')` for headless rendering, meaning it’s safe to run on servers without a display.
* The `Backend`, `Core_Functions`, and `Helpers` folders must each define an `__all__` list for the auto-import to work correctly.
* The app expects that your data files have a numeric ID as the fourth element in the filename (e.g., `data.sample.001.hdf5`).


---



---

That’s it! If you have any questions or want to contribute, feel free to open an issue or submit a PR.
