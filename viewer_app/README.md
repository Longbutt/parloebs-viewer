# Parløbs Analyse Viewer

Interactive Streamlit app for uploading two FIT rides, aligning timestamps, and exploring power/speed plots, intersections, and segment stats. Built from `DataViewerOperations.ipynb`.

## Quick run (needs Python installed)
```powershell
cd viewer_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The browser opens automatically; use the sidebar to upload two FIT files and adjust options.

## Build a standalone .exe (no Python needed to run)
This wraps Streamlit in a single executable using PyInstaller. Build once on a Windows machine with Python installed; distribute the resulting `dist\run_app.exe` to machines without Python.

1) Install build deps (one time):
```powershell
pip install -r requirements.txt
pip install pyinstaller
```
2) Create the exe:
```powershell
cd viewer_app
pyinstaller --onefile --noconfirm --hidden-import streamlit.web.cli run_app.py
```
3) Run on any Windows machine:
```powershell
dist\run_app.exe
```
It will start a local Streamlit server and open your default browser. Keep `dist\run_app.exe` in the same folder as your FIT files, or browse to them from the upload widgets.

Notes:
- The executable is larger because it bundles Python and all dependencies.
- First launch may take a few seconds while the bundled server starts.
- If Windows SmartScreen warns, choose “More info” → “Run anyway” (you are the publisher).
