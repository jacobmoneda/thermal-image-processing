# Thermal Image Processing

Thermal image processing and visualisation software developed for the Spark Robotics Digital Twin and Thermal Analytics Project.

The system processes original UNI-T UTi260E thermal images, extracts per-pixel temperature data, associates images with robot arm inspection poses, and displays the results in an interactive Streamlit dashboard.

## Features

- Extracts the full **256 × 192 thermal temperature matrix** from UTi260E JPG files
- Extracts the embedded **RGB image**
- Calculates:
  - Minimum temperature
  - Maximum temperature
  - Mean temperature
  - Median temperature
  - Centre temperature
  - Maximum temperature above median (ΔT)
- Generates thermal heatmaps and CSV files
- Supports LOW, MID and HIGH robot inspection positions
- Can associate thermal images with poses using `pose_log.csv`
- Provides per-cabinet analysis and LOW/MID/HIGH comparison
- Interactive dashboard views:
  - RGB
  - Thermal
  - Ironbow
  - Grayscale
  - Hotspot
  - Delta T
  - Contour
  - RGB + Thermal
- Hotspot threshold and temperature distribution analysis

## Installation

```bash
pip install numpy matplotlib pandas streamlit plotly
```

## Processing Images

Process a cabinet folder directly:

```bash
python process.py inputs/CAB01
```

Or process the images through the dashboard.

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard allows a cabinet to be selected, processed and analysed using both cabinet-level and individual thermal image statistics.

## Inspection Workflow

```text
Go2 + D1 Arm
      ↓
LOW / MID / HIGH inspection
      ↓
UTi260E timelapse images
      ↓
process.py
      ↓
Temperature + RGB extraction
      ↓
Cabinet analysis
      ↓
Streamlit dashboard
```

This project is intended to support repeatable thermal inspection of battery charging/housing cabinets and provide processed thermal information for later integration with digital-twin visualisation.
