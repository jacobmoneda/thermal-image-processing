import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from process import process_inspection


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Spark Thermal Analytics",
    layout="wide"
)

st.title(
    "Spark Thermal Analytics"
)


INPUT_ROOT = Path("inputs")
OUTPUT_ROOT = Path("outputs")

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not INPUT_ROOT.exists():

    st.warning(
        "The inputs/ directory does not exist."
    )

    st.stop()


# ============================================================
# FIND INSPECTIONS
# ============================================================

inspections = sorted([
    folder.name
    for folder in INPUT_ROOT.iterdir()
    if folder.is_dir()
])


if not inspections:

    st.warning(
        "No inspection folders found inside inputs/"
    )

    st.stop()


inspection = st.selectbox(
    "Select inspection",
    inspections
)


input_folder = (
    INPUT_ROOT
    / inspection
)

output_folder = (
    OUTPUT_ROOT
    / inspection
)


# ============================================================
# PROCESS BUTTON
# ============================================================

if st.button(
    "Process thermal images",
    type="primary"
):

    with st.spinner(
        "Processing UTi260E images..."
    ):

        results = process_inspection(
            input_folder,
            output_folder
        )

    successful_count = sum(
        1
        for result in results
        if result.get("status") == "OK"
    )

    st.success(
        f"Processed {successful_count} "
        f"of {len(results)} images successfully."
    )


# ============================================================
# LOAD SUMMARY
# ============================================================

summary_file = (
    output_folder
    / "inspection_summary.csv"
)


if not summary_file.exists():

    st.info(
        "Process this inspection to generate results."
    )

    st.stop()


summary = pd.read_csv(
    summary_file
)


# ============================================================
# SUMMARY TABLE
# ============================================================

st.subheader(
    "Inspection summary"
)


display_columns = [
    "source_file",
    "minimum_C",
    "maximum_C",
    "mean_C",
    "median_C",
    "center_C",
    "delta_max_C",
    "status"
]


# Only display columns that actually exist.
display_columns = [
    column
    for column in display_columns
    if column in summary.columns
]


st.dataframe(
    summary[display_columns],
    width="stretch"
)


# ============================================================
# SELECT SUCCESSFUL IMAGE
# ============================================================

successful = summary[
    summary["status"] == "OK"
]


if successful.empty:

    st.warning(
        "No successfully processed thermal images."
    )

    st.stop()


selected_file = st.selectbox(
    "Select thermal image",
    successful["source_file"]
)


row = successful[
    successful["source_file"]
    == selected_file
].iloc[0]


stem = Path(
    selected_file
).stem


# ============================================================
# STATISTICS
# ============================================================

st.subheader(
    selected_file
)


col1, col2, col3, col4, col5 = (
    st.columns(5)
)


col1.metric(
    "Minimum",
    f"{row['minimum_C']:.2f} °C"
)


col2.metric(
    "Maximum",
    f"{row['maximum_C']:.2f} °C"
)


col3.metric(
    "Mean",
    f"{row['mean_C']:.2f} °C"
)


if "median_C" in row:

    col4.metric(
        "Median",
        f"{row['median_C']:.2f} °C"
    )

else:

    col4.metric(
        "Center",
        f"{row['center_C']:.2f} °C"
    )


if "delta_max_C" in row:

    col5.metric(
        "Max above median",
        f"{row['delta_max_C']:.2f} °C"
    )

else:

    col5.metric(
        "Center",
        f"{row['center_C']:.2f} °C"
    )


# ============================================================
# LOAD TEMPERATURE MATRIX
# ============================================================

temperature_file = (
    output_folder
    / "temperature_csv"
    / f"{stem}_temperature.csv"
)


if not temperature_file.exists():

    st.error(
        f"Temperature CSV not found: "
        f"{temperature_file}"
    )

    st.stop()


temperature = np.loadtxt(
    temperature_file,
    delimiter=","
)


# ============================================================
# FIND RGB IMAGE
# ============================================================

rgb_path = None


# First try the path stored by process.py
if "rgb_image" in row:

    rgb_value = row["rgb_image"]

    if pd.notna(rgb_value):

        rgb_value = str(
            rgb_value
        ).strip()

        if rgb_value:

            possible_path = Path(
                rgb_value
            )

            if possible_path.exists():

                rgb_path = (
                    possible_path
                )


# Fallback to expected output location
if rgb_path is None:

    possible_path = (
        output_folder
        / "rgb_images"
        / f"{stem}_rgb.jpg"
    )

    if possible_path.exists():

        rgb_path = (
            possible_path
        )


# ============================================================
# IMAGE VIEW
# ============================================================

st.subheader(
    "Image view"
)


view = st.radio(
    "View",
    [
        "RGB",
        "Thermal",
        "Ironbow",
        "Grayscale",
        "Hotspot",
        "Delta T",
        "Contour",
        "RGB + Thermal"
    ],
    horizontal=True
)


# ============================================================
# RGB ONLY
# ============================================================

if view == "RGB":

    if rgb_path is not None:

        st.image(
            str(rgb_path),
            caption=(
                f"{selected_file} "
                f"- RGB view"
            ),
            width=DISPLAY_WIDTH
        )

    else:

        st.warning(
            "No embedded RGB image was found "
            "for this thermal image."
        )


# ============================================================
# THERMAL FIGURES
# ============================================================

else:

    fig = None


    # --------------------------------------------------------
    # THERMAL
    # --------------------------------------------------------

    if view in [
        "Thermal",
        "RGB + Thermal"
    ]:

        fig = px.imshow(
            temperature,

            color_continuous_scale="Turbo",

            labels={
                "x":
                    "Thermal pixel X",

                "y":
                    "Thermal pixel Y",

                "color":
                    "Temperature °C"
            },

            aspect="equal"
        )


    # --------------------------------------------------------
    # IRONBOW
    # --------------------------------------------------------

    elif view == "Ironbow":

        fig = px.imshow(
            temperature,

            color_continuous_scale="Inferno",

            labels={
                "x":
                    "Thermal pixel X",

                "y":
                    "Thermal pixel Y",

                "color":
                    "Temperature °C"
            },

            aspect="equal"
        )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    elif view == "Grayscale":

        fig = px.imshow(
            temperature,

            color_continuous_scale="Gray",

            labels={
                "x":
                    "Thermal pixel X",

                "y":
                    "Thermal pixel Y",

                "color":
                    "Temperature °C"
            },

            aspect="equal"
        )


    # --------------------------------------------------------
    # HOTSPOT
    # --------------------------------------------------------

    elif view == "Hotspot":

        hotspot_threshold = st.slider(
            "Hotspot threshold (°C)",

            min_value=float(
                temperature.min()
            ),

            max_value=float(
                temperature.max()
            ),

            value=float(
                np.percentile(
                    temperature,
                    90
                )
            ),

            step=0.1,

            key="image_hotspot_threshold"
        )


        hotspot = np.where(
            temperature
            >= hotspot_threshold,

            temperature,

            np.nan
        )


        fig = px.imshow(
            hotspot,

            color_continuous_scale="Hot",

            labels={
                "x":
                    "Thermal pixel X",

                "y":
                    "Thermal pixel Y",

                "color":
                    "Temperature °C"
            },

            aspect="equal"
        )


    # --------------------------------------------------------
    # DELTA T
    # --------------------------------------------------------

    elif view == "Delta T":

        median_temp = float(
            np.median(
                temperature
            )
        )


        delta = (
            temperature
            - median_temp
        )


        fig = px.imshow(
            delta,

            color_continuous_scale="RdBu_r",

            color_continuous_midpoint=0,

            labels={
                "x":
                    "Thermal pixel X",

                "y":
                    "Thermal pixel Y",

                "color":
                    "ΔT °C"
            },

            aspect="equal"
        )


    # --------------------------------------------------------
    # CONTOUR
    # --------------------------------------------------------

    elif view == "Contour":

        contour_step = st.slider(
            "Contour interval (°C)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1
        )


        fig = go.Figure(

            data=go.Contour(

                z=temperature,

                colorscale="Turbo",

                contours=dict(

                    start=float(
                        temperature.min()
                    ),

                    end=float(
                        temperature.max()
                    ),

                    size=contour_step,

                    coloring="heatmap",

                    showlabels=True,

                    labelfont=dict(
                        size=10
                    )
                ),

                colorbar=dict(
                    title="Temperature °C"
                ),

                hovertemplate=
                    "X: %{x}<br>"
                    "Y: %{y}<br>"
                    "Temperature: "
                    "%{z:.2f} °C"
                    "<extra></extra>"
            )
        )


        fig.update_layout(
            xaxis_title=(
                "Thermal pixel X"
            ),

            yaxis_title=(
                "Thermal pixel Y"
            )
        )


        fig.update_yaxes(
            autorange="reversed"
        )


    # ========================================================
    # HOVER DATA
    # ========================================================

    if (
        fig is not None
        and view == "Delta T"
    ):

        fig.update_traces(
            hovertemplate=
                "X: %{x}<br>"
                "Y: %{y}<br>"
                "ΔT: %{z:.2f} °C"
                "<extra></extra>"
        )


    elif (
        fig is not None
        and view != "Contour"
    ):

        fig.update_traces(
            hovertemplate=
                "X: %{x}<br>"
                "Y: %{y}<br>"
                "Temperature: "
                "%{z:.2f} °C"
                "<extra></extra>"
        )


    # ========================================================
    # FIXED DISPLAY SIZE
    # ========================================================

    if fig is not None:

        fig.update_layout(

            width=DISPLAY_WIDTH,

            height=DISPLAY_HEIGHT,

            autosize=False,

            margin=dict(
                l=60,
                r=80,
                t=20,
                b=50
            )
        )


    # ========================================================
    # RGB + THERMAL SIDE BY SIDE
    # ========================================================

    if view == "RGB + Thermal":

        left, right = st.columns(
            [1, 1]
        )


        with left:

            st.markdown(
                "**RGB image**"
            )

            if rgb_path is not None:

                st.image(
                    str(rgb_path),
                    width=DISPLAY_WIDTH
                )

            else:

                st.warning(
                    "RGB image unavailable."
                )


        with right:

            st.markdown(
                "**Thermal image**"
            )

            st.plotly_chart(
                fig,
                width=DISPLAY_WIDTH,
                height=DISPLAY_HEIGHT
            )


    # ========================================================
    # NORMAL THERMAL DISPLAY
    # ========================================================

    else:

        if fig is not None:

            st.plotly_chart(
                fig,
                width=DISPLAY_WIDTH,
                height=DISPLAY_HEIGHT
            )


# ============================================================
# HOTSPOT ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "Hotspot analysis"
)


analysis_threshold = st.slider(
    "Analysis threshold (°C)",

    min_value=float(
        temperature.min()
    ),

    max_value=float(
        temperature.max()
    ),

    value=float(
        np.percentile(
            temperature,
            90
        )
    ),

    step=0.1,

    key="analysis_threshold"
)


hot_mask = (
    temperature
    >= analysis_threshold
)


hot_pixels = int(
    hot_mask.sum()
)


hot_percent = (
    hot_pixels
    / temperature.size
    * 100
)


median_temperature = float(
    np.median(
        temperature
    )
)


maximum_temperature = float(
    np.max(
        temperature
    )
)


delta_max = (
    maximum_temperature
    - median_temperature
)


# ============================================================
# HOTSPOT LOCATION
# ============================================================

max_y, max_x = np.unravel_index(
    np.argmax(
        temperature
    ),

    temperature.shape
)

# ============================================================
# TEMPERATURE DISTRIBUTION
# ============================================================

st.subheader("Temperature distribution")

temperature_flat = temperature.flatten()

fig_hist = px.histogram(
    x=temperature_flat,
    nbins=50,
    labels={
        "x": "Temperature (°C)",
        "y": "Thermal pixels"
    }
)

fig_hist.update_layout(
    width=640,
    height=350,
    autosize=False,
    xaxis_title="Temperature (°C)",
    yaxis_title="Pixel count",
    showlegend=False,
    margin=dict(
        l=60,
        r=30,
        t=20,
        b=50
    )
)

# Threshold line
fig_hist.add_vline(
    x=analysis_threshold,
    line_dash="dash",
    line_color="red",
    annotation_text=(
        f"Threshold "
        f"{analysis_threshold:.1f}°C"
    ),
    annotation_position="top"
)

# Median line
fig_hist.add_vline(
    x=median_temperature,
    line_dash="dot",
    annotation_text=(
        f"Median "
        f"{median_temperature:.1f}°C"
    ),
    annotation_position="bottom"
)

st.plotly_chart(
    fig_hist,
    width=640,
    height=350
)


# ============================================================
# ANALYSIS METRICS
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "Median temperature",
    f"{median_temperature:.2f} °C"
)


col2.metric(
    "Max above median",
    f"{delta_max:.2f} °C"
)


col3.metric(
    "Area above threshold",
    f"{hot_percent:.2f}%"
)


col4.metric(
    "Hotspot pixel",
    f"({max_x}, {max_y})"
)


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

with st.expander(
    "Image details"
):

    st.write(
        f"**Source:** {selected_file}"
    )

    st.write(
        f"**Thermal resolution:** "
        f"{temperature.shape[1]} × "
        f"{temperature.shape[0]}"
    )

    st.write(
        f"**Total thermal pixels:** "
        f"{temperature.size:,}"
    )

    st.write(
        f"**Pixels above threshold:** "
        f"{hot_pixels:,}"
    )

    if rgb_path is not None:

        st.write(
            f"**RGB image:** "
            f"{rgb_path.name}"
        )

    else:

        st.write(
            "**RGB image:** unavailable"
        )