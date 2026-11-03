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

st.title("Spark Thermal Analytics")


INPUT_ROOT = Path("inputs")
OUTPUT_ROOT = Path("outputs")

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Slightly smaller when showing RGB and thermal side-by-side.
SIDE_WIDTH = 520
SIDE_HEIGHT = 390


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_pose(filename):
    """
    Infer LOW / MID / HIGH from the filename.

    Examples:
        CAB01_LOW.jpg
        LOW_00015E.jpg
        inspection_HIGH_01.jpg
    """

    name = str(filename).upper()

    if "LOW" in name:
        return "LOW"

    if "MID" in name:
        return "MID"

    if "HIGH" in name:
        return "HIGH"

    return "UNKNOWN"


POSE_ORDER = {
    "LOW": 0,
    "MID": 1,
    "HIGH": 2
}


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not INPUT_ROOT.exists():

    st.warning(
        "The inputs/ directory does not exist."
    )

    st.stop()


# ============================================================
# FIND CABINETS / INSPECTIONS
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


# Each inspection folder is treated as one cabinet.
inspection = st.selectbox(
    "Select cabinet",
    inspections
)

cabinet_id = inspection


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
        "Process this cabinet to generate results."
    )

    st.stop()


summary = pd.read_csv(
    summary_file
)


# ============================================================
# ADD POSE INFORMATION
# ============================================================

summary["pose"] = (
    summary["source_file"]
    .apply(get_pose)
)


# ============================================================
# SUCCESSFUL IMAGES
# ============================================================

successful = summary[
    summary["status"] == "OK"
].copy()


if successful.empty:

    st.warning(
        "No successfully processed thermal images."
    )

    st.stop()


# Convert thermal columns to numeric.
numeric_columns = [
    "minimum_C",
    "maximum_C",
    "mean_C",
    "median_C",
    "center_C",
    "delta_max_C"
]

for column in numeric_columns:

    if column in successful.columns:

        successful[column] = pd.to_numeric(
            successful[column],
            errors="coerce"
        )


# ============================================================
# CABINET ANALYSIS
# ============================================================

st.header(
    f"Cabinet analysis — {cabinet_id}"
)


# ------------------------------------------------------------
# OVERALL CABINET STATISTICS
# ------------------------------------------------------------

cabinet_max = float(
    successful["maximum_C"].max()
)

cabinet_mean = float(
    successful["mean_C"].mean()
)


if "delta_max_C" in successful.columns:

    cabinet_delta_max = float(
        successful["delta_max_C"].max()
    )

else:

    cabinet_delta_max = float(
        successful["maximum_C"].max()
        - successful["mean_C"].mean()
    )


hottest_index = (
    successful["maximum_C"]
    .idxmax()
)

hottest_row = (
    successful.loc[
        hottest_index
    ]
)

hottest_pose = (
    hottest_row["pose"]
)

if hottest_pose == "UNKNOWN":

    hottest_label = str(
        hottest_row["source_file"]
    )

else:

    hottest_label = hottest_pose


metric1, metric2, metric3, metric4 = (
    st.columns(4)
)


metric1.metric(
    "Cabinet maximum",
    f"{cabinet_max:.2f} °C"
)


metric2.metric(
    "Average temperature",
    f"{cabinet_mean:.2f} °C"
)


metric3.metric(
    "Largest ΔT",
    f"{cabinet_delta_max:.2f} °C"
)


metric4.metric(
    "Hottest section",
    hottest_label
)


# ============================================================
# LOW / MID / HIGH ANALYSIS
# ============================================================

known_pose_rows = successful[
    successful["pose"].isin(
        ["LOW", "MID", "HIGH"]
    )
].copy()


if not known_pose_rows.empty:

    # --------------------------------------------------------
    # AGGREGATE MULTIPLE FRAMES PER HEIGHT
    # --------------------------------------------------------

    aggregation = {
        "maximum_C": "max",
        "mean_C": "mean",
        "source_file": "count"
    }

    if "median_C" in known_pose_rows.columns:
        aggregation["median_C"] = "median"

    if "delta_max_C" in known_pose_rows.columns:
        aggregation["delta_max_C"] = "max"


    pose_summary = (
        known_pose_rows
        .groupby(
            "pose",
            as_index=False
        )
        .agg(aggregation)
    )


    pose_summary = pose_summary.rename(
        columns={
            "source_file":
                "image_count"
        }
    )


    pose_summary["pose_order"] = (
        pose_summary["pose"]
        .map(POSE_ORDER)
    )


    pose_summary = (
        pose_summary
        .sort_values(
            "pose_order"
        )
    )


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    st.subheader(
        "LOW / MID / HIGH comparison"
    )


    comparison_columns = [
        "pose",
        "maximum_C",
        "mean_C"
    ]

    if "median_C" in pose_summary.columns:
        comparison_columns.append(
            "median_C"
        )

    if "delta_max_C" in pose_summary.columns:
        comparison_columns.append(
            "delta_max_C"
        )

    comparison_columns.append(
        "image_count"
    )


    st.dataframe(
        pose_summary[
            comparison_columns
        ],
        hide_index=True,
        width="content"
    )


    # ========================================================
    # CABINET TEMPERATURE PROFILE
    # ========================================================

    st.subheader(
        "Cabinet temperature profile"
    )


    fig_profile = go.Figure()


    fig_profile.add_trace(
        go.Scatter(
            x=pose_summary["pose"],
            y=pose_summary["maximum_C"],

            mode="lines+markers",

            name="Maximum",

            hovertemplate=
                "Section: %{x}<br>"
                "Maximum: %{y:.2f} °C"
                "<extra></extra>"
        )
    )


    fig_profile.add_trace(
        go.Scatter(
            x=pose_summary["pose"],
            y=pose_summary["mean_C"],

            mode="lines+markers",

            name="Mean",

            hovertemplate=
                "Section: %{x}<br>"
                "Mean: %{y:.2f} °C"
                "<extra></extra>"
        )
    )


    if "median_C" in pose_summary.columns:

        fig_profile.add_trace(
            go.Scatter(
                x=pose_summary["pose"],
                y=pose_summary["median_C"],

                mode="lines+markers",

                name="Median",

                hovertemplate=
                    "Section: %{x}<br>"
                    "Median: %{y:.2f} °C"
                    "<extra></extra>"
            )
        )


    fig_profile.update_layout(

        width=640,
        height=400,

        autosize=False,

        xaxis_title=(
            "Cabinet section"
        ),

        yaxis_title=(
            "Temperature (°C)"
        ),

        xaxis=dict(
            categoryorder="array",

            categoryarray=[
                "LOW",
                "MID",
                "HIGH"
            ]
        ),

        margin=dict(
            l=60,
            r=30,
            t=20,
            b=50
        )
    )


    st.plotly_chart(
        fig_profile,
        width=640,
        height=400
    )


else:

    st.info(
        "No LOW / MID / HIGH labels were detected "
        "in the filenames. Name images with LOW, MID "
        "or HIGH to enable cabinet height comparison."
    )


# ============================================================
# FULL CABINET INSPECTION SUMMARY
# ============================================================

with st.expander(
    "Full cabinet inspection table"
):

    display_columns = [
        "source_file",
        "pose",
        "minimum_C",
        "maximum_C",
        "mean_C",
        "median_C",
        "center_C",
        "delta_max_C",
        "status"
    ]


    display_columns = [
        column
        for column in display_columns
        if column in summary.columns
    ]


    st.dataframe(
        summary[
            display_columns
        ],
        width="stretch"
    )


# ============================================================
# SELECT CABINET SECTION / IMAGE
# ============================================================

st.divider()

st.header(
    "Detailed thermal inspection"
)


available_poses = [
    pose
    for pose in [
        "LOW",
        "MID",
        "HIGH"
    ]
    if pose in successful["pose"].values
]


# ------------------------------------------------------------
# USE LOW / MID / HIGH SELECTION WHEN POSSIBLE
# ------------------------------------------------------------

if available_poses:

    selected_pose = st.radio(
        "Cabinet section",
        available_poses,
        horizontal=True
    )


    pose_rows = successful[
        successful["pose"]
        == selected_pose
    ]


    # If timelapse captured more than one image
    # at this height, allow selection of the frame.
    if len(pose_rows) > 1:

        selected_file = st.selectbox(
            f"Select {selected_pose} frame",
            pose_rows["source_file"]
        )

        row = pose_rows[
            pose_rows["source_file"]
            == selected_file
        ].iloc[0]

    else:

        row = (
            pose_rows.iloc[0]
        )

        selected_file = (
            row["source_file"]
        )


# ------------------------------------------------------------
# FALLBACK FOR UNLABELLED FILES
# ------------------------------------------------------------

else:

    st.info(
        "Filename pose labels were not found, "
        "so image selection is being used instead."
    )


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
# SELECTED IMAGE STATISTICS
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


if (
    "median_C" in row.index
    and pd.notna(row["median_C"])
):

    col4.metric(
        "Median",
        f"{row['median_C']:.2f} °C"
    )

else:

    col4.metric(
        "Center",
        f"{row['center_C']:.2f} °C"
    )


if (
    "delta_max_C" in row.index
    and pd.notna(row["delta_max_C"])
):

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


# First try path stored by process.py.
if "rgb_image" in row.index:

    rgb_value = row[
        "rgb_image"
    ]


    if pd.notna(
        rgb_value
    ):

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


# Fallback to normal output path.
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
# DEFAULT ANALYSIS THRESHOLD
# ============================================================

temp_min = float(
    np.min(temperature)
)

temp_max = float(
    np.max(temperature)
)

default_threshold = float(
    np.percentile(
        temperature,
        90
    )
)

# Reset/clamp threshold when switching images
if (
    "analysis_threshold" not in st.session_state
    or st.session_state.analysis_threshold < temp_min
    or st.session_state.analysis_threshold > temp_max
):
    st.session_state.analysis_threshold = default_threshold


analysis_threshold = (
    st.session_state.analysis_threshold
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

            color_continuous_scale=(
                "Turbo"
            ),

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

            color_continuous_scale=(
                "Inferno"
            ),

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

            color_continuous_scale=(
                "Gray"
            ),

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

        hotspot = np.where(

            temperature
            >= analysis_threshold,

            temperature,

            np.nan
        )


        fig = px.imshow(

            hotspot,

            color_continuous_scale=(
                "Hot"
            ),

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

            color_continuous_scale=(
                "RdBu_r"
            ),

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
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1
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
                    width=SIDE_WIDTH
                )

            else:

                st.warning(
                    "RGB image unavailable."
                )


        with right:

            st.markdown(
                "**Thermal image**"
            )


            # Use slightly smaller plot
            # for the two-column layout.
            fig.update_layout(

                width=SIDE_WIDTH,

                height=SIDE_HEIGHT
            )


            st.plotly_chart(
                fig,
                width=SIDE_WIDTH,
                height=SIDE_HEIGHT
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


# ------------------------------------------------------------
# THRESHOLD SLIDER
# ------------------------------------------------------------

analysis_threshold = st.slider(
    "Hotspot / analysis threshold (°C)",

    min_value=temp_min,

    max_value=temp_max,

    value=float(
        st.session_state.analysis_threshold
    ),

    step=0.1,

    key="analysis_threshold"
)


# ------------------------------------------------------------
# HOTSPOT MASK
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# TEMPERATURE STATISTICS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# HOTSPOT LOCATION
# ------------------------------------------------------------

max_y, max_x = np.unravel_index(

    np.argmax(
        temperature
    ),

    temperature.shape
)


# ------------------------------------------------------------
# ANALYSIS METRICS
# ------------------------------------------------------------

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
# TEMPERATURE DISTRIBUTION
# ============================================================

st.subheader(
    "Temperature distribution"
)


temperature_flat = (
    temperature.flatten()
)


fig_hist = px.histogram(

    x=temperature_flat,

    nbins=50,

    labels={
        "x":
            "Temperature (°C)",

        "y":
            "Thermal pixels"
    }
)


fig_hist.update_layout(

    width=640,

    height=350,

    autosize=False,

    xaxis_title=(
        "Temperature (°C)"
    ),

    yaxis_title=(
        "Pixel count"
    ),

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

    line_width=2,

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
# HOTSPOT LOCATION
# ============================================================

max_y, max_x = np.unravel_index(

    np.argmax(
        temperature
    ),

    temperature.shape
)