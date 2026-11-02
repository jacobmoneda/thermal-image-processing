from pathlib import Path
import argparse
import csv
import struct

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# UTi260E THERMAL CONSTANTS
# ============================================================

THERMAL_WIDTH = 256
THERMAL_HEIGHT = 192

THERMAL_PIXELS = THERMAL_WIDTH * THERMAL_HEIGHT

# Each thermal pixel record is:
#   4 bytes -> big-endian float32 temperature
#   4 bytes -> auxiliary data
THERMAL_RECORD_SIZE = 8

THERMAL_BLOCK_SIZE = (
    THERMAL_PIXELS
    * THERMAL_RECORD_SIZE
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def create_output_dirs(output_dir: Path) -> dict:
    """
    Create the output directory structure for one inspection.
    """

    dirs = {
        "root":
            output_dir,

        "temperature_csv":
            output_dir
            / "temperature_csv",

        "heatmaps":
            output_dir
            / "heatmaps",

        "rgb_images":
            output_dir
            / "rgb_images",

        "summaries":
            output_dir
            / "summaries",
    }

    for directory in dirs.values():
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    return dirs


# ============================================================
# INPUT FILE COLLECTION
# ============================================================

def collect_input_files(input_path: Path):
    """
    Return UTi260E JPG files from either:
      - one JPG file
      - a folder of JPG files
    """

    if input_path.is_file():

        if input_path.suffix.lower() not in [
            ".jpg",
            ".jpeg"
        ]:
            raise ValueError(
                "Input file must be JPG/JPEG."
            )

        return [input_path]

    if input_path.is_dir():

        files = sorted([
            path
            for path in input_path.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in [".jpg", ".jpeg"]
        ])

        if not files:
            raise FileNotFoundError(
                f"No JPG files found in {input_path}"
            )

        return files

    raise FileNotFoundError(
        f"Input does not exist: {input_path}"
    )


# ============================================================
# EMBEDDED JPEG EXTRACTION
# ============================================================

def extract_embedded_jpegs(
    image_path: Path
):
    """
    Extract all complete JPEG streams embedded inside
    the UTi260E image file.

    UTi260E files observed in this project contain
    multiple JPEG streams.

    Returns:
        list of JPEG byte strings
    """

    data = image_path.read_bytes()

    jpeg_start = b"\xff\xd8"
    jpeg_end = b"\xff\xd9"

    images = []

    search_position = 0

    while True:

        start = data.find(
            jpeg_start,
            search_position
        )

        if start == -1:
            break

        end = data.find(
            jpeg_end,
            start + 2
        )

        if end == -1:
            break

        end += 2

        images.append(
            data[start:end]
        )

        search_position = end

    return images


# ============================================================
# RGB EXTRACTION
# ============================================================

def save_rgb_image(
    image_path: Path,
    dirs: dict
):
    """
    Extract the embedded visible-light image.

    Based on the UTi260E files tested:
        JPEG 1 = rendered thermal image
        JPEG 2 = RGB / visible-light image

    Returns:
        Path to RGB JPG or None.
    """

    embedded_images = (
        extract_embedded_jpegs(
            image_path
        )
    )

    if len(embedded_images) < 2:

        print(
            f"  RGB image not found: "
            f"{image_path.name}"
        )

        return None

    rgb_data = embedded_images[1]

    output_path = (
        dirs["rgb_images"]
        / f"{image_path.stem}_rgb.jpg"
    )

    output_path.write_bytes(
        rgb_data
    )

    return output_path


# ============================================================
# THERMAL PAYLOAD LOCATION
# ============================================================

def find_final_jpeg_end(
    data: bytes
):
    """
    Find the final JPEG EOI marker FF D9.

    The radiometric block begins after the final
    embedded JPEG plus four additional bytes.
    """

    marker = b"\xff\xd9"

    position = data.rfind(marker)

    if position == -1:
        raise ValueError(
            "Could not find final JPEG EOI marker."
        )

    return position + 2


# ============================================================
# TEMPERATURE EXTRACTION
# ============================================================

def extract_temperature_matrix(
    image_path: Path
):
    """
    Extract the 256 x 192 radiometric temperature
    matrix from an original UTi260E JPG.

    File structure observed:

        [JPEG data]
        FF D9
        [4 bytes]
        [thermal records]

    Each thermal record:
        bytes 0-3 -> big-endian float32 temperature °C
        bytes 4-7 -> auxiliary data

    Returns:
        numpy array shape (192, 256)
    """

    data = image_path.read_bytes()

    jpeg_end = find_final_jpeg_end(
        data
    )

    # Radiometric data begins four bytes
    # after the final JPEG.
    thermal_start = (
        jpeg_end + 4
    )

    thermal_end = (
        thermal_start
        + THERMAL_BLOCK_SIZE
    )

    if thermal_end > len(data):

        available = (
            len(data)
            - thermal_start
        )

        raise ValueError(
            f"Thermal payload too small. "
            f"Expected {THERMAL_BLOCK_SIZE} bytes, "
            f"found {available}."
        )

    thermal_data = data[
        thermal_start:
        thermal_end
    ]

    temperatures = np.empty(
        THERMAL_PIXELS,
        dtype=np.float32
    )

    for index in range(
        THERMAL_PIXELS
    ):

        offset = (
            index
            * THERMAL_RECORD_SIZE
        )

        temperature = struct.unpack(
            ">f",
            thermal_data[
                offset:
                offset + 4
            ]
        )[0]

        temperatures[index] = (
            temperature
        )

    matrix = temperatures.reshape(
        (
            THERMAL_HEIGHT,
            THERMAL_WIDTH
        )
    )

    # Basic sanity check.
    if not np.all(
        np.isfinite(matrix)
    ):

        raise ValueError(
            "Temperature matrix contains "
            "invalid values."
        )

    return matrix


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    temperature: np.ndarray
):
    """
    Calculate thermal statistics for one image.
    """

    min_index = np.unravel_index(
        np.argmin(temperature),
        temperature.shape
    )

    max_index = np.unravel_index(
        np.argmax(temperature),
        temperature.shape
    )

    min_y, min_x = min_index
    max_y, max_x = max_index

    center_x = (
        THERMAL_WIDTH // 2
    )

    center_y = (
        THERMAL_HEIGHT // 2
    )

    minimum = float(
        temperature[min_y, min_x]
    )

    maximum = float(
        temperature[max_y, max_x]
    )

    mean = float(
        np.mean(temperature)
    )

    median = float(
        np.median(temperature)
    )

    center = float(
        temperature[
            center_y,
            center_x
        ]
    )

    delta_max = (
        maximum - median
    )

    return {
        "minimum_C":
            minimum,

        "minimum_x_px":
            int(min_x),

        "minimum_y_px":
            int(min_y),

        "maximum_C":
            maximum,

        "maximum_x_px":
            int(max_x),

        "maximum_y_px":
            int(max_y),

        "mean_C":
            mean,

        "median_C":
            median,

        "center_C":
            center,

        "delta_max_C":
            delta_max,

        "center_x_px":
            center_x,

        "center_y_px":
            center_y,
    }


# ============================================================
# SAVE TEMPERATURE CSV
# ============================================================

def save_temperature_csv(
    temperature: np.ndarray,
    output_path: Path
):
    """
    Save all 49,152 thermal temperatures as CSV.
    """

    np.savetxt(
        output_path,
        temperature,
        delimiter=",",
        fmt="%.4f"
    )


# ============================================================
# SAVE HEATMAP
# ============================================================

def save_heatmap(
    temperature: np.ndarray,
    output_path: Path,
    title: str
):
    """
    Save a PNG heatmap derived from the
    temperature matrix.
    """

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(
        temperature,
        cmap="inferno",
        interpolation="nearest"
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Thermal pixel X"
    )

    ax.set_ylabel(
        "Thermal pixel Y"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Temperature (°C)"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150
    )

    plt.close(fig)


# ============================================================
# SAVE INDIVIDUAL SUMMARY
# ============================================================

def save_image_summary(
    summary: dict,
    output_path: Path
):
    """
    Save statistics for one thermal image.
    """

    fieldnames = list(
        summary.keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerow(
            summary
        )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one(
    image_path: Path,
    dirs: dict
):
    """
    Process one original UTi260E JPG.
    """

    print(
        f"Processing {image_path.name}"
    )

    stem = image_path.stem

    # --------------------------------------------------------
    # THERMAL MATRIX
    # --------------------------------------------------------

    temperature = (
        extract_temperature_matrix(
            image_path
        )
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    stats = calculate_statistics(
        temperature
    )

    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    temperature_csv_path = (
        dirs["temperature_csv"]
        / f"{stem}_temperature.csv"
    )

    heatmap_path = (
        dirs["heatmaps"]
        / f"{stem}_heatmap.png"
    )

    summary_path = (
        dirs["summaries"]
        / f"{stem}_summary.csv"
    )

    # --------------------------------------------------------
    # SAVE TEMPERATURE CSV
    # --------------------------------------------------------

    save_temperature_csv(
        temperature,
        temperature_csv_path
    )

    # --------------------------------------------------------
    # SAVE HEATMAP
    # --------------------------------------------------------

    save_heatmap(
        temperature,
        heatmap_path,
        image_path.name
    )

    # --------------------------------------------------------
    # EXTRACT RGB IMAGE
    # --------------------------------------------------------

    rgb_path = save_rgb_image(
        image_path,
        dirs
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    row = {
        "source_file":
            image_path.name,

        "thermal_width_px":
            THERMAL_WIDTH,

        "thermal_height_px":
            THERMAL_HEIGHT,

        "minimum_C":
            stats["minimum_C"],

        "minimum_x_px":
            stats["minimum_x_px"],

        "minimum_y_px":
            stats["minimum_y_px"],

        "maximum_C":
            stats["maximum_C"],

        "maximum_x_px":
            stats["maximum_x_px"],

        "maximum_y_px":
            stats["maximum_y_px"],

        "mean_C":
            stats["mean_C"],

        "median_C":
            stats["median_C"],

        "center_C":
            stats["center_C"],

        "delta_max_C":
            stats["delta_max_C"],

        "center_x_px":
            stats["center_x_px"],

        "center_y_px":
            stats["center_y_px"],

        "temperature_csv":
            str(
                temperature_csv_path
            ),

        "heatmap_png":
            str(
                heatmap_path
            ),

        "rgb_image":
            str(rgb_path)
            if rgb_path
            else "",

        "summary_csv":
            str(summary_path),

        "status":
            "OK",

        "error":
            "",
    }

    save_image_summary(
        row,
        summary_path
    )

    print(
        f"  Min:    "
        f"{stats['minimum_C']:.2f} °C"
    )

    print(
        f"  Max:    "
        f"{stats['maximum_C']:.2f} °C"
    )

    print(
        f"  Mean:   "
        f"{stats['mean_C']:.2f} °C"
    )

    print(
        f"  Median: "
        f"{stats['median_C']:.2f} °C"
    )

    print(
        f"  Center: "
        f"{stats['center_C']:.2f} °C"
    )

    print(
        f"  ΔT max: "
        f"{stats['delta_max_C']:.2f} °C"
    )

    if rgb_path:

        print(
            f"  RGB:    {rgb_path.name}"
        )

    return row


# ============================================================
# INSPECTION SUMMARY
# ============================================================

def write_inspection_summary(
    rows,
    output_path: Path
):
    """
    Write one CSV containing all processed images.
    """

    if not rows:
        return

    # Use union of all keys so failed and
    # successful rows always work together.
    fieldnames = []

    for row in rows:

        for key in row.keys():

            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# PROCESS WHOLE INSPECTION
# ============================================================

def process_inspection(
    input_folder: Path,
    output_folder: Path
):
    """
    Process all UTi260E JPGs in one inspection.

    This function is imported directly by app.py.

    Returns:
        list of result dictionaries
    """

    input_folder = Path(
        input_folder
    )

    output_folder = Path(
        output_folder
    )

    files = collect_input_files(
        input_folder
    )

    dirs = create_output_dirs(
        output_folder
    )

    summary_rows = []

    for image_path in files:

        try:

            row = process_one(
                image_path,
                dirs
            )

            summary_rows.append(
                row
            )

        except Exception as exc:

            print(
                f"FAILED: "
                f"{image_path.name}: "
                f"{exc}"
            )

            summary_rows.append({
                "source_file":
                    image_path.name,

                "thermal_width_px":
                    "",

                "thermal_height_px":
                    "",

                "minimum_C":
                    "",

                "minimum_x_px":
                    "",

                "minimum_y_px":
                    "",

                "maximum_C":
                    "",

                "maximum_x_px":
                    "",

                "maximum_y_px":
                    "",

                "mean_C":
                    "",

                "median_C":
                    "",

                "center_C":
                    "",

                "delta_max_C":
                    "",

                "center_x_px":
                    "",

                "center_y_px":
                    "",

                "temperature_csv":
                    "",

                "heatmap_png":
                    "",

                "rgb_image":
                    "",

                "summary_csv":
                    "",

                "status":
                    "FAILED",

                "error":
                    str(exc),
            })

    inspection_summary = (
        dirs["root"]
        / "inspection_summary.csv"
    )

    write_inspection_summary(
        summary_rows,
        inspection_summary
    )

    print()
    print(
        f"Inspection summary: "
        f"{inspection_summary}"
    )

    return summary_rows


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    """
    Command line usage:

        python process.py inputs/inspection1

    or:

        python process.py inputs/00015E.jpg
    """

    parser = argparse.ArgumentParser(
        description=(
            "Process original UTi260E "
            "radiometric JPG files."
        )
    )

    parser.add_argument(
        "input",
        help=(
            "Input UTi260E JPG "
            "or inspection folder."
        )
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    # --------------------------------------------------------
    # OUTPUT INSPECTION NAME
    # --------------------------------------------------------

    if input_path.is_dir():

        inspection_name = (
            input_path.name
        )

    else:

        inspection_name = (
            input_path.stem
        )

    # --------------------------------------------------------
    # CLEAN PROJECT OUTPUT STRUCTURE
    # --------------------------------------------------------

    project_dir = Path.cwd()

    output_folder = (
        project_dir
        / "outputs"
        / inspection_name
    )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    if input_path.is_dir():

        process_inspection(
            input_path,
            output_folder
        )

    else:

        dirs = create_output_dirs(
            output_folder
        )

        try:

            row = process_one(
                input_path,
                dirs
            )

            write_inspection_summary(
                [row],
                output_folder
                / "inspection_summary.csv"
            )

        except Exception as exc:

            print(
                f"FAILED: {exc}"
            )

            return 1

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )