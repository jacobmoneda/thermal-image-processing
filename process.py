#!/usr/bin/env python3
"""
process.py - Batch-process original UNI-T UTi260E JPGs.

Usage:
    python process.py image.jpg
    python process.py path/to/folder

Outputs:
    outputs/
    ├── temperature_csv/
    │   └── <name>_temperature.csv
    ├── heatmaps/
    │   └── <name>_heatmap.png
    ├── summaries/
    │   └── <name>_summary.csv
    └── inspection_summary.csv

Dependencies:
    pip install numpy matplotlib

Important:
    Use original JPG files copied directly from the UTi260E / SD card.
    Re-saving an image in another application may remove the embedded
    radiometric payload.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WIDTH = 256
HEIGHT = 192
PIXEL_COUNT = WIDTH * HEIGHT
BYTES_PER_PIXEL_RECORD = 8
THERMAL_BYTES = PIXEL_COUNT * BYTES_PER_PIXEL_RECORD
PAYLOAD_HEADER_BYTES = 4


def extract_temperature_matrix(image_path: Path) -> np.ndarray:
    """Return the UTi260E 192x256 temperature matrix in degrees Celsius."""
    data = image_path.read_bytes()

    if len(data) < 2 or data[:2] != b"\xff\xd8":
        raise ValueError("Input does not appear to be a JPEG file.")

    # Tested UTi260E files contain more than one JPEG section.
    # The thermal payload follows the final JPEG EOI marker.
    jpeg_end = data.rfind(b"\xff\xd9")
    if jpeg_end == -1:
        raise ValueError("Could not find JPEG end marker (FFD9).")

    thermal_start = jpeg_end + 2 + PAYLOAD_HEADER_BYTES
    thermal_end = thermal_start + THERMAL_BYTES

    if thermal_end > len(data):
        available = max(0, len(data) - thermal_start)
        raise ValueError(
            "Not enough data for a UTi260E "
            f"{WIDTH}x{HEIGHT} thermal payload. "
            f"Expected {THERMAL_BYTES} bytes, found {available}."
        )

    thermal_block = data[thermal_start:thermal_end]

    # Observed record format:
    #   bytes 0..3 : big-endian float32 temperature in degC
    #   bytes 4..7 : auxiliary value
    record_dtype = np.dtype([
        ("temperature", ">f4"),
        ("aux", ">u4"),
    ])

    records = np.frombuffer(
        thermal_block,
        dtype=record_dtype,
        count=PIXEL_COUNT,
    )

    matrix = (
        records["temperature"]
        .astype(np.float32)
        .reshape((HEIGHT, WIDTH))
    )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Decoded matrix contains NaN/Inf values. "
            "This file may use a different payload format."
        )

    if matrix.min() < -300 or matrix.max() > 1000:
        raise ValueError(
            "Decoded temperatures are outside a plausible range. "
            "This file may use a different UTi260E payload layout."
        )

    return matrix


def compute_stats(matrix: np.ndarray) -> dict:
    min_y, min_x = np.unravel_index(np.argmin(matrix), matrix.shape)
    max_y, max_x = np.unravel_index(np.argmax(matrix), matrix.shape)

    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    return {
        "minimum_C": float(matrix[min_y, min_x]),
        "minimum_x_px": int(min_x),
        "minimum_y_px": int(min_y),
        "maximum_C": float(matrix[max_y, max_x]),
        "maximum_x_px": int(max_x),
        "maximum_y_px": int(max_y),
        "mean_C": float(matrix.mean()),
        "center_C": float(matrix[center_y, center_x]),
        "center_x_px": center_x,
        "center_y_px": center_y,
    }


def write_temperature_csv(matrix: np.ndarray, output_path: Path) -> None:
    np.savetxt(
        output_path,
        matrix,
        delimiter=",",
        fmt="%.4f",
    )


def write_summary_csv(
    stats: dict,
    output_path: Path,
    source: Path,
) -> None:
    rows = [
        ("source_file", source.name),
        ("thermal_width_px", WIDTH),
        ("thermal_height_px", HEIGHT),
        ("minimum_C", f"{stats['minimum_C']:.4f}"),
        ("minimum_x_px", stats["minimum_x_px"]),
        ("minimum_y_px", stats["minimum_y_px"]),
        ("maximum_C", f"{stats['maximum_C']:.4f}"),
        ("maximum_x_px", stats["maximum_x_px"]),
        ("maximum_y_px", stats["maximum_y_px"]),
        ("mean_C", f"{stats['mean_C']:.4f}"),
        ("center_C", f"{stats['center_C']:.4f}"),
        ("center_x_px", stats["center_x_px"]),
        ("center_y_px", stats["center_y_px"]),
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_heatmap(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.imshow(
        matrix,
        origin="upper",
        interpolation="nearest",
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Temperature (°C)")

    max_y, max_x = np.unravel_index(np.argmax(matrix), matrix.shape)
    min_y, min_x = np.unravel_index(np.argmin(matrix), matrix.shape)

    ax.annotate(
        f"MAX {matrix[max_y, max_x]:.1f}°C",
        xy=(max_x, max_y),
        xytext=(8, 8),
        textcoords="offset points",
    )

    ax.annotate(
        f"MIN {matrix[min_y, min_x]:.1f}°C",
        xy=(min_x, min_y),
        xytext=(8, -16),
        textcoords="offset points",
    )

    ax.set_title(title)
    ax.set_xlabel("Thermal pixel X")
    ax.set_ylabel("Thermal pixel Y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError("Input file must be .jpg or .jpeg.")
        return [input_path]

    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
        )

        if not files:
            raise ValueError(
                f"No JPG/JPEG files found in folder: {input_path}"
            )

        return files

    raise ValueError(f"Input path does not exist: {input_path}")


def create_output_dirs(output_dir: Path) -> dict:
    dirs = {
        "root": output_dir,
        "temperature_csv": output_dir / "temperature_csv",
        "heatmaps": output_dir / "heatmaps",
        "summaries": output_dir / "summaries",
    }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def process_one(image_path: Path, dirs: dict) -> dict:
    matrix = extract_temperature_matrix(image_path)
    stats = compute_stats(matrix)
    stem = image_path.stem

    temperature_csv = (
        dirs["temperature_csv"] / f"{stem}_temperature.csv"
    )

    heatmap_png = (
        dirs["heatmaps"] / f"{stem}_heatmap.png"
    )

    summary_csv = (
        dirs["summaries"] / f"{stem}_summary.csv"
    )

    write_temperature_csv(
        matrix,
        temperature_csv
    )

    write_heatmap(
        matrix,
        heatmap_png,
        f"UTi260E Thermal Map - {image_path.name}",
    )

    write_summary_csv(
        stats,
        summary_csv,
        image_path
    )

    return {
        "source_file": image_path.name,
        "thermal_width_px": WIDTH,
        "thermal_height_px": HEIGHT,
        **stats,
        "temperature_csv": str(temperature_csv),
        "heatmap_png": str(heatmap_png),
        "summary_csv": str(summary_csv),
        "status": "OK",
        "error": "",
    }


def write_inspection_summary(
    rows: list[dict],
    output_path: Path
) -> None:

    fieldnames = [
        "source_file",
        "thermal_width_px",
        "thermal_height_px",

        "minimum_C",
        "minimum_x_px",
        "minimum_y_px",

        "maximum_C",
        "maximum_x_px",
        "maximum_y_px",

        "mean_C",
        "center_C",

        "center_x_px",
        "center_y_px",

        "temperature_csv",
        "heatmap_png",
        "summary_csv",

        "status",
        "error",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            formatted = dict(row)

            for key in (
                "minimum_C",
                "maximum_C",
                "mean_C",
                "center_C",
            ):
                if isinstance(
                    formatted.get(key),
                    (float, np.floating)
                ):
                    formatted[key] = (
                        f"{formatted[key]:.4f}"
                    )

            writer.writerow(formatted)


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Process one UTi260E JPG "
            "or every JPG in a folder."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Original UTi260E JPG file "
            "or folder containing JPGs"
        ),
    )

    args = parser.parse_args()

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    try:
        files = collect_input_files(
            input_path
        )

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr
        )
        return 1

    # Output location:
    #
    # Single file:
    #   same folder as the image
    #
    # Folder:
    #   inside the supplied folder

    # Store all results in ./outputs/
    project_dir = Path.cwd()

    if input_path.is_dir():
        inspection_name = input_path.name
    else:
        inspection_name = input_path.stem

    base_dir = project_dir / "outputs" / inspection_name

    dirs = create_output_dirs(base_dir)

    print(
        f"Found {len(files)} image(s)."
    )

    print(
        f"Output folder: "
        f"{dirs['root']}"
    )

    print()

    summary_rows = []

    successful = 0
    failed = 0

    for index, image_path in enumerate(
        files,
        start=1
    ):

        print(
            f"[{index}/{len(files)}] "
            f"Processing {image_path.name} ..."
        )

        try:
            row = process_one(
                image_path,
                dirs
            )

            summary_rows.append(row)

            successful += 1

            print(
                f"    MIN "
                f"{row['minimum_C']:.2f} °C | "

                f"MAX "
                f"{row['maximum_C']:.2f} °C | "

                f"MEAN "
                f"{row['mean_C']:.2f} °C | "

                f"CENTER "
                f"{row['center_C']:.2f} °C"
            )

        except Exception as exc:

            failed += 1

            error_message = str(exc)

            print(
                f"    FAILED: "
                f"{error_message}"
            )

            summary_rows.append({

                "source_file":
                    image_path.name,

                "thermal_width_px": "",
                "thermal_height_px": "",

                "minimum_C": "",
                "minimum_x_px": "",
                "minimum_y_px": "",

                "maximum_C": "",
                "maximum_x_px": "",
                "maximum_y_px": "",

                "mean_C": "",
                "center_C": "",

                "center_x_px": "",
                "center_y_px": "",

                "temperature_csv": "",
                "heatmap_png": "",
                "summary_csv": "",

                "status": "FAILED",

                "error":
                    error_message,
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
        "Processing complete."
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Summary:    "
        f"{inspection_summary}"
    )

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())