from pathlib import Path
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POSE_ORDER = {
    "LOW": 0,
    "MID": 1,
    "HIGH": 2,
}


def resolve_temperature_csv(cabinet_output: Path, source_file: str, stored_path) -> Path:
    if pd.notna(stored_path):
        stored_path = str(stored_path).strip()
        if stored_path:
            candidate = Path(stored_path)
            if candidate.exists():
                return candidate

    stem = Path(source_file).stem
    fallback = cabinet_output / "temperature_csv" / f"{stem}_temperature.csv"

    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Temperature CSV not found for {source_file}. Tried: {fallback}"
    )


def load_temperature_matrix(cabinet_output: Path, row: pd.Series) -> np.ndarray:
    path = resolve_temperature_csv(
        cabinet_output,
        row["source_file"],
        row.get("temperature_csv", "")
    )

    matrix = np.loadtxt(path, delimiter=",")

    if matrix.ndim != 2:
        raise ValueError(f"{path} is not a 2D temperature matrix.")

    return matrix


def save_matrix_map(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "inferno",
    symmetric: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    if symmetric:
        limit = float(np.nanmax(np.abs(matrix)))
        if limit == 0:
            limit = 1e-6

        image = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
            vmin=-limit,
            vmax=limit,
        )
    else:
        image = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
        )

    ax.set_title(title)
    ax.set_xlabel("Thermal pixel X")
    ax.set_ylabel("Thermal pixel Y")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def calculate_frame_statistics(successful: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "minimum_C",
        "maximum_C",
        "mean_C",
        "median_C",
        "center_C",
        "delta_max_C",
    ]

    available_metrics = [
        column
        for column in metric_columns
        if column in successful.columns
    ]

    rows = []

    for pose in ["LOW", "MID", "HIGH"]:
        pose_rows = successful[
            successful["pose"] == pose
        ].copy()

        if pose_rows.empty:
            continue

        result = {
            "pose": pose,
            "frame_count": len(pose_rows),
        }

        for metric in available_metrics:
            values = pd.to_numeric(
                pose_rows[metric],
                errors="coerce"
            ).dropna()

            if values.empty:
                continue

            result[f"{metric}_average"] = float(values.mean())
            result[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )
            result[f"{metric}_range"] = float(
                values.max() - values.min()
            )

        rows.append(result)

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        result_df["pose_order"] = result_df["pose"].map(POSE_ORDER)
        result_df = (
            result_df
            .sort_values("pose_order")
            .drop(columns=["pose_order"])
        )

    return result_df


def analyse_pose_pixels(
    cabinet_output: Path,
    pose: str,
    pose_rows: pd.DataFrame,
    output_dir: Path,
):
    matrices = []
    names = []

    for _, row in pose_rows.iterrows():
        matrix = load_temperature_matrix(
            cabinet_output,
            row
        )

        matrices.append(matrix)
        names.append(str(row["source_file"]))

    if not matrices:
        return [], None

    shape = matrices[0].shape

    for matrix, name in zip(matrices, names):
        if matrix.shape != shape:
            raise ValueError(
                f"{name} has shape {matrix.shape}, expected {shape}."
            )

    stack = np.stack(matrices, axis=0)

    if len(matrices) > 1:
        pixel_std = np.std(
            stack,
            axis=0,
            ddof=1
        )
    else:
        pixel_std = np.zeros(shape, dtype=float)

    pixel_range = (
        np.max(stack, axis=0)
        - np.min(stack, axis=0)
    )

    save_matrix_map(
        pixel_std,
        output_dir / f"{pose.lower()}_pixel_std.png",
        f"{pose} - Per-pixel temperature standard deviation",
        "Standard deviation (°C)",
        cmap="inferno",
    )

    save_matrix_map(
        pixel_range,
        output_dir / f"{pose.lower()}_pixel_range.png",
        f"{pose} - Per-pixel temperature range",
        "Temperature range (°C)",
        cmap="inferno",
    )

    pairwise_rows = []

    for index_a, index_b in itertools.combinations(
        range(len(matrices)),
        2
    ):
        matrix_a = matrices[index_a]
        matrix_b = matrices[index_b]

        name_a = names[index_a]
        name_b = names[index_b]

        difference = matrix_b - matrix_a
        absolute_difference = np.abs(difference)

        mae = float(np.mean(absolute_difference))
        rmse = float(
            np.sqrt(
                np.mean(difference ** 2)
            )
        )
        max_abs_difference = float(
            np.max(absolute_difference)
        )
        mean_signed_difference = float(
            np.mean(difference)
        )

        pairwise_rows.append({
            "pose": pose,
            "frame_a": name_a,
            "frame_b": name_b,
            "mean_abs_difference_C": mae,
            "rmse_C": rmse,
            "max_abs_difference_C": max_abs_difference,
            "mean_signed_difference_C": mean_signed_difference,
        })

        map_name = (
            f"{pose.lower()}_"
            f"{Path(name_a).stem}_vs_"
            f"{Path(name_b).stem}_difference.png"
        )

        save_matrix_map(
            difference,
            output_dir / map_name,
            f"{pose}: {name_b} - {name_a}",
            "Temperature difference (°C)",
            cmap="RdBu_r",
            symmetric=True,
        )

    pose_pixel_summary = {
        "pose": pose,
        "frame_count": len(matrices),
        "mean_pixel_std_C": float(np.mean(pixel_std)),
        "median_pixel_std_C": float(np.median(pixel_std)),
        "max_pixel_std_C": float(np.max(pixel_std)),
        "mean_pixel_range_C": float(np.mean(pixel_range)),
        "max_pixel_range_C": float(np.max(pixel_range)),
    }

    if pairwise_rows:
        pairwise_df = pd.DataFrame(pairwise_rows)

        pose_pixel_summary[
            "average_pairwise_MAE_C"
        ] = float(
            pairwise_df[
                "mean_abs_difference_C"
            ].mean()
        )

        pose_pixel_summary[
            "average_pairwise_RMSE_C"
        ] = float(
            pairwise_df["rmse_C"].mean()
        )

        pose_pixel_summary[
            "largest_pairwise_pixel_difference_C"
        ] = float(
            pairwise_df[
                "max_abs_difference_C"
            ].max()
        )

    return pairwise_rows, pose_pixel_summary


def analyse_repeatability(cabinet_output: Path):
    cabinet_output = Path(cabinet_output)

    summary_path = (
        cabinet_output
        / "inspection_summary.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"inspection_summary.csv not found: {summary_path}"
        )

    summary = pd.read_csv(summary_path)

    required_columns = {
        "source_file",
        "pose",
        "status",
    }

    missing = (
        required_columns
        - set(summary.columns)
    )

    if missing:
        raise ValueError(
            "inspection_summary.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    summary["pose"] = (
        summary["pose"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    successful = summary[
        (
            summary["status"]
            .astype(str)
            .str.upper()
            == "OK"
        )
        &
        summary["pose"].isin(
            ["LOW", "MID", "HIGH"]
        )
    ].copy()

    if successful.empty:
        raise ValueError(
            "No successful LOW/MID/HIGH images found."
        )

    output_dir = (
        cabinet_output
        / "repeatability"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    frame_summary = calculate_frame_statistics(
        successful
    )

    frame_summary_path = (
        output_dir
        / "frame_repeatability_summary.csv"
    )

    frame_summary.to_csv(
        frame_summary_path,
        index=False
    )

    all_pairwise_rows = []
    pixel_summary_rows = []

    for pose in ["LOW", "MID", "HIGH"]:
        pose_rows = successful[
            successful["pose"] == pose
        ].copy()

        if pose_rows.empty:
            print(f"{pose}: no frames found.")
            continue

        print(
            f"{pose}: analysing "
            f"{len(pose_rows)} frame(s)..."
        )

        if len(pose_rows) != 3:
            print(
                f"  WARNING: expected 3 frames for {pose}, "
                f"found {len(pose_rows)}."
            )

        pairwise_rows, pixel_summary = analyse_pose_pixels(
            cabinet_output,
            pose,
            pose_rows,
            output_dir
        )

        all_pairwise_rows.extend(pairwise_rows)

        if pixel_summary is not None:
            pixel_summary_rows.append(
                pixel_summary
            )

    pairwise_df = pd.DataFrame(
        all_pairwise_rows
    )

    pairwise_path = (
        output_dir
        / "pairwise_frame_comparisons.csv"
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False
    )

    pixel_summary_df = pd.DataFrame(
        pixel_summary_rows
    )

    if not pixel_summary_df.empty:
        pixel_summary_df["pose_order"] = (
            pixel_summary_df["pose"]
            .map(POSE_ORDER)
        )

        pixel_summary_df = (
            pixel_summary_df
            .sort_values("pose_order")
            .drop(columns=["pose_order"])
        )

    pixel_summary_path = (
        output_dir
        / "pixel_repeatability_summary.csv"
    )

    pixel_summary_df.to_csv(
        pixel_summary_path,
        index=False
    )

    print()
    print("========================================")
    print("FRAME-LEVEL REPEATABILITY")
    print("========================================")

    if not frame_summary.empty:
        display_columns = [
            column
            for column in [
                "pose",
                "frame_count",
                "mean_C_average",
                "mean_C_std",
                "maximum_C_average",
                "maximum_C_std",
                "median_C_average",
                "median_C_std",
                "delta_max_C_average",
                "delta_max_C_std",
            ]
            if column in frame_summary.columns
        ]

        print(
            frame_summary[
                display_columns
            ].to_string(index=False)
        )

    print()
    print("========================================")
    print("PIXEL-LEVEL REPEATABILITY")
    print("========================================")

    if not pixel_summary_df.empty:
        print(
            pixel_summary_df.to_string(
                index=False
            )
        )

    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("Key files:")
    print(f"  {frame_summary_path.name}")
    print(f"  {pairwise_path.name}")
    print(f"  {pixel_summary_path.name}")
    print()
    print(
        "Lower SD, MAE and RMSE mean better repeatability."
    )
    print(
        "Large structured areas in difference maps may indicate "
        "camera/arm movement rather than sensor noise."
    )

    return {
        "frame_summary": frame_summary,
        "pairwise": pairwise_df,
        "pixel_summary": pixel_summary_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Investigate thermal repeatability between "
            "repeated LOW/MID/HIGH frames."
        )
    )

    parser.add_argument(
        "cabinet_output",
        help=(
            "Processed cabinet output folder, "
            "for example outputs/CAB01"
        )
    )

    args = parser.parse_args()

    analyse_repeatability(
        Path(args.cabinet_output)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
