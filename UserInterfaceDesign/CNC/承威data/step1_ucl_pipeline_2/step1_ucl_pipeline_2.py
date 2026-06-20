import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Make project root importable so we can reuse ConfigManager.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import ConfigManager  # noqa: E402

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
STH_DIR = DATA_DIR / "sth"
CNC_DIR = DATA_DIR / "cnc"
EXP_DIR = DATA_DIR / "exp"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_REF_PATH = OUTPUT_DIR / "reference.csv"
DEFAULT_ALIGNED_PATH = OUTPUT_DIR / "aligned_no_ucl.csv"

# Processing parameters
WINDOW_SIZE = 100
UCL_SHIFT_AMOUNT = 30

# Optional hard-coded mode override. Set to "reference" or "align" to bypass CLI mode flag.
HARDCODE_MODE: Optional[str] = "align"

def load_first_coeff_row(coeff_path: Path) -> Dict[str, float]:
    """
    Read the first row from coefficient.csv and return a dict of coefficients.
    MAC is ignored; only the numeric columns are used.
    """
    df = pd.read_csv(coeff_path)
    if df.empty:
        raise ValueError(f"No rows found in coefficient file: {coeff_path}")
    row = df.iloc[0]
    return {
        "Torque": float(row["Torque"]),
        "BendingX": float(row["BendingX"]),
        "BendingY": float(row["BendingY"]),
        "CH1": float(row["CH1"]),
        "CH2": float(row["CH2"]),
    }


def pick_latest_csv(folder: Path) -> Path:
    """
    Pick the newest CSV file in a folder. Raises if none exist.
    """
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def time_sync(pdata_df: pd.DataFrame, cnc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align CNC and Pdata by nearest absolute TimeInSeconds, keeping offsets/NC_Line and adding Times/TimeTag.
    """
    def ensure_time_in_seconds(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Add TimeInSeconds if missing by parsing a datetime-like column (absolute seconds)."""
        out = df.copy()
        if time_col in out.columns:
            dt_series = pd.to_datetime(out[time_col], errors="coerce")
            if dt_series.isna().all():
                raise ValueError(f"Cannot parse any datetime values from {time_col}")
            time_sec = dt_series.view("int64") / 1e9
            time_sec = time_sec.where(dt_series.notna(), np.nan)
            out["TimeInSeconds"] = time_sec
            return out
        if "TimeInSeconds" in out.columns:
            return out
        raise KeyError(f"Expected either {time_col} or TimeInSeconds in dataframe")

    pdata_df = pdata_df.copy()
    cnc_df = cnc_df.copy()

    # Derive TimeInSeconds if not present.
    try:
        cnc_df = ensure_time_in_seconds(cnc_df, "Time")
    except Exception as e:
        raise ValueError(f"CNC time column missing or unparsable: {e}")
    try:
        pdata_df = ensure_time_in_seconds(pdata_df, "TimeTag")
    except Exception as e:
        raise ValueError(f"Pdata time column missing or unparsable: {e}")

    if cnc_df.empty:
        empty_cols = [
            "MAC",
            "Power",
            "Temperature",
            "Packet Sequence Number",
            "RSSI",
            "TTOLP",
            "Torque",
            "BendingX",
            "BendingY",
            "CH1",
            "CH2",
            "TimeTag",
            "TimeInSeconds",
            "Offset_X",
            "Offset_Y",
            "Offset_Z",
            "NC_Line",
            "Times",
            "Pdata_TimeInSeconds",
        ]
        return pd.DataFrame(columns=empty_cols)

    cnc_df["TimeInSeconds"] = cnc_df["TimeInSeconds"].astype(float)
    if not pdata_df.empty:
        pdata_df["TimeInSeconds"] = pdata_df["TimeInSeconds"].astype(float)

    if "NC_Line" in cnc_df.columns:
        cnc_df["NC_Line"] = pd.to_numeric(cnc_df["NC_Line"], errors="coerce")
    else:
        cnc_df["NC_Line"] = np.nan

    for col in ["Offset_X", "Offset_Y", "Offset_Z"]:
        cnc_df[col] = pd.to_numeric(cnc_df[col], errors="coerce")

    sync_df = pd.merge_asof(
        cnc_df.sort_values("TimeInSeconds"),
        pdata_df.sort_values("TimeInSeconds"),
        on="TimeInSeconds",
        direction="nearest",
        suffixes=("", "_pdata"),
    )

    if "TimeInSeconds_pdata" in sync_df.columns:
        sync_df["Pdata_TimeInSeconds"] = sync_df["TimeInSeconds_pdata"]
        sync_df = sync_df.drop(columns=["TimeInSeconds_pdata"])
    else:
        sync_df["Pdata_TimeInSeconds"] = None

    if "Time" in sync_df.columns and not sync_df["Time"].isna().all():
        sync_df["TimeTag"] = sync_df["Time"].fillna(
            sync_df["TimeInSeconds"].apply(
                lambda x: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{x % 60:06.3f}"
            )
        )
    else:
        sync_df["TimeTag"] = sync_df["TimeInSeconds"].apply(
            lambda x: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{x % 60:06.3f}"
        )

    sync_df["Times"] = sync_df["TimeInSeconds"].apply(
        lambda x: f"{int(x % 60):02d}.{int((x % 1) * 1000):03d}"
    )

    return sync_df


def apply_coefficients_to_pdata(sync_df: pd.DataFrame, coeffs: Dict[str, float]) -> pd.DataFrame:
    """
    Apply calibration to Pdata columns (_pdata), fallback to CNC columns if needed.
    Produces calibrated columns without suffix for downstream calculations.
    """
    df = sync_df.copy()
    mapping = {
        "Torque": "Torque_pdata" if "Torque_pdata" in df.columns else "Torque",
        "BendingX": "BendingX_pdata" if "BendingX_pdata" in df.columns else "BendingX",
        "BendingY": "BendingY_pdata" if "BendingY_pdata" in df.columns else "BendingY",
        "CH1": "CH1_pdata" if "CH1_pdata" in df.columns else "CH1",
        "CH2": "CH2_pdata" if "CH2_pdata" in df.columns else "CH2",
    }

    for target_col, source_col in mapping.items():
        if source_col not in df.columns:
            df[f"{target_col}_calibrated"] = np.nan
            continue
        df[f"{target_col}_calibrated"] = df[source_col].astype(float) * coeffs[target_col]

    return df


def calculate_ucl(
    df: pd.DataFrame, ucl_sigmas: Iterable[int], window_size: int = WINDOW_SIZE
) -> pd.DataFrame:
    """
    Rolling mean/std on calibrated Torque/BendingX; add Torque_UCL_n/BendingX_UCL_n.
    """
    out = df.copy()
    rolling_targets = {
        "Torque": "Torque_calibrated",
        "BendingX": "BendingX_calibrated",
    }
    for raw_name, source_col in rolling_targets.items():
        if source_col not in out.columns:
            for n in ucl_sigmas:
                out[f"{raw_name}_UCL_{n}"] = np.nan
            continue
        roll = out[source_col].rolling(window=window_size, min_periods=2)
        mean = roll.mean()
        std = roll.std()
        for n in ucl_sigmas:
            col_name = f"{raw_name}_UCL_{n}"
            out[col_name] = (mean + n * std).fillna(out[source_col])
    return out


def shift_ucl_data(
    df: pd.DataFrame, ucl_sigmas: Iterable[int], shift_amount: int = UCL_SHIFT_AMOUNT
) -> pd.DataFrame:
    """
    Shift UCL columns backward (negative shift), fill trailing gaps, and compute Flute_UCL_n from CH1/CH2.
    """
    out = df.copy()
    dynamic_cols: List[str] = []
    for prefix in ("Torque", "BendingX"):
        for n in ucl_sigmas:
            dynamic_cols.append(f"{prefix}_UCL_{n}")

    for col in dynamic_cols:
        if col not in out.columns:
            out[col] = np.nan
        shifted = out[col].shift(-shift_amount)
        last_valid_index = len(shifted) - shift_amount - 1
        fill_value: Optional[float] = None
        if last_valid_index >= 0 and pd.notna(shifted.iloc[last_valid_index]):
            fill_value = shifted.iloc[last_valid_index]
        if fill_value is not None:
            shifted = shifted.fillna(fill_value)
        out[col] = shifted

    ch1 = out.get("CH1_calibrated")
    ch2 = out.get("CH2_calibrated")
    if ch1 is not None and ch2 is not None:
        mean = pd.concat([ch1, ch2]).mean()
        std = pd.concat([ch1, ch2]).std()
        for n in ucl_sigmas:
            out[f"Flute_UCL_{n}"] = mean + n * std
    else:
        for n in ucl_sigmas:
            out[f"Flute_UCL_{n}"] = np.nan

    return out


def finalize_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reference output: keep UCL columns, drop unused fields, keep calibrated values under original names.
    """
    out = df.copy()

    drop_cols = {
        "NC_File",
        "Tool_Code",
        "Tool_Life",
        "Radius_Comp",
        "Radius_Wear",
        "Length_Comp",
        "Length_Wear",
        "Tool_Group",
        "Tool_Group_Info",
        "Spindle_Number",
        "TimeTag",
        "TempRaw",
        "Pdata_TimeInSeconds",
        "Times",
    }
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    replacements = ["Torque", "BendingX", "BendingY", "CH1", "CH2"]
    for col in replacements:
        cal_col = f"{col}_calibrated"
        if cal_col in out.columns:
            out[col] = out[cal_col]
    out = out.drop(columns=[f"{col}_calibrated" for col in replacements if f"{col}_calibrated" in out.columns], errors="ignore")

    return out


def finalize_no_ucl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligned output without UCL columns; keep calibrated values under original names.
    """
    out = df.copy()
    out = finalize_reference(out)
    ucl_cols = [c for c in out.columns if "UCL" in c]
    out = out.drop(columns=ucl_cols, errors="ignore")
    return out


def run_pipeline(mode: str, out_path: Optional[Path] = None, pdata_path: Optional[Path] = None, cnc_path: Optional[Path] = None):
    cfg = ConfigManager()
    coeffs = load_first_coeff_row(Path(cfg.COEFF_PATH))

    pdata_path = pdata_path or pick_latest_csv(STH_DIR)
    cnc_path = cnc_path or pick_latest_csv(CNC_DIR)

    logger.info("Using Pdata: %s", pdata_path)
    logger.info("Using CNC: %s", cnc_path)
    logger.info("Using coefficient row: %s", coeffs)

    pdata_df = pd.read_csv(pdata_path)
    cnc_df = pd.read_csv(cnc_path)

    synced = time_sync(pdata_df, cnc_df)
    calibrated = apply_coefficients_to_pdata(synced, coeffs)

    if mode == "reference":
        with_ucl = calculate_ucl(calibrated, cfg.UCL_SIGMA, window_size=WINDOW_SIZE)
        shifted = shift_ucl_data(with_ucl, cfg.UCL_SIGMA, shift_amount=UCL_SHIFT_AMOUNT)
        finalized = finalize_reference(shifted)
        out_file = out_path or DEFAULT_REF_PATH
    else:
        finalized = finalize_no_ucl(calibrated)
        out_file = out_path or DEFAULT_ALIGNED_PATH

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    finalized.to_csv(out_file, index=False, float_format="%.6f")
    print(f"Completed. Mode={mode}. Output written to {out_file}")


def iter_trimmed_csv_files(root_dir: Path) -> Iterable[Path]:
    for path in root_dir.rglob("*_trimmed.csv"):
        if path.is_file():
            yield path


def batch_align_exp(exp_root: Path, cnc_path: Optional[Path] = None) -> int:
    """
    For each *_trimmed.csv under exp_root, generate EXP-X-X_aligned.csv in the same folder.
    """
    processed = 0
    for pdata_path in iter_trimmed_csv_files(exp_root):
        exp_name = pdata_path.parent.parent.name if pdata_path.parent.name.lower() == "sth" else pdata_path.parent.name
        out_path = pdata_path.parent / f"{exp_name}_aligned.csv"
        run_pipeline("align", out_path=out_path, pdata_path=pdata_path, cnc_path=cnc_path)
        processed += 1
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCL pipeline: build reference or aligned output.")
    parser.add_argument("--mode", choices=["reference", "align"], default="align", help="reference: create reference.csv with UCL; align: time-aligned data without UCL.")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path (optional).")
    parser.add_argument("--pdata", type=Path, default=None, help="Override Pdata CSV path.")
    parser.add_argument("--cnc", type=Path, default=None, help="Override CNC CSV path.")
    parser.add_argument("--batch-exp", action="store_true", help="Process all *_trimmed.csv under data/exp.")
    parser.add_argument("--exp-root", type=Path, default=EXP_DIR, help="EXP root path for batch mode.")
    args = parser.parse_args()
    chosen_mode = HARDCODE_MODE or args.mode
    if args.batch_exp:
        total = batch_align_exp(args.exp_root, cnc_path=args.cnc)
        print(f"Done. Processed {total} file(s).")
    else:
        run_pipeline(chosen_mode, args.out, args.pdata, args.cnc)
