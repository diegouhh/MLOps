from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

APPLEE_REVISION = "5b554417070aed47fcd240b6923ada68f62db029"
SUPPORTED_EEG_EXTENSIONS = {".edf", ".bdf", ".vhdr", ".set"}
BANDS = {
    "Delta": (4, 6),
    "Theta": (6, 8.5),
    "Alpha-1": (8.5, 10.5),
    "Alpha-2": (10.5, 12.5),
    "Beta1": (12.5, 18.5),
    "Beta2": (18.5, 21),
    "Beta3": (21, 30),
    "Gamma": (30, 45),
}


def _entity(path: Path, key: str) -> str | None:
    prefix = f"{key}-"
    for token in path.stem.split("_"):
        if token.startswith(prefix) and len(token) > len(prefix):
            return token[len(prefix) :]
    for part in path.parts:
        if part.startswith(prefix) and len(part) > len(prefix):
            return part[len(prefix) :]
    return None


def _raw_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("*_eeg.*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EEG_EXTENSIONS:
            continue
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if "derivatives" not in relative.parts:
            result.append(path)
    return sorted(result)


def _safe_link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _mirror_bids(source_root: Path, target_root: Path, subject: str | None = None) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        if "derivatives" in relative.parts:
            continue
        if source.is_dir():
            continue
        if subject:
            subject_parts = [part for part in relative.parts if part.startswith("sub-")]
            if subject_parts and subject_parts[0] != f"sub-{subject}":
                continue
        _safe_link_or_copy(source, target_root / relative)


def _resolve_recording(source_root: Path, raw_recording: str | None) -> Path | None:
    if not raw_recording:
        return None
    relative = Path(raw_recording)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("El registro solicitado no es una ruta BIDS relativa válida")
    selected = (source_root / relative).resolve()
    selected.relative_to(source_root.resolve())
    if not selected.is_file() or selected.suffix.lower() not in SUPPORTED_EEG_EXTENSIONS:
        raise ValueError("El registro solicitado no es un EEG compatible")
    return selected


def _resolve_task(root: Path, configured: str | None, selected: Path | None) -> str:
    if selected is not None:
        task = _entity(selected, "task")
        if task:
            return task
    if configured:
        return configured
    tasks = sorted({task for path in _raw_files(root) if (task := _entity(path, "task"))})
    if len(tasks) == 1:
        return tasks[0]
    if not tasks:
        raise ValueError("No se pudo determinar la tarea BIDS para APPLEE")
    raise ValueError(
        "El dataset contiene varias tareas BIDS; define task en la configuración APPLEE"
    )


def _feature_spec(feature: str) -> tuple[str, dict[str, Any]]:
    if feature == "cohfreq":
        return feature, {"window": 3, "bands": BANDS}
    if feature == "sl":
        return feature, {"bands": BANDS}
    if feature == "power":
        return feature, {"bands": BANDS, "irasa": False, "osc": False, "aperiodic": False}
    if feature == "crossfreq":
        return feature, {"bands": BANDS}
    if feature == "entropy":
        return feature, {"bands": BANDS, "D": 3}
    raise ValueError(f"Característica APPLEE no soportada: {feature}")


def _combine_feature_tables(metrics_dir: Path, features: list[str]) -> pd.DataFrame:
    combined: pd.DataFrame | None = None
    for feature in features:
        candidates = sorted(metrics_dir.glob(f"{feature}_*_columns.feather"))
        if not candidates:
            raise RuntimeError(
                f"APPLEE no produjo el archivo columnar esperado para la característica {feature}"
            )
        table = pd.read_feather(candidates[-1])
        if "subject" not in table.columns:
            raise RuntimeError(f"La salida {candidates[-1].name} no contiene subject")
        metadata_columns = {"subject", "Task", "group", "SITE"}
        feature_columns = [column for column in table.columns if column not in metadata_columns]
        if not feature_columns:
            raise RuntimeError(
                f"La salida {candidates[-1].name} no contiene variables de {feature}"
            )
        selected = table[["subject", *feature_columns]].copy()
        selected = selected.rename(
            columns={column: f"{feature}__{column}" for column in feature_columns}
        )
        for column in selected.columns:
            if column != "subject":
                selected[column] = pd.to_numeric(selected[column], errors="coerce")
        selected = selected.groupby("subject", as_index=False).mean(numeric_only=True)
        combined = (
            selected
            if combined is None
            else combined.merge(selected, on="subject", how="inner")
        )

    if combined is None or combined.empty:
        raise RuntimeError("APPLEE no produjo filas de características")
    numeric_columns = [column for column in combined.columns if column != "subject"]
    combined[numeric_columns] = combined[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return combined


def _run(request: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_root = Path(request["dataset_path"]).resolve()
    workspace = Path(request["workspace"]).resolve()
    output_csv = Path(request["output_csv"]).resolve()
    summary_json = Path(request["summary_json"]).resolve()
    config = dict(request["config"])
    selected = _resolve_recording(source_root, request.get("recording"))
    selected_subject = _entity(selected, "sub") if selected is not None else None

    if workspace.exists():
        shutil.rmtree(workspace)
    bids_root = workspace / "bids"
    metrics_dir = workspace / "metrics"
    qc_dir = workspace / "qc"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    _mirror_bids(source_root, bids_root, subject=selected_subject)

    task = _resolve_task(source_root, config.get("task"), selected)
    session = _entity(selected, "ses") if selected is not None else config.get("session")
    run = _entity(selected, "run") if selected is not None else None
    acquisition = _entity(selected, "acq") if selected is not None else None
    recording = _entity(selected, "recording") if selected is not None else None
    extension = selected.suffix.lower() if selected is not None else None

    layout: dict[str, Any] = {
        "task": task,
        "suffix": "eeg",
        "return_type": "filename",
    }
    if session:
        layout["session"] = session
    if selected_subject:
        layout["subject"] = selected_subject
    if run:
        layout["run"] = run
    if acquisition:
        layout["acquisition"] = acquisition
    if recording:
        layout["recording"] = recording
    if extension:
        layout["extension"] = extension

    dataset = {
        "name": "NeuroOps_APPLEE",
        "input_path": str(bids_root),
        "layout": layout,
        "args": {"line_freqs": [float(config["line_freq"])]},
        "group_regex": None,
        "events_to_keep": None,
        "run-label": f"neuroops-{task}",
    }

    from APPLEE.ld_postprocessing import run_postprocessing
    from APPLEE.ld_preprocessing import preprocessing_ld
    from sovaflow.flow import standardize

    preprocessing_ld(
        THE_DATASET=dataset,
        resample=config.get("resample"),
        reduction_channels=config["channels"],
        ica_method="infomax",
        montage_kind="standard_1005",
        fun_names_map=standardize,
        L_FREQ=float(config["l_freq"]),
        H_FREQ=float(config["h_freq"]),
        prep=bool(config["prep"]),
        path_save=str(qc_dir),
        N_Epochs_concat=24,
        SCOREPOCHS=False,
    )

    normalization = config.get("normalization", "none")
    run_postprocessing(
        DATASET=dataset,
        electrodes=config["channels"],
        bands=BANDS,
        path=str(metrics_dir),
        features_tuples=[_feature_spec(feature) for feature in config["features"]],
        epoch=float(config["epoch"]),
        norm=normalization != "none",
        z_score=normalization == "zscore",
        huber=normalization == "huber",
        N_Epochs_concat=24,
        SCOREPOCHS=False,
    )

    frame = _combine_feature_tables(metrics_dir, config["features"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    summary = {
        "adapter": "APPLEE",
        "applee_revision": APPLEE_REVISION,
        "task": task,
        "session": session,
        "recording": request.get("recording"),
        "subjects": int(frame["subject"].nunique()),
        "rows": int(frame.shape[0]),
        "feature_count": int(frame.shape[1] - 1),
        "features": list(config["features"]),
        "channels": list(config["channels"]),
        "bands": {name: list(bounds) for name, bounds in BANDS.items()},
        "normalization": normalization,
        "l_freq": float(config["l_freq"]),
        "h_freq": float(config["h_freq"]),
        "epoch": float(config["epoch"]),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame, summary


def main() -> int:
    if len(sys.argv) != 2:
        print("Uso: runner.py request.json", file=sys.stderr)
        return 2
    request_path = Path(sys.argv[1])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        _run(request)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
