from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from prediction import DemandPredictionService  # noqa: E402
from prediction.config import ENRICHED_TRAINING_DIR, MODEL_DIR, OUTPUT_DIR  # noqa: E402
from wvv_dashboard.app import TransitDataRepository  # noqa: E402
from wvv_dashboard.config import DATA_DIR  # noqa: E402


STOP_FILE = OUTPUT_DIR / "train_prediction_models.stop"
PID_FILE = OUTPUT_DIR / "train_prediction_models.pid"
PROGRESS_FILE = OUTPUT_DIR / "train_prediction_models_progress.json"
LOG_FILE = OUTPUT_DIR / "train_prediction_models.log"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def model_path_for_line(line: int) -> Path:
    return MODEL_DIR / f"wvv_prediction_lines_{int(line)}.pkl"


def discover_enriched_lines() -> list[int]:
    pattern = re.compile(r"data_2025-\d{2}-\d{2}_2025-\d{2}-\d{2}_line_(\d+)_clean_context_2025\.parquet$")
    lines: set[int] = set()
    if not ENRICHED_TRAINING_DIR.exists():
        return []
    for path in ENRICHED_TRAINING_DIR.glob("*.parquet"):
        match = pattern.match(path.name)
        if match:
            lines.add(int(match.group(1)))
    return sorted(lines)


def parse_lines(raw_lines: list[str] | None) -> list[int]:
    if not raw_lines:
        return discover_enriched_lines()
    lines: set[int] = set()
    for chunk in raw_lines:
        for part in re.split(r"[,;\s]+", chunk.strip()):
            if part:
                lines.add(int(part))
    return sorted(lines)


def prioritize_untrained_lines(lines: list[int], force: bool) -> list[int]:
    if force:
        return lines
    untrained = [line for line in lines if not model_path_for_line(line).exists()]
    trained = [line for line in lines if model_path_for_line(line).exists()]
    return untrained + trained


def write_progress(payload: dict[str, object]) -> None:
    payload = {"updated_at": timestamp(), **payload}
    PROGRESS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_progress() -> dict[str, object]:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def command_run(args: argparse.Namespace) -> int:
    lines = parse_lines(args.lines)
    if not lines:
        log(f"Keine enriched-2025 Parquet-Dateien gefunden: {ENRICHED_TRAINING_DIR}")
        return 1
    lines = prioritize_untrained_lines(lines, args.force)

    STOP_FILE.unlink(missing_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    completed: list[int] = []
    skipped: list[int] = []
    failed: dict[str, str] = {}
    repo = TransitDataRepository(DATA_DIR)

    untrained_count = sum(1 for line in lines if not model_path_for_line(line).exists())
    trained_count = len(lines) - untrained_count
    log(f"Starte Prediction-Training für {len(lines)} Linien: {lines}")
    log(f"Priorität: zuerst {untrained_count} noch nicht trainierte Linien, danach {trained_count} bestehende Modelle.")
    log(f"Quelle: {ENRICHED_TRAINING_DIR}")
    write_progress(
        {
            "status": "running",
            "pid": os.getpid(),
            "total": len(lines),
            "current_line": None,
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
        }
    )

    try:
        for index, line in enumerate(lines, start=1):
            if STOP_FILE.exists():
                log("Stop angefordert. Beende nach abgeschlossener Linie.")
                break

            model_exists = model_path_for_line(line).exists()
            if model_exists and args.skip_existing:
                skipped.append(line)
                log(f"[{index}/{len(lines)}] Linie {line}: Modell existiert bereits, überspringe.")
                write_progress(
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "total": len(lines),
                        "current_line": None,
                        "completed": completed,
                        "skipped": skipped,
                        "failed": failed,
                    }
                )
                continue

            if args.force:
                mode_text = "neues Volltraining"
            elif model_exists:
                mode_text = f"inkrementelles Weitertraining (+{args.additional_trees} Bäume)"
            else:
                mode_text = "erstes Training"
            log(f"[{index}/{len(lines)}] Linie {line}: {mode_text} startet.")
            write_progress(
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "total": len(lines),
                    "current_line": line,
                    "completed": completed,
                    "skipped": skipped,
                    "failed": failed,
                }
            )
            try:
                service = DemandPredictionService(repo)
                result = service.fit(
                    [line],
                    date(2026, 1, 1),
                    date(2026, 12, 31),
                    incremental=not args.force,
                    additional_trees=args.additional_trees,
                )
                if service.trained:
                    completed.append(line)
                    log(f"[{index}/{len(lines)}] Linie {line}: {result.message}")
                else:
                    failed[str(line)] = result.message
                    log(f"[{index}/{len(lines)}] Linie {line}: nicht trainiert - {result.message}")
            except Exception as exc:
                failed[str(line)] = str(exc)
                log(f"[{index}/{len(lines)}] Linie {line}: Fehler - {exc}")
                log(traceback.format_exc())
            finally:
                gc.collect()

        status = "stopped" if STOP_FILE.exists() else "finished"
        write_progress(
            {
                "status": status,
                "pid": os.getpid(),
                "total": len(lines),
                "current_line": None,
                "completed": completed,
                "skipped": skipped,
                "failed": failed,
            }
        )
        log(f"Training {status}. Fertig: {len(completed)}, übersprungen: {len(skipped)}, Fehler: {len(failed)}.")
        return 0 if not failed else 2
    finally:
        PID_FILE.unlink(missing_ok=True)


def command_stop(_args: argparse.Namespace) -> int:
    STOP_FILE.write_text(f"stop requested at {timestamp()}\n", encoding="utf-8")
    print(f"Stop angefordert. Der Trainer beendet sich nach der aktuell laufenden Linie.\nStop-Datei: {STOP_FILE}")
    return 0


def command_status(_args: argparse.Namespace) -> int:
    progress = read_progress()
    pid = PID_FILE.read_text(encoding="utf-8").strip() if PID_FILE.exists() else "-"
    stop_requested = STOP_FILE.exists()
    print(f"PID: {pid}")
    print(f"Stop angefordert: {'ja' if stop_requested else 'nein'}")
    if progress:
        print(json.dumps(progress, indent=2, ensure_ascii=False))
    else:
        print("Noch kein Fortschritt vorhanden.")
    if LOG_FILE.exists():
        print(f"\nLog: {LOG_FILE}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trainiert gespeicherte WVV-Prediction-Modelle außerhalb der GUI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Trainiert zuerst fehlende Modelle und erweitert danach bestehende Modelle inkrementell.",
    )
    run_parser.add_argument("--lines", nargs="*", help="Optional: Linienliste, z.B. --lines 10 20 27 oder --lines 10,20,27")
    run_parser.add_argument("--force", action="store_true", help="Ignoriert gespeicherte Modelle und trainiert komplett neu.")
    run_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Trainiert nur noch fehlende Modelle und überspringt bereits vorhandene Modelle.",
    )
    run_parser.add_argument("--additional-trees", type=int, default=30, help="Neue RandomForest-Bäume pro inkrementellem Lauf.")
    run_parser.set_defaults(func=command_run)

    stop_parser = subparsers.add_parser("stop", help="Fordert einen sauberen Stop nach der aktuellen Linie an.")
    stop_parser.set_defaults(func=command_stop)

    status_parser = subparsers.add_parser("status", help="Zeigt Fortschritt, PID und Logpfad.")
    status_parser.set_defaults(func=command_status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
