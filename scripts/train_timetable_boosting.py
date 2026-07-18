from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from prediction.boosting import TimetableGradientBoostingOptimizer  # noqa: E402
from prediction.config import ENRICHED_TRAINING_DIR, OUTPUT_DIR  # noqa: E402


STOP_FILE = OUTPUT_DIR / "train_timetable_boosting.stop"
PID_FILE = OUTPUT_DIR / "train_timetable_boosting.pid"
PROGRESS_FILE = OUTPUT_DIR / "train_timetable_boosting_progress.json"
LOG_FILE = OUTPUT_DIR / "train_timetable_boosting.log"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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


def write_progress(payload: dict[str, object]) -> None:
    payload = {"updated_at": timestamp(), **payload}
    PROGRESS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def command_run(args: argparse.Namespace) -> int:
    lines = parse_lines(args.lines)
    if not lines:
        log(f"Keine enriched-2025 Parquet-Dateien gefunden: {ENRICHED_TRAINING_DIR}")
        return 1

    STOP_FILE.unlink(missing_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    optimizer = TimetableGradientBoostingOptimizer()
    iterations_done = 0
    target_iterations = None if args.continuous else max(1, int(args.iterations))
    log(f"Starte Gradient-Boosting-Fahrplantraining fuer Linien: {lines}")
    log(f"Kostenparameter: {args.cost:.2f} EUR pro Busstunde")
    log("Stoppen: Ctrl+C oder in zweiter Konsole `python scripts/train_timetable_boosting.py stop`")

    try:
        while True:
            if STOP_FILE.exists():
                log("Stop-Datei gefunden. Beende Training sauber.")
                break
            iterations_done += 1
            write_progress(
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "iteration": iterations_done,
                    "lines": lines,
                    "cost_per_bus_hour": args.cost,
                }
            )
            try:
                result = optimizer.fit(
                    lines=lines,
                    start=date(2025, 1, 1),
                    end=date(2025, 12, 31),
                    cost_per_bus_hour=args.cost,
                    estimators_per_iteration=args.estimators,
                    warm_start=not args.reset,
                )
                log(result.message)
                write_progress(
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "iteration": iterations_done,
                        "lines": lines,
                        "rows": result.rows,
                        "mae_runs": result.mae_runs,
                        "n_estimators": result.n_estimators,
                        "cost_per_bus_hour": result.cost_per_bus_hour,
                    }
                )
            except Exception:
                details = traceback.format_exc()
                log("Fehler im Gradient-Boosting-Training:\n" + details)
                write_progress({"status": "failed", "pid": os.getpid(), "error": details})
                return 1

            if target_iterations is not None and iterations_done >= target_iterations:
                break
            if args.sleep > 0:
                time.sleep(args.sleep)
            args.reset = False
    except KeyboardInterrupt:
        log("Training per Ctrl+C gestoppt.")
    finally:
        PID_FILE.unlink(missing_ok=True)
        STOP_FILE.unlink(missing_ok=True)

    write_progress({"status": "stopped", "iteration": iterations_done, "lines": lines})
    log("Gradient-Boosting-Fahrplantraining beendet.")
    return 0


def command_stop(_args: argparse.Namespace) -> int:
    STOP_FILE.write_text(timestamp(), encoding="utf-8")
    print(f"Stop angefordert: {STOP_FILE}")
    return 0


def command_status(_args: argparse.Namespace) -> int:
    if PROGRESS_FILE.exists():
        print(PROGRESS_FILE.read_text(encoding="utf-8"))
    else:
        print("Noch kein Gradient-Boosting-Training gestartet.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trainiert den kostenbewussten Gradient-Boosting-Fahrplanoptimizer.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Training starten")
    run.add_argument("--lines", nargs="*", help="Linien, z.B. 10 20 27 oder 10,20,27. Ohne Angabe: alle enriched-2025 Linien.")
    run.add_argument("--cost", type=float, default=230.0, help="Kosten pro Busstunde in EUR.")
    run.add_argument("--iterations", type=int, default=1, help="Anzahl Trainingsiterationen.")
    run.add_argument("--estimators", type=int, default=40, help="Neue Gradient-Boosting-Trees pro Iteration.")
    run.add_argument("--sleep", type=int, default=0, help="Pause in Sekunden zwischen Iterationen.")
    run.add_argument("--continuous", action="store_true", help="Laeuft dauerhaft bis Ctrl+C oder stop.")
    run.add_argument("--reset", action="store_true", help="Bestehendes Boosting-Modell ignorieren und neu starten.")
    run.set_defaults(func=command_run)

    stop = sub.add_parser("stop", help="Laufendes Training nach der aktuellen Iteration stoppen")
    stop.set_defaults(func=command_stop)

    status = sub.add_parser("status", help="Status anzeigen")
    status.set_defaults(func=command_status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["run"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
