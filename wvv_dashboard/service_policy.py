from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ServicePolicy:
    """
    Bündelt linientypspezifische Regeln für die adaptive Angebotsplanung.

    Die Klasse hält fachliche Parameter fest, die für die Ableitung eines
    angepassten Fahrtenangebots verwendet werden. Dazu zählen Mindestbedienung,
    maximal zulässige Taktabstände, Zielauslastung, Produktivitätsschwelle und
    angenommene Umlaufzeit je Linie bzw. Linienkategorie.

    Projektkontext:
        Die Parameter werden von der adaptiven Fahrtenheuristik sowie von
        KPI-Berechnungen verwendet, um betriebliche Entscheidungen nicht rein
        nach Nachfrage, sondern zusätzlich nach Service- und Wirtschaftlichkeits-
        regeln zu steuern.
    """

    name: str
    min_runs_when_active: int
    max_headway_minutes: int
    peak_max_headway_minutes: int
    min_baseline_share: float
    target_utilization: float
    productive_floor: float
    cycle_minutes: float
    max_extra_runs: int = 3
    max_new_runs: int = 2


UNI_HIGH_FREQUENCY_LINES = {10, 14, 114, 214}
CITY_CORE_LINES = {4, 5, 6, 16, 20, 26}
COMMUTER_FEEDER_LINES = {8, 13, 21, 27, 29, 33, 54, 55}


def service_policy_for_line(line: int | float | str | None) -> ServicePolicy:
    """
    Ordnet einer Linie eine vordefinierte Service-Policy zu.

    Die Funktion klassifiziert Linien in grobe Betriebskategorien wie
    Uni-/Hauptkorridor, Stadt-Hauptlinie oder Pendler-/Grundversorgung.
    Auf Basis dieser Kategorie werden feste Schwellenwerte für die adaptive
    Planung zurückgegeben.

    Parameter:
        line (int | float | str | None): Linienkennung in numerischer oder
            textueller Form.

    Rückgabewerte:
        ServicePolicy: Passende Richtlinie für die angegebene Linie.

    Fehler/Sonderfälle:
        Nicht numerisch interpretierbare Linienwerte werden defensiv behandelt
        und fallen auf die Standardlinie zurück.

    Projektkontext:
        Die Funktion bildet die zentrale Zuordnungsschicht zwischen
        Linienidentität und betrieblicher Angebotslogik.
    """
    try:
        line_number = int(float(line)) if line is not None else None
    except (TypeError, ValueError):
        line_number = None

    if line_number in UNI_HIGH_FREQUENCY_LINES:
        return ServicePolicy(
            name="Uni-/Hauptkorridor",
            min_runs_when_active=3,
            max_headway_minutes=20,
            peak_max_headway_minutes=15,
            min_baseline_share=0.62,
            target_utilization=0.68,
            productive_floor=0.34,
            cycle_minutes=58.0,
            max_extra_runs=4,
            max_new_runs=3,
        )

    if line_number in CITY_CORE_LINES:
        return ServicePolicy(
            name="Stadt-Hauptlinie",
            min_runs_when_active=2,
            max_headway_minutes=30,
            peak_max_headway_minutes=20,
            min_baseline_share=0.55,
            target_utilization=0.72,
            productive_floor=0.34,
            cycle_minutes=52.0,
            max_extra_runs=3,
            max_new_runs=2,
        )

    if line_number in COMMUTER_FEEDER_LINES:
        return ServicePolicy(
            name="Pendler-/Grundversorgung",
            min_runs_when_active=1,
            max_headway_minutes=60,
            peak_max_headway_minutes=30,
            min_baseline_share=0.45,
            target_utilization=0.76,
            productive_floor=0.32,
            cycle_minutes=46.0,
            max_extra_runs=3,
            max_new_runs=2,
        )

    return ServicePolicy(
        name="Standardlinie",
        min_runs_when_active=1,
        max_headway_minutes=60,
        peak_max_headway_minutes=30,
        min_baseline_share=0.42,
        target_utilization=0.76,
        productive_floor=0.30,
        cycle_minutes=45.0,
        max_extra_runs=2,
        max_new_runs=2,
    )


def is_peak_hour(hour: int | float | None) -> bool:
    """
    Prüft, ob eine Stunde als Hauptverkehrszeit behandelt wird.

    Als Peak gelten hier die morgendlichen und nachmittäglichen
    Verkehrsspitzen. Die Einordnung beeinflusst insbesondere strengere
    Taktgrenzen innerhalb der adaptiven Angebotslogik.

    Parameter:
        hour (int | float | None): Zu prüfende Stunde.

    Rückgabewerte:
        bool: ``True`` für Peak-Stunden, sonst ``False``.
    """
    if hour is None:
        return False
    hour_int = int(hour) % 24
    return 6 <= hour_int <= 9 or 15 <= hour_int <= 18


def estimated_bus_hours(runs: float, line: int | float | str | None = None) -> float:
    """
    Schätzt Busstunden aus Fahrtenzahl und angenommener Umlaufzeit.

    Die Funktion verwendet die in der Service-Policy hinterlegte
    linientypische Umlaufdauer, um aus einer Fahrtenanzahl eine
    approximierte Betriebszeit zu berechnen.

    Parameter:
        runs (float): Anzahl geplanter oder beobachteter Fahrten.
        line (int | float | str | None): Linienkennung zur Auswahl der
            passenden Umlaufzeit.

    Rückgabewerte:
        float: Geschätzte Busstunden.
    """
    policy = service_policy_for_line(line)
    return max(0.0, _to_float(runs, 0.0)) * policy.cycle_minutes / 60.0


def constrained_adaptive_runs(
    *,
    demand: float,
    baseline_runs: float,
    avg_capacity: float,
    cost_per_bus_hour: float,
    default_cost_per_bus_hour: float,
    line: int | float | str | None = None,
    hour: int | float | None = None,
    allow_new_service: bool = True,
) -> int:
    """
    Berechnet eine begrenzte adaptive Fahrtenempfehlung.

    Die Heuristik verbindet Nachfrage, vorhandenes Basisangebot,
    durchschnittliche Kapazität und Kostenparameter mit fachlichen
    Servicegrenzen. Dabei werden Komfortauslastung, Mindestproduktivität,
    Taktvorgaben, Anteil des Basisangebots und eine Obergrenze für zusätzliche
    Fahrten gemeinsam berücksichtigt.

    Parameter:
        demand (float): Erwartete Nachfrage in der betrachteten Stunde.
        baseline_runs (float): Fahrtenzahl des bestehenden Angebots.
        avg_capacity (float): Durchschnittliche Fahrzeugkapazität.
        cost_per_bus_hour (float): Aktueller Kostensatz je Busstunde.
        default_cost_per_bus_hour (float): Referenzkostensatz zur Einordnung
            des Kostendrucks.
        line (int | float | str | None): Linienkennung für die passende Policy.
        hour (int | float | None): Stunde zur Peak-Erkennung.
        allow_new_service (bool): Steuert, ob bei fehlendem Bestand neue
            Bedienung aufgebaut werden darf.

    Rückgabewerte:
        int: Empfohlene Zahl an Fahrten für die Stunde.

    Fehler/Sonderfälle:
        Bei fehlendem Basisangebot und deaktivierter Neubedienung wird direkt
        ``0`` zurückgegeben. Ungültige oder nicht endliche Eingabewerte werden
        über robuste Standardwerte abgefangen. Die Kapazität wird zusätzlich
        auf mindestens 35 Plätze begrenzt, um unrealistische Kleinstwerte zu
        vermeiden.

    Projektkontext:
        Diese Funktion ist der fachliche Kern der adaptiven Angebotsplanung und
        wird sowohl in KPI-Auswertungen als auch in Fahrplanvergleichen
        verwendet.
    """
    policy = service_policy_for_line(line)
    demand = max(0.0, _to_float(demand, 0.0))
    baseline = max(0, int(round(_to_float(baseline_runs, 0.0))))
    capacity_raw = _to_float(avg_capacity, 90.0)
    capacity = max(35.0, capacity_raw)

    cost = _to_float(cost_per_bus_hour, default_cost_per_bus_hour)
    cost_pressure = max(0.65, min(1.65, cost / max(default_cost_per_bus_hour, 1.0)))
    target_utilization = max(0.60, min(0.84, policy.target_utilization + (cost_pressure - 1.0) * 0.04))
    productive_floor = max(0.25, min(0.48, policy.productive_floor + (cost_pressure - 1.0) * 0.05))
    comfort_load = max(1.0, capacity * target_utilization)
    productive_load = max(1.0, capacity * productive_floor)

    if baseline <= 0:
        if not allow_new_service or demand <= 0:
            return 0
        required = int(math.ceil(demand / comfort_load))
        if required <= 0:
            return 0
        return min(policy.max_new_runs, required)

    peak = is_peak_hour(hour) or demand >= baseline * comfort_load * 0.82
    headway = policy.peak_max_headway_minutes if peak else policy.max_headway_minutes
    headway_floor = int(math.ceil(60 / max(5, headway)))
    headway_floor = min(headway_floor, baseline)
    baseline_floor = int(math.ceil(baseline * policy.min_baseline_share))
    min_service = max(policy.min_runs_when_active, headway_floor, baseline_floor)
    min_service = min(max(min_service, 1), baseline)

    if demand <= 0:
        return min_service

    required_for_comfort = int(math.ceil(demand / comfort_load))
    target = max(min_service, required_for_comfort)

    while target > min_service and demand / max(target, 1) < productive_load:
        target -= 1

    if target <= baseline:
        return max(min_service, target)

    overload_ratio = demand / max(baseline * comfort_load, 1.0)
    if overload_ratio <= 1.2:
        max_extra = 1
    elif overload_ratio <= 1.8:
        max_extra = 2
    else:
        max_extra = policy.max_extra_runs
    return min(target, baseline + max_extra)


def _to_float(value: object, default: float) -> float:
    """
    Konvertiert einen Wert robust in ``float``.

    Die Hilfsfunktion dient der defensiven Verarbeitung heterogener Eingaben
    aus DataFrames, Konfigurationswerten oder UI-Komponenten.

    Parameter:
        value (object): Zu konvertierender Wert.
        default (float): Rückfallwert bei ungültiger Eingabe.

    Rückgabewerte:
        float: Endlicher numerischer Wert.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number