"""
Gluco-X — Response Time Pipeline
==================================
Reusable module that parses raw app event logs and attributes glucose
measurements to nudges based on the following rules:

  1. A measurement counts as a "response" if it occurs within 4 hours
     of a nudge sent to the same patient.
  2. If the patient received multiple nudges before logging, the
     measurement is credited to the most recent nudge only.
  3. Each nudge can receive at most one response credit.

Usage
-----
    from glucox_pipeline import NudgeResponsePipeline

    pipeline = NudgeResponsePipeline("data/app_logs.jsonl", "data/patient_registry.csv")
    pipeline.parse()
    pipeline.attribute()

    df = pipeline.get_attribution_table()   # pandas DataFrame ready for analysis
    df.to_csv("nudge_attribution.csv", index=False)

AI Disclosure
-------------
Used Claude (Anthropic) to assist with initial code scaffolding and docstrings.
All attribution logic was reviewed and validated manually.

Author : Miguel Ángel Palomino
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


# Config

RESPONSE_WINDOW = timedelta(hours=4)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Pipeline

class NudgeResponsePipeline:
    """Parse app logs and attribute measurements to nudges."""

    def __init__(self, logs_path: str, registry_path: str):
        self.logs_path = Path(logs_path)
        self.registry_path = Path(registry_path)

        # internal storage filled during parse()
        self._nudges: list[dict] = []
        self._measurements: list[dict] = []
        self._patients: pd.DataFrame = pd.DataFrame()

        # filled during attribute()
        self._nudge_responses: dict[str, float] = {}  # event_id -> response_time_min
        self._attribution_df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Step 1 — Parse
    # ------------------------------------------------------------------
    def parse(self) -> "NudgeResponsePipeline":
        """Read the JSONL event log and the patient registry CSV."""
        logger.info("Parsing logs from %s", self.logs_path)

        with open(self.logs_path, "r") as fh:
            for line in fh:
                record = json.loads(line)
                ts = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")

                if record["event_type"] == "nudge_sent":
                    self._nudges.append({
                        "event_id": record["event_id"],
                        "patient_id": record["patient_id"],
                        "timestamp": ts,
                        "nudge_type": record["payload"]["nudge_type"],
                    })
                elif record["event_type"] == "measurement_logged":
                    self._measurements.append({
                        "event_id": record["event_id"],
                        "patient_id": record["patient_id"],
                        "timestamp": ts,
                        "glucose_value": record["payload"]["glucose_value"],
                    })

        self._patients = pd.read_csv(self.registry_path)

        logger.info(
            "Parsed %d nudges, %d measurements, %d patients",
            len(self._nudges), len(self._measurements), len(self._patients),
        )
        return self

    # ------------------------------------------------------------------
    # Step 2 — Attribute
    # ------------------------------------------------------------------
    def attribute(self) -> "NudgeResponsePipeline":
        """
        For each measurement, find the most recent nudge sent to the same
        patient within the preceding 4-hour window and credit it.

        - Each measurement is credited to at most one nudge.
        - Each nudge can receive at most one response (the first qualifying
          measurement).
        """
        logger.info("Attributing measurements to nudges …")

        # index nudges by patient, sorted chronologically
        nudge_lookup: dict[str, list[dict]] = defaultdict(list)
        for n in sorted(self._nudges, key=lambda x: x["timestamp"]):
            nudge_lookup[n["patient_id"]].append(n)

        # process measurements in chronological order
        sorted_measurements = sorted(self._measurements, key=lambda m: m["timestamp"])

        for meas in sorted_measurements:
            patient_nudges = nudge_lookup.get(meas["patient_id"], [])

            # walk backwards: most recent nudge first
            best: Optional[dict] = None
            for nudge in reversed(patient_nudges):
                if nudge["timestamp"] > meas["timestamp"]:
                    continue
                delta = meas["timestamp"] - nudge["timestamp"]
                if delta > RESPONSE_WINDOW:
                    break
                best = nudge
                break

            if best is not None and best["event_id"] not in self._nudge_responses:
                response_min = (meas["timestamp"] - best["timestamp"]).total_seconds() / 60.0
                self._nudge_responses[best["event_id"]] = response_min

        logger.info("Attributed %d measurements to nudges", len(self._nudge_responses))
        self._build_attribution_df()
        return self

    # ------------------------------------------------------------------
    # Build output table
    # ------------------------------------------------------------------
    def _build_attribution_df(self) -> None:
        """Assemble the final attribution DataFrame."""
        df = pd.DataFrame(self._nudges)
        df.sort_values(["patient_id", "timestamp"], inplace=True)

        df["responded"] = df["event_id"].isin(self._nudge_responses)
        df["response_time_min"] = df["event_id"].map(self._nudge_responses)

        # sequential nudge counters (useful downstream for fatigue analysis)
        df["nudge_seq"] = df.groupby("patient_id").cumcount() + 1
        df["nudge_seq_by_type"] = (
            df.groupby(["patient_id", "nudge_type"]).cumcount() + 1
        )

        # merge demographics
        df = df.merge(self._patients, on="patient_id", how="left")
        self._attribution_df = df

    # ------------------------------------------------------------------
    # Public accessor
    # ------------------------------------------------------------------
    def get_attribution_table(self) -> pd.DataFrame:
        """Return the attribution DataFrame for downstream analysis."""
        if self._attribution_df.empty:
            raise RuntimeError("Run .parse().attribute() before accessing the table.")
        return self._attribution_df.copy()



# CLI entry point

if __name__ == "__main__":
    pipeline = NudgeResponsePipeline(
        logs_path="data/app_logs.jsonl",
        registry_path="data/patient_registry.csv",
    )
    pipeline.parse().attribute()

    df = pipeline.get_attribution_table()
    df.to_csv("nudge_attribution.csv", index=False)

    # quick summary
    total = len(df)
    responded = df["responded"].sum()
    print(f"\nTotal nudges:   {total}")
    print(f"Responded:      {responded}")
    print(f"Response rate:  {responded / total:.1%}")
    print("\nBy nudge type:")
    print(df.groupby("nudge_type")["responded"].agg(["sum", "count", "mean"]).to_string())
    print(f"\nAttribution table saved to nudge_attribution.csv")
