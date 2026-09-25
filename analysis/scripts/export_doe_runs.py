"""Export the campaign-v3 per-run and per-cell tables that fig-summary and the paper's
cell means are read from.

Inputs (outputs/): posthoc_v3_report.json and contact_invariance_v3.json.
Outputs (outputs/): doe_v3_runs_{hamiltonian,direct}.csv (one row per run) and
doe_v3_cells_{hamiltonian,direct}.csv (mean and spread = max-min over the 3 seeds).
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
POSTHOC_PATH = OUT_DIR / "posthoc_v3_report.json"
CONTACT_PATH = OUT_DIR / "contact_invariance_v3.json"

SUPERVISION_LABELS = {
    "dos": "DOS+T",
    "ldos": "LDOS+DOS+T",
    "ldosonly": "LDOS+T",
    "tonly": "T-only",
}

HAM_COLUMNS = [
    "run_name", "supervision", "n_orb", "num_layers", "geometry", "seed",
    "val_transmission", "l12_transmission", "l16_transmission",
    "l12_dos", "l16_dos", "onsite_near", "onsite_far",
    "ci_interior_mode", "ci_interior_total", "onsite_abs_mean",
    "onsite_std", "saved_at_epoch",
]

DIRECT_COLUMNS = [
    "run_name", "supervision", "num_layers", "geometry", "seed",
    "val_transmission", "l12_transmission", "l16_transmission",
    "l12_dos", "l16_dos", "saved_at_epoch",
]

HAM_CELL_COLUMNS = [
    "supervision", "n_orb", "num_layers", "geometry",
    "val_transmission", "val_transmission_spread",
    "l12_transmission", "l12_transmission_spread",
    "l16_transmission", "l16_transmission_spread",
    "l12_dos", "l12_dos_spread",
    "l16_dos", "l16_dos_spread",
    "onsite_near", "onsite_near_spread",
    "ci_interior_mode", "ci_interior_mode_spread",
    "ci_interior_total", "ci_interior_total_spread",
]

DIRECT_CELL_COLUMNS = [
    "supervision", "num_layers", "geometry",
    "val_transmission", "val_transmission_spread",
    "l12_transmission", "l12_transmission_spread",
    "l16_transmission", "l16_transmission_spread",
    "l12_dos", "l12_dos_spread",
    "l16_dos", "l16_dos_spread",
]

HAM_CELL_METRICS = [
    "val_transmission", "l12_transmission", "l16_transmission",
    "l12_dos", "l16_dos", "onsite_near",
    "ci_interior_mode", "ci_interior_total",
]

DIRECT_CELL_METRICS = [
    "val_transmission", "l12_transmission", "l16_transmission",
    "l12_dos", "l16_dos",
]


def geom_label(v):
    return "on" if v else "off"


def median(values):
    return statistics.median(values)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_rows():
    posthoc = load_json(POSTHOC_PATH)["runs"]  # module global; see main()
    contact = load_json(CONTACT_PATH)["runs"]

    ham_rows = []
    direct_rows = []

    ham_names_posthoc = [n for n, v in posthoc.items() if v["family"] == "ham"]
    direct_names_posthoc = [n for n, v in posthoc.items() if v["family"] != "ham"]

    missing_from_contact = sorted(set(ham_names_posthoc) - set(contact.keys()))
    missing_from_posthoc = sorted(set(contact.keys()) - set(ham_names_posthoc))
    if missing_from_contact:
        raise SystemExit(
            f"FATAL: {len(missing_from_contact)} Hamiltonian run(s) in "
            f"posthoc_v3_report.json missing from contact_invariance_v3.json: "
            f"{missing_from_contact}"
        )
    if missing_from_posthoc:
        raise SystemExit(
            f"FATAL: {len(missing_from_posthoc)} Hamiltonian run(s) in "
            f"contact_invariance_v3.json missing from posthoc_v3_report.json: "
            f"{missing_from_posthoc}"
        )

    for name in ham_names_posthoc:
        p = posthoc[name]
        c = contact[name]
        d_mode = median(c["vectors"]["D_interior_mode"])
        d_total = median(c["vectors"]["D_interior_total"])
        companion = c["companion"]
        row = {
            "run_name": name,
            "supervision": SUPERVISION_LABELS[p["supervision"]],
            "n_orb": p["n_orb"],
            "num_layers": p["num_layers"],
            "geometry": geom_label(p["geometry"]),
            "seed": p["seed"],
            "val_transmission": p["val_transmission_at_selection"],
            "l12_transmission": p["l12_transmission"],
            "l16_transmission": p["l16_transmission"],
            "l12_dos": p["l12_dos"],
            "l16_dos": p["l16_dos"],
            "onsite_near": p["onsite_near"],
            "onsite_far": p["onsite_far"],
            "ci_interior_mode": d_mode,
            "ci_interior_total": d_total,
            "onsite_abs_mean": companion["onsite_abs_mean"],
            "onsite_std": companion["onsite_std"],
            "saved_at_epoch": p.get("saved_at_epoch", ""),
        }
        ham_rows.append(row)

    for name in direct_names_posthoc:
        p = posthoc[name]
        row = {
            "run_name": name,
            "supervision": SUPERVISION_LABELS[p["supervision"]],
            "num_layers": p["num_layers"],
            "geometry": geom_label(p["geometry"]),
            "seed": p["seed"],
            "val_transmission": p["val_transmission_at_selection"],
            "l12_transmission": p["l12_transmission"],
            "l16_transmission": p["l16_transmission"],
            "l12_dos": p["l12_dos"],
            "l16_dos": p["l16_dos"],
            "saved_at_epoch": p.get("saved_at_epoch", ""),
        }
        direct_rows.append(row)

    return ham_rows, direct_rows


def build_cells(ham_rows, direct_rows):
    """Aggregate run-level rows to cell-level (mean and spread over 3 seeds)."""

    def aggregate_cell(rows_for_cell, metrics):
        if len(rows_for_cell) != 3:
            raise SystemExit(
                f"FATAL: cell aggregation expects exactly 3 seeds per cell, "
                f"got {len(rows_for_cell)}"
            )
        result = {}
        for metric in metrics:
            values = [row[metric] for row in rows_for_cell]
            result[metric] = statistics.mean(values)
            result[f"{metric}_spread"] = max(values) - min(values)
        return result

    ham_cells_dict = {}
    for row in ham_rows:
        key = (row["supervision"], row["n_orb"], row["num_layers"], row["geometry"])
        if key not in ham_cells_dict:
            ham_cells_dict[key] = []
        ham_cells_dict[key].append(row)

    ham_cells = []
    for (supervision, n_orb, num_layers, geometry), cell_rows in sorted(
        ham_cells_dict.items()
    ):
        agg = aggregate_cell(cell_rows, HAM_CELL_METRICS)
        cell_row = {
            "supervision": supervision,
            "n_orb": n_orb,
            "num_layers": num_layers,
            "geometry": geometry,
        }
        cell_row.update(agg)
        ham_cells.append(cell_row)

    direct_cells_dict = {}
    for row in direct_rows:
        key = (row["supervision"], row["num_layers"], row["geometry"])
        if key not in direct_cells_dict:
            direct_cells_dict[key] = []
        direct_cells_dict[key].append(row)

    direct_cells = []
    for (supervision, num_layers, geometry), cell_rows in sorted(
        direct_cells_dict.items()
    ):
        agg = aggregate_cell(cell_rows, DIRECT_CELL_METRICS)
        cell_row = {
            "supervision": supervision,
            "num_layers": num_layers,
            "geometry": geometry,
        }
        cell_row.update(agg)
        direct_cells.append(cell_row)

    return ham_cells, direct_cells


def is_finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sanity_check(ham_rows, direct_rows, ham_cells, direct_cells):
    assert len(ham_rows) == 96, f"expected 96 Hamiltonian rows, got {len(ham_rows)}"
    assert len(direct_rows) == 24, f"expected 24 Direct rows, got {len(direct_rows)}"

    for rows, columns, label in (
        (ham_rows, HAM_COLUMNS, "Hamiltonian"),
        (direct_rows, DIRECT_COLUMNS, "Direct"),
    ):
        for row in rows:
            for col in columns:
                val = row[col]
                if col == "saved_at_epoch":
                    if val != "" and not is_finite_number(val):
                        raise SystemExit(
                            f"FATAL: non-finite saved_at_epoch in {label} row "
                            f"{row['run_name']}: {val!r}"
                        )
                    continue
                if val is None or val == "":
                    raise SystemExit(
                        f"FATAL: empty value for {col!r} in {label} row "
                        f"{row['run_name']}"
                    )
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if not is_finite_number(val):
                        raise SystemExit(
                            f"FATAL: non-finite value for {col!r} in {label} "
                            f"row {row['run_name']}: {val!r}"
                        )

    assert len(ham_cells) == 32, f"expected 32 Hamiltonian cells, got {len(ham_cells)}"
    assert len(direct_cells) == 8, f"expected 8 Direct cells, got {len(direct_cells)}"

    for cells, columns, label in (
        (ham_cells, HAM_CELL_COLUMNS, "Hamiltonian cell"),
        (direct_cells, DIRECT_CELL_COLUMNS, "Direct cell"),
    ):
        for cell_row in cells:
            for col in columns:
                val = cell_row[col]
                if val is None or val == "":
                    raise SystemExit(
                        f"FATAL: empty value for {col!r} in {label} row "
                        f"{cell_row}"
                    )
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if not is_finite_number(val):
                        raise SystemExit(
                            f"FATAL: non-finite value for {col!r} in {label}: "
                            f"{val!r}"
                        )

    # Pinned cross-checks against the 2026-09-16 floor-20 report: a silently different input
    # file aborts here rather than exporting a plausible-looking table.
    target_name = "ham_ldosonly_n2_L4_nogeom_s1179027592"
    matches = [r for r in ham_rows if r["run_name"] == target_name]
    if not matches:
        raise SystemExit(f"FATAL: cross-check run {target_name!r} not found")
    l12t = matches[0]["l12_transmission"]
    if round(l12t, 4) != 2.5978:
        raise SystemExit(
            f"FATAL: cross-check failed for {target_name}: "
            f"l12_transmission={l12t!r} rounds to {round(l12t, 4)}, expected 2.5978"
        )
    print(f"Cross-check OK: {target_name} l12_transmission={l12t!r} (rounds to 2.5978)")

    target_cell = ("LDOS+T", 2, 4, "off")
    matches = [c for c in ham_cells if (
        c["supervision"] == target_cell[0] and
        c["n_orb"] == target_cell[1] and
        c["num_layers"] == target_cell[2] and
        c["geometry"] == target_cell[3]
    )]
    if not matches:
        raise SystemExit(
            f"FATAL: cross-check cell {target_cell} not found in ham_cells"
        )
    cell = matches[0]
    l12t_mean = cell["l12_transmission"]
    l12t_spread = cell["l12_transmission_spread"]
    if round(l12t_mean, 4) != 1.7898:
        raise SystemExit(
            f"FATAL: cell cross-check failed for {target_cell}: "
            f"l12_transmission mean={l12t_mean!r} rounds to {round(l12t_mean, 4)}, "
            f"expected 1.7898"
        )
    if round(l12t_spread, 4) != 1.3525:
        raise SystemExit(
            f"FATAL: cell cross-check failed for {target_cell}: "
            f"l12_transmission_spread={l12t_spread!r} rounds to {round(l12t_spread, 4)}, "
            f"expected 1.3525"
        )
    print(f"Cell cross-check OK: {target_cell} l12_transmission={l12t_mean!r} "
          f"(mean, rounds to 1.7898), spread={l12t_spread!r} (rounds to 1.3525)")


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    global POSTHOC_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(POSTHOC_PATH),
                    help="posthoc report to export (default: outputs/posthoc_v3_report.json)")
    ap.add_argument("--suffix", default="",
                    help="suffix inserted before .csv on every output (default: none)")
    args = ap.parse_args()
    POSTHOC_PATH = Path(args.report)
    sfx = args.suffix
    ham_rows, direct_rows = build_rows()
    ham_cells, direct_cells = build_cells(ham_rows, direct_rows)
    sanity_check(ham_rows, direct_rows, ham_cells, direct_cells)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ham_path = OUT_DIR / f"doe_v3_runs_hamiltonian{sfx}.csv"
    direct_path = OUT_DIR / f"doe_v3_runs_direct{sfx}.csv"
    write_csv(ham_path, HAM_COLUMNS, ham_rows)
    write_csv(direct_path, DIRECT_COLUMNS, direct_rows)
    print("wrote run-level CSV:")
    print(f"  {ham_path} ({len(ham_rows)} rows)")
    print(f"  {direct_path} ({len(direct_rows)} rows)")

    ham_cell_path = OUT_DIR / f"doe_v3_cells_hamiltonian{sfx}.csv"
    direct_cell_path = OUT_DIR / f"doe_v3_cells_direct{sfx}.csv"
    write_csv(ham_cell_path, HAM_CELL_COLUMNS, ham_cells)
    write_csv(direct_cell_path, DIRECT_CELL_COLUMNS, direct_cells)
    print("Wrote cell-level CSV:")
    print(f"  {ham_cell_path} ({len(ham_cells)} rows)")
    print(f"  {direct_cell_path} ({len(direct_cells)} rows)")


if __name__ == "__main__":
    sys.exit(main())
