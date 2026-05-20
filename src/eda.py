from __future__ import annotations

from pathlib import Path

import pandas as pd

from .visualization import (
    plot_case_matrix,
    plot_cloud,
    plot_correlation_heatmap,
    plot_htd_region,
    plot_wall_curve,
)


def run_basic_eda(
    full: pd.DataFrame,
    wall: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: str | Path,
    max_cases: int = 6,
) -> None:
    """Generate the standard MVP EDA figures."""
    out_dir = Path(out_dir)
    plot_case_matrix(summary, out_dir)
    plot_wall_curve(wall, "wall_temperature", out_dir, "wall_temperature_x", "Wall temperature", max_cases)
    plot_wall_curve(wall, "nusselt_number", out_dir, "nusselt_number_x", "Nusselt number", max_cases)
    plot_wall_curve(wall, "dTw_dx", out_dir, "dTw_dx_x", "dTw/dx", max_cases)
    plot_wall_curve(wall, "dNu_dx", out_dir, "dNu_dx_x", "dNu/dx", max_cases)
    plot_wall_curve(wall, "Tw_minus_Tb", out_dir, "Tw_minus_Tb_x", "Tw - Tb", max_cases)

    case_id = str(full["case_id"].iloc[0]) if not full.empty else None
    for col in [
        "temperature",
        "u",
        "density",
        "turb_reynolds_number",
        "prandtl_number_lam",
        "omega_z",
    ]:
        plot_cloud(full, col, out_dir, f"{col}_cloud_{case_id}", case_id=case_id)
    plot_htd_region(full, out_dir, case_id=case_id)

    corr_cols = [
        "diameter",
        "inlet_temperature",
        "wall_heat_flux",
        "x",
        "r_over_R",
        "temperature",
        "u",
        "pressure",
        "density",
        "wall_temperature",
        "nusselt_number",
        "turb_reynolds_number",
        "prandtl_number_lam",
        "omega_z",
        "Tw_minus_Tb",
        "dTw_dx",
        "dNu_dx",
    ]
    joined = full.merge(
        wall[["case_id", "x", "dTw_dx", "dNu_dx"]].drop_duplicates(["case_id", "x"]),
        on=["case_id", "x"],
        how="left",
    )
    plot_correlation_heatmap(joined, corr_cols, out_dir)

