#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math

# Unified material database
MATERIALS = {
    "OFHC Copper": {"A_MPa": 90.0, "B_MPa": 292.0, "n": 0.31, "Hv": 60.0, "E_GPa": 124.0, "nu": 0.34},
    "Armco Iron": {"A_MPa": 175.0, "B_MPa": 380.0, "n": 0.32, "Hv": 80.0, "E_GPa": 207.0, "nu": 0.29},
    "AISI 4340 Steel": {"A_MPa": 792.0, "B_MPa": 510.0, "n": 0.26, "Hv": 339.0, "E_GPa": 205.0, "nu": 0.29},
    "ASTM A36 Steel": {"A_MPa": 286.0, "B_MPa": 500.0, "n": 0.228, "Hv": 164.0, "E_GPa": 200.0, "nu": 0.29},
    "Ti-6Al-4V": {"A_MPa": 1090.0, "B_MPa": 1092.0, "n": 0.93, "Hv": 349.0, "E_GPa": 114.0, "nu": 0.34},
}

TOOL = {"name": "WC-Co (Tungsten Carbide Tool)", "E_GPa": 600.0, "nu": 0.22}

HV_TO_MPA = 9.807
C = 0.25
DEFAULT_EPSILON = 0.045


def gpa_to_pa(E_gpa: float) -> float:
    return E_gpa * 1e9


def cbrt(x: float) -> float:
    return x ** (1.0 / 3.0)


def calc_h_mm(r_mm: float, sigma_mpa: float, hv_hv: float) -> tuple[float, float, float]:
    hv_mpa = hv_hv * HV_TO_MPA
    ratio = 0.33 * hv_mpa / sigma_mpa
    h_mm = 2.0 * r_mm * (1.0 - ratio)
    return h_mm, ratio, hv_mpa


def calc_sigma_mpa(a_mpa: float, b_mpa: float, n: float, epsilon: float) -> float:
    return a_mpa + b_mpa * (epsilon**n)


def calc_delta_m(
    cutting_force_n: float,
    r_mm: float,
    tool_E_pa: float,
    tool_nu: float,
    work_E_pa: float,
    work_nu: float,
) -> float:
    r_m = r_mm * 1e-3
    compliance = (1.0 - tool_nu**2) / tool_E_pa + (1.0 - work_nu**2) / work_E_pa
    inside = (9.0 * cutting_force_n**2) / (16.0 * r_m)
    return compliance * cbrt(inside)


def calc_R0(r_mm: float, kr_deg: float, fz_mm_per_tooth: float) -> float:
    kr = math.radians(kr_deg)
    sin_kr = math.sin(kr)
    if abs(sin_kr) < 1e-12:
        raise ValueError("k_r cannot be 0 or 180 degrees.")
    sin2_kr = sin_kr * sin_kr
    sin_2kr = math.sin(2 * kr)
    cot_kr = math.cos(kr) / sin_kr
    inside = (r_mm**2) - (fz_mm_per_tooth**2) + (2.0 * r_mm * fz_mm_per_tooth * cot_kr)
    if inside < 0:
        raise ValueError("Invalid geometry: sqrt argument < 0. Check r, k_r, and f_z.")
    return 0.5 * fz_mm_per_tooth * sin_2kr + r_mm * sin2_kr - sin2_kr * math.sqrt(inside)


def calculate_theory_ra(
    workpiece: str,
    r_mm: float,
    kr_deg: float,
    fz_mm_per_tooth: float,
    cutting_force_n: float,
    epsilon: float = DEFAULT_EPSILON,
) -> dict:
    if workpiece not in MATERIALS:
        raise ValueError(f"Unknown material: {workpiece}")
    if r_mm <= 0:
        raise ValueError("Tool nose radius r must be > 0.")
    if fz_mm_per_tooth <= 0:
        raise ValueError("f_z must be > 0.")
    if cutting_force_n <= 0:
        raise ValueError("Cutting force F must be > 0.")
    if kr_deg <= 0 or kr_deg >= 180:
        raise ValueError("k_r must be between 0 and 180 degrees.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    props = MATERIALS[workpiece]
    sigma_mpa = calc_sigma_mpa(props["A_MPa"], props["B_MPa"], props["n"], epsilon)

    h_mm, ratio, hv_mpa = calc_h_mm(r_mm, sigma_mpa, props["Hv"])
    tool_E = gpa_to_pa(TOOL["E_GPa"])
    work_E = gpa_to_pa(props["E_GPa"])
    delta_m = calc_delta_m(cutting_force_n, r_mm, tool_E, TOOL["nu"], work_E, props["nu"])
    delta_mm = delta_m * 1e3
    r0_mm = calc_R0(r_mm, kr_deg, fz_mm_per_tooth)
    delta_h_mm = h_mm - delta_mm
    ra_mm = C * (r0_mm + delta_h_mm)

    return {
        "material": workpiece,
        "r_mm": r_mm,
        "kr_deg": kr_deg,
        "fz_mm_per_tooth": fz_mm_per_tooth,
        "cutting_force_n": cutting_force_n,
        "epsilon": epsilon,
        "A_MPa": props["A_MPa"],
        "B_MPa": props["B_MPa"],
        "n": props["n"],
        "sigma_MPa": sigma_mpa,
        "h_um": h_mm * 1e3,
        "delta_um": delta_mm * 1e3,
        "delta_h_um": delta_h_mm * 1e3,
        "R0_um": r0_mm * 1e3,
        "Ra_um": ra_mm * 1e3,
        "ratio_hv_sigma": ratio,
        "hv_mpa": hv_mpa,
        "C": C,
        "warning_h_non_positive": ratio >= 1.0,
    }


def _prompt_float(label: str) -> float:
    return float(input(label).strip())


def _prompt_int(label: str) -> int:
    return int(input(label).strip())


def run_cli() -> None:
    print("Select workpiece:")
    names = list(MATERIALS.keys())
    for i, name in enumerate(names, start=1):
        print(f"{i}. {name}")

    sel = _prompt_int(f"Workpiece (1-{len(names)}): ")
    if sel < 1 or sel > len(names):
        raise ValueError("Workpiece selection out of range.")
    workpiece = names[sel - 1]

    r_mm = _prompt_float("Tool nose radius r (mm): ")
    kr_deg = _prompt_float("k_r (deg): ")

    mode = input("f_z known? (1=yes f_z, 2=use F,n,z): ").strip()
    if mode == "1":
        fz = _prompt_float("f_z (mm/tooth): ")
    elif mode == "2":
        feed_mm_min = _prompt_float("Feed rate F (mm/min): ")
        spindle_rpm = _prompt_float("Spindle speed n (rpm): ")
        flutes = _prompt_int("Number of teeth z: ")
        if spindle_rpm <= 0 or flutes <= 0:
            raise ValueError("Spindle speed n and teeth z must be > 0.")
        fz = feed_mm_min / (spindle_rpm * flutes)
        print(f"Computed f_z = {fz:.6f} mm/tooth")
    else:
        raise ValueError("mode must be 1 or 2")

    cutting_force_n = _prompt_float("Cutting force F (N): ")
    result = calculate_theory_ra(
        workpiece=workpiece,
        r_mm=r_mm,
        kr_deg=kr_deg,
        fz_mm_per_tooth=fz,
        cutting_force_n=cutting_force_n,
    )

    print("\n=== Results (all in um) ===")
    print(f"Material : {result['material']}")
    print("\n--- Relationship ---")
    print("delta_h = h - delta")
    print("Ra = C * (R0 + delta_h)")
    print(f"C = {result['C']:.3f}")
    print("\n--- Values ---")
    print(f"h       = {result['h_um']:.3f} um")
    print(f"delta   = {result['delta_um']:.3f} um")
    print(f"delta_h = {result['delta_h_um']:.3f} um")
    print(f"R0      = {result['R0_um']:.3f} um")
    print(f"Ra      = {result['Ra_um']:.3f} um")
    if result["warning_h_non_positive"]:
        print("\nWARNING: 0.33*Hv/sigma >= 1, h may become <= 0.")


if __name__ == "__main__":
    run_cli()
