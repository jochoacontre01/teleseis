import numpy as np
import scipy.io
from pathlib import Path

# Import the previously translated functions
# (Assuming they are saved as rotations.py and spectral.py in the same directory)
from rotate import nez_to_rtz, nez_to_lqt, nez_to_psvh
from spectral import decon
from plotting import compare_traces, plot_traces, map_1rf


def main():
    # ==========================================
    # 1. Load Data
    # ==========================================
    print("Loading data...")
    try:
        data = scipy.io.loadmat(Path(__file__).resolve().parents[2] / "lab6_material/Input_data.mat")
        ns_trace1 = data["ns_trace1"].flatten()
        ew_trace1 = data["ew_trace1"].flatten()
        z_trace1 = data["z_trace1"].flatten()
    except FileNotFoundError:
        print(
            f"Warning: {data.stem} not found"
        )

    # Plot raw data
    print("Plotting raw traces...")
    plot_traces(ns_trace1, ew_trace1, z_trace1, labels=["NS", "EW", "Z"])

    # ==========================================
    # 2. Rotate into all 3 coordinate systems
    # ==========================================
    baz = 78.75
    rayp = 0.07125
    vp = 3.0
    vs = 1.5

    print(f"Rotating traces (baz={baz}, rayp={rayp}, vp={vp}, vs={vs})...")
    rcomp, tcomp, zcomp_rot = nez_to_rtz(ns_trace1, ew_trace1, z_trace1, baz)
    lcomp, qcomp, tcomp_lqt = nez_to_lqt(ns_trace1, ew_trace1, z_trace1, baz, rayp, vp)
    pcomp, svcomp, shcomp = nez_to_psvh(
        ns_trace1, ew_trace1, z_trace1, baz, rayp, vp, vs
    )

    # Plot all 3 rotated components separately
    plot_traces(rcomp, tcomp, zcomp_rot, labels=["Radial", "Transverse", "Z"])
    plot_traces(lcomp, qcomp, tcomp_lqt, labels=["L", "Q", "T"])
    plot_traces(pcomp, svcomp, shcomp, labels=["P", "SV", "SH"])

    # Compare versions
    compare_traces(rcomp, qcomp, svcomp, labels=["Radial", "Q", "SV"])

    # ==========================================
    # 3. Change velocities and compare
    # ==========================================
    vp_new = 2.5
    vs_new = 1.0
    print(f"\nRe-evaluating with changed velocities (vp={vp_new}, vs={vs_new})...")

    lcomp_changev, qcomp_changev, tcomp_changev = nez_to_lqt(
        ns_trace1, ew_trace1, z_trace1, baz, rayp, vp_new
    )
    pcomp_changev, svcomp_changev, shcomp_changev = nez_to_psvh(
        ns_trace1, ew_trace1, z_trace1, baz, rayp, vp_new, vs_new
    )

    compare_traces(lcomp, lcomp_changev, labels=[f"L (Vp={vp})", f"L (Vp={vp_new})"])
    compare_traces(qcomp, qcomp_changev, labels=[f"Q (Vp={vp})", f"Q (Vp={vp_new})"])
    # shows a small change near 0
    compare_traces(
        tcomp_lqt, tcomp_changev, labels=[f"T (Vp={vp})", f"T (Vp={vp_new})"]
    )

    # I detect no differences here if I change the 3 to 2.5,
    # if I change the 1.5 to 1 then I see small differences in the P and SV components
    compare_traces(pcomp, pcomp_changev, labels=[f"P (Vs={vs})", f"P (Vs={vs_new})"])
    compare_traces(
        svcomp, svcomp_changev, labels=[f"SV (Vs={vs})", f"SV (Vs={vs_new})"]
    )
    compare_traces(
        shcomp, shcomp_changev, labels=[f"SH (Vs={vs})", f"SH (Vs={vs_new})"]
    )

    # ==========================================
    # 4. Part 2: Deconvolution
    # ==========================================
    print("\nRunning Deconvolution Testing...")
    # in the vicinity of 100 works well for the water level ...
    # really have to look at the power spectrum to see that the larger values are too large.
    dlist = [10.0, 20.0, 100.0, 200.0, 500.0, 1000.0, 10000.0]

    decon_results = []
    decon_labels = []

    # Typically in RF studies, the radial (or Q/SV) is deconvolved by the vertical (or L/P)
    signal_trace = svcomp
    source_trace = pcomp

    for water_level in dlist:
        print(f"Deconvolving with water level: {water_level}")
        # Note: decon(fseis, source, water)
        dseis = decon(signal_trace, source_trace, water_level)
        decon_results.append(dseis)
        decon_labels.append(f"Water = {water_level}")

    # Plot all the deconvolved traces together to observe the effect of the water level
    plot_traces(*decon_results, labels=decon_labels)

    # ==========================================
    # 5. Part 3: Depth Mapping
    # ==========================================
    print("\nMapping SV-P receiver function to depth...")
    # Using water level 100.0 as noted in the comments that it "works well"
    svp_rf = decon(svcomp, pcomp, 100.0)
    map_1rf(svp_rf, rayp)

if __name__ == "__main__":
    main()
