import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# MVA hodograms only: Bx* vs By*
# ACE/MFI magnetic field
#
# Event: 19 Jan 2026 ICME / possible flux rope
#
# Purpose:
#   Produce paper-style MVA hodograms:
#       Bx* vs By*
#
# Here:
#   Bx* = maximum variance component
#   By* = intermediate variance component
#   Bz* = minimum variance component
#
# Input:
#   AC_H0_MFI_2217540.csv
#
# Outputs:
#   outputs/mva_hodogram_Bxstar_Bystar_<interval>.png/pdf
#   outputs/mva_summary_Bxstar_Bystar_intervals.txt
# ============================================================

ace_file = Path("AC_H0_MFI_2217540.csv")
outdir = Path("outputs")
outdir.mkdir(exist_ok=True)

# Candidate intervals.
# These are deliberately focused on the possible magnetic obstacle / flux rope,
# not on the full shock + sheath + ejecta sequence.
intervals = [
    ("I1_2105_0600", "2026-01-19 21:05:00", "2026-01-20 06:00:00"),
    ("I2_2120_0600", "2026-01-19 21:20:00", "2026-01-20 06:00:00"),
    ("I3_2130_0600", "2026-01-19 21:30:00", "2026-01-20 06:00:00"),
    ("I4_2200_0600", "2026-01-19 22:00:00", "2026-01-20 06:00:00"),
    ("I5_0000_0600", "2026-01-20 00:00:00", "2026-01-20 06:00:00"),
]

# Cadence for MVA/hodogram.
# Use 5 min to make the curve comparable to paper-style hodograms.
# Change to "1min" if you want more detail.
resample_rule = "5min"


def find_epoch_header(path):
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el archivo: {path.resolve()}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("EPOCH"):
                return i

    raise ValueError(f"No encontré encabezado EPOCH en {path}")


def read_ace_mfi(path):
    skiprows = find_epoch_header(path)

    df = pd.read_csv(path, skiprows=skiprows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)

    if len(cols) < 8:
        raise ValueError(
            "El archivo ACE/MFI debe incluir |B|, vector GSE y vector GSM."
        )

    df = df.rename(columns={
        cols[0]: "time",
        cols[1]: "B",
        cols[2]: "Bx_GSE",
        cols[3]: "By_GSE",
        cols[4]: "Bz_GSE",
        cols[5]: "Bx_GSM",
        cols[6]: "By_GSM",
        cols[7]: "Bz_GSM",
    })

    df["time"] = pd.to_datetime(
        df["time"], utc=True, errors="coerce"
    ).dt.tz_convert(None)

    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c].abs() > 1e20, c] = np.nan
            df.loc[np.isclose(df[c], 9999.99), c] = np.nan

    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    return df


def run_mva(df, component_cols=("Bx_GSE", "By_GSE", "Bz_GSE")):
    B = df[list(component_cols)].to_numpy(dtype=float)
    B = B[np.isfinite(B).all(axis=1)]

    if len(B) < 10:
        raise ValueError("Muy pocos puntos válidos para MVA.")

    Bmean = B.mean(axis=0)
    dB = B - Bmean
    cov = np.cov(dB, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)

    # Descending order: max, intermediate, min
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Right-handed basis
    if np.dot(np.cross(eigvecs[:, 0], eigvecs[:, 1]), eigvecs[:, 2]) < 0:
        eigvecs[:, 2] *= -1

    return eigvals, eigvecs, Bmean, cov


def project_to_star(df, eigvecs, component_cols=("Bx_GSE", "By_GSE", "Bz_GSE")):
    B = df[list(component_cols)].to_numpy(dtype=float)
    Bstar = B @ eigvecs

    out = df.copy()
    out["Bx_star"] = Bstar[:, 0]  # max variance
    out["By_star"] = Bstar[:, 1]  # intermediate variance
    out["Bz_star"] = Bstar[:, 2]  # minimum variance

    return out


def plot_hodogram(df_star, label, t0, t1, eigvals, eigvecs, outdir):
    lam_max, lam_int, lam_min = eigvals

    ratio_max_int = lam_max / lam_int if lam_int != 0 else np.inf
    ratio_int_min = lam_int / lam_min if lam_min != 0 else np.inf

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    ax.plot(
        df_star["Bx_star"],
        df_star["By_star"],
        lw=1.4,
        marker="o",
        markersize=3,
        label="MVA interval"
    )

    ax.scatter(
        df_star["Bx_star"].iloc[0],
        df_star["By_star"].iloc[0],
        s=80,
        marker="D",
        label="start"
    )

    ax.scatter(
        df_star["Bx_star"].iloc[-1],
        df_star["By_star"].iloc[-1],
        s=90,
        marker="X",
        label="end"
    )

    # Fewer direction arrows, paper-like
    step = max(1, len(df_star) // 10)
    for i in range(0, len(df_star) - step, step):
        x0 = df_star["Bx_star"].iloc[i]
        y0 = df_star["By_star"].iloc[i]
        x1 = df_star["Bx_star"].iloc[i + step]
        y1 = df_star["By_star"].iloc[i + step]

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.45)
        )

    ax.axhline(0, color="black", lw=0.8, alpha=0.7)
    ax.axvline(0, color="black", lw=0.8, alpha=0.7)

    ax.set_xlabel(r"$B_x^*$ [nT]")
    ax.set_ylabel(r"$B_y^*$ [nT]")

    duration_h = (pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds() / 3600.0

    ax.set_title(
        f"MVA hodogram: {label}\n"
        f"{pd.Timestamp(t0):%Y-%m-%d %H:%M} – {pd.Timestamp(t1):%Y-%m-%d %H:%M} UTC "
        f"({duration_h:.1f} h)"
    )

    ax.text(
        0.02,
        0.03,
        rf"$\lambda_2/\lambda_3$ = {ratio_int_min:.2f}" + "\n" +
        rf"$\lambda_1/\lambda_2$ = {ratio_max_int:.2f}",
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
    )

    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()

    png = outdir / f"mva_hodogram_Bxstar_Bystar_{label}.png"
    pdf = outdir / f"mva_hodogram_Bxstar_Bystar_{label}.pdf"

    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf, ratio_max_int, ratio_int_min


# ============================================================
# Main
# ============================================================

ace = read_ace_mfi(ace_file)

# Use resampled data only for the hodogram. This makes it easier
# to compare with published paper-style hodograms.
ace_rs = (
    ace.set_index("time")[["B", "Bx_GSE", "By_GSE", "Bz_GSE", "Bx_GSM", "By_GSM", "Bz_GSM"]]
    .resample(resample_rule)
    .median()
    .reset_index()
)

summary_lines = []
summary_lines.append("MVA Bx* vs By* hodogram summary\n")
summary_lines.append(f"Input: {ace_file}\n")
summary_lines.append(f"Cadence used: {resample_rule} median\n")
summary_lines.append(
    "Notation: Bx* = maximum variance component, "
    "By* = intermediate variance component, "
    "Bz* = minimum variance component.\n"
)
summary_lines.append(
    "Important: these are MVA/local coordinates, not original GSE/GSM Bx and By.\n"
)

created_files = []

for label, start, stop in intervals:
    t_start = pd.Timestamp(start)
    t_stop = pd.Timestamp(stop)

    df_int = ace_rs[
        (ace_rs["time"] >= t_start) &
        (ace_rs["time"] <= t_stop)
    ].dropna(subset=["Bx_GSE", "By_GSE", "Bz_GSE"]).copy()

    if len(df_int) < 10:
        print(f"Skipping {label}: too few points.")
        continue

    eigvals, eigvecs, Bmean, cov = run_mva(df_int)
    df_star = project_to_star(df_int, eigvecs)

    csv_out = outdir / f"mva_Bxstar_Bystar_{label}.csv"
    df_star.to_csv(csv_out, index=False)

    png, pdf, ratio_max_int, ratio_int_min = plot_hodogram(
        df_star,
        label,
        t_start,
        t_stop,
        eigvals,
        eigvecs,
        outdir
    )

    created_files.extend([png, pdf, csv_out])

    duration_h = (t_stop - t_start).total_seconds() / 3600.0

    summary_lines.append("\n" + "=" * 70 + "\n")
    summary_lines.append(f"Interval: {label}\n")
    summary_lines.append(f"Start: {t_start}\n")
    summary_lines.append(f"Stop:  {t_stop}\n")
    summary_lines.append(f"Duration: {duration_h:.2f} h\n")
    summary_lines.append(f"Number of points: {len(df_int)}\n")
    summary_lines.append(f"Eigenvalues: {eigvals[0]:.6g}, {eigvals[1]:.6g}, {eigvals[2]:.6g}\n")
    summary_lines.append(f"lambda_max/lambda_int: {ratio_max_int:.3f}\n")
    summary_lines.append(f"lambda_int/lambda_min: {ratio_int_min:.3f}\n")
    summary_lines.append(
        f"e_xstar/maxvar GSE: [{eigvecs[0,0]: .5f}, {eigvecs[1,0]: .5f}, {eigvecs[2,0]: .5f}]\n"
    )
    summary_lines.append(
        f"e_ystar/intvar GSE: [{eigvecs[0,1]: .5f}, {eigvecs[1,1]: .5f}, {eigvecs[2,1]: .5f}]\n"
    )
    summary_lines.append(
        f"e_zstar/minvar GSE: [{eigvecs[0,2]: .5f}, {eigvecs[1,2]: .5f}, {eigvecs[2,2]: .5f}]\n"
    )
    summary_lines.append(
        f"Mean B GSE: [{Bmean[0]:.3f}, {Bmean[1]:.3f}, {Bmean[2]:.3f}] nT\n"
    )

summary_txt = outdir / "mva_summary_Bxstar_Bystar_intervals.txt"

with open(summary_txt, "w", encoding="utf-8") as f:
    f.writelines(summary_lines)

print("\nSaved files:")
for f in created_files:
    print(f)
print(summary_txt)
print("\nDone.")
