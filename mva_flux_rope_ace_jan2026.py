import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================
# Minimum Variance Analysis (MVA)
# ACE/MFI magnetic field
#
# Event: 19 Jan 2026 ICME / possible flux rope
#
# Input:
#   AC_H0_MFI_2217540.csv
#
# Outputs:
#   outputs/mva_timeseries_ACE_2026Jan19_20.png/pdf
#   outputs/mva_hodogram_BL_BM_ACE_2026Jan19_20.png/pdf
#   outputs/mva_hodogram_BM_BN_ACE_2026Jan19_20.png/pdf
#   outputs/mva_flux_rope_ACE_2026Jan19_20.csv
#   outputs/mva_flux_rope_summary_ACE_2026Jan19_20.txt
# ============================================================

ace_file = Path("AC_H0_MFI_2217540.csv")

outdir = Path("outputs")
outdir.mkdir(exist_ok=True)

# Context window for plotting
t_context0 = pd.Timestamp("2026-01-19 18:00:00")
t_context1 = pd.Timestamp("2026-01-20 06:00:00")

# Candidate flux-rope / magnetic-obstacle interval
# Ajusta estos tiempos si quieres probar otras fronteras.
t_mva0 = pd.Timestamp("2026-01-19 21:05:00")
t_mva1 = pd.Timestamp("2026-01-20 06:00:00")

shock_l1 = pd.Timestamp("2026-01-19 18:59:00")
fd_min = pd.Timestamp("2026-01-20 01:05:00")


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
            "El archivo ACE/MFI debe incluir: |B|, vector GSE y vector GSM."
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

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)

    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c].abs() > 1e20, c] = np.nan
            df.loc[np.isclose(df[c], 9999.99), c] = np.nan

    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    df["theta_B_GSE"] = np.degrees(np.arcsin(df["Bz_GSE"] / df["B"]))
    df["phi_B_GSE"] = (
        np.degrees(np.arctan2(df["By_GSE"], df["Bx_GSE"])) + 360
    ) % 360

    return df


def run_mva(df, component_cols):
    """
    MVA using covariance matrix of B components.

    Eigenvectors are returned as columns:
      column 0 = maximum variance direction
      column 1 = intermediate variance direction
      column 2 = minimum variance direction
    """
    Bmat = df[component_cols].to_numpy(dtype=float)
    Bmat = Bmat[np.isfinite(Bmat).all(axis=1)]

    if len(Bmat) < 10:
        raise ValueError("Muy pocos puntos válidos para MVA.")

    Bmean = Bmat.mean(axis=0)
    dB = Bmat - Bmean

    cov = np.cov(dB, rowvar=False)

    # For symmetric matrix, eigh returns eigenvalues ascending
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending: max, intermediate, min
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Enforce right-handed basis: eL x eM = eN
    eL = eigvecs[:, 0]
    eM = eigvecs[:, 1]
    eN = eigvecs[:, 2]

    if np.dot(np.cross(eL, eM), eN) < 0:
        eigvecs[:, 2] *= -1

    return eigvals, eigvecs, Bmean, cov


def project_to_mva(df, component_cols, eigvecs):
    """
    Project B into MVA coordinates:
      B_L = projection onto max variance direction
      B_M = projection onto intermediate variance direction
      B_N = projection onto minimum variance direction
    """
    Bmat = df[component_cols].to_numpy(dtype=float)
    projected = Bmat @ eigvecs

    out = df.copy()
    out["B_L_maxvar"] = projected[:, 0]
    out["B_M_intvar"] = projected[:, 1]
    out["B_N_minvar"] = projected[:, 2]

    out["B_LM_transverse"] = np.sqrt(
        out["B_L_maxvar"]**2 + out["B_M_intvar"]**2
    )

    out["clock_LM_deg"] = (
        np.degrees(np.arctan2(out["B_M_intvar"], out["B_L_maxvar"])) + 360
    ) % 360

    out["clock_MN_deg"] = (
        np.degrees(np.arctan2(out["B_N_minvar"], out["B_M_intvar"])) + 360
    ) % 360

    return out


# ============================================================
# Load data
# ============================================================

ace = read_ace_mfi(ace_file)

# Use 1-min median for MVA to reduce 16-s spikes but keep event structure.
ace_1min = (
    ace.set_index("time")[
        ["B", "Bx_GSE", "By_GSE", "Bz_GSE", "Bx_GSM", "By_GSM", "Bz_GSM",
         "theta_B_GSE", "phi_B_GSE"]
    ]
    .resample("1min")
    .median()
    .reset_index()
)

context = ace_1min[
    (ace_1min["time"] >= t_context0) &
    (ace_1min["time"] <= t_context1)
].copy()

mva_interval = ace_1min[
    (ace_1min["time"] >= t_mva0) &
    (ace_1min["time"] <= t_mva1)
].dropna(subset=["Bx_GSE", "By_GSE", "Bz_GSE"]).copy()

if context.empty:
    raise ValueError("La ventana de contexto quedó vacía.")

if mva_interval.empty:
    raise ValueError("El intervalo MVA quedó vacío.")

component_cols = ["Bx_GSE", "By_GSE", "Bz_GSE"]

eigvals, eigvecs, Bmean, cov = run_mva(mva_interval, component_cols)

context_mva = project_to_mva(
    context.dropna(subset=component_cols),
    component_cols,
    eigvecs
)

mva_proj = context_mva[
    (context_mva["time"] >= t_mva0) &
    (context_mva["time"] <= t_mva1)
].copy()

lambda_max, lambda_int, lambda_min = eigvals

ratio_max_int = lambda_max / lambda_int if lambda_int != 0 else np.inf
ratio_int_min = lambda_int / lambda_min if lambda_min != 0 else np.inf

if ratio_int_min >= 5:
    quality = "good separation of minimum variance direction"
elif ratio_int_min >= 2:
    quality = "moderate separation of minimum variance direction"
else:
    quality = "poor separation; MVA minimum-variance axis is uncertain"

# In simple flux-rope MVA, the intermediate variance direction is often used
# as an approximate axis estimate, but it depends on geometry.
axis_estimate = eigvecs[:, 1]

# ============================================================
# Save outputs
# ============================================================

out_csv = outdir / "mva_flux_rope_ACE_2026Jan19_20.csv"
context_mva.to_csv(out_csv, index=False)

summary_txt = outdir / "mva_flux_rope_summary_ACE_2026Jan19_20.txt"

summary = f"""
Minimum Variance Analysis (MVA) summary
Event: 19 Jan 2026 ICME / possible flux rope
Spacecraft/data: ACE/MFI
Coordinate system: GSE
Context window: {t_context0} to {t_context1}
MVA interval: {t_mva0} to {t_mva1}
Cadence used for MVA: 1-min median from ACE/MFI

Eigenvalues, descending:
lambda_max = {lambda_max:.6g}
lambda_int = {lambda_int:.6g}
lambda_min = {lambda_min:.6g}

Eigenvalue ratios:
lambda_max / lambda_int = {ratio_max_int:.3f}
lambda_int / lambda_min = {ratio_int_min:.3f}

MVA quality note:
{quality}

Eigenvectors in GSE coordinates:
e_L max variance = [{eigvecs[0,0]: .6f}, {eigvecs[1,0]: .6f}, {eigvecs[2,0]: .6f}]
e_M int variance = [{eigvecs[0,1]: .6f}, {eigvecs[1,1]: .6f}, {eigvecs[2,1]: .6f}]
e_N min variance = [{eigvecs[0,2]: .6f}, {eigvecs[1,2]: .6f}, {eigvecs[2,2]: .6f}]

Approximate flux-rope axis candidate, using intermediate variance direction:
axis_GSE = [{axis_estimate[0]: .6f}, {axis_estimate[1]: .6f}, {axis_estimate[2]: .6f}]

Mean magnetic field over MVA interval, GSE:
<B> = [{Bmean[0]: .3f}, {Bmean[1]: .3f}, {Bmean[2]: .3f}] nT

Covariance matrix:
{cov}

Interpretation guide:
- A coherent magnetic obstacle / flux-rope interval should show organized rotation in MVA components.
- B_N_minvar should remain relatively small compared with B_L and B_M if the minimum-variance direction is meaningful.
- lambda_int/lambda_min quantifies how well the minimum-variance direction is separated.
- MVA alone does not confirm a magnetic cloud; compare with Tp/Texp, beta_p and bidirectional electrons.
"""

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})


def add_markers(ax):
    markers = [
        ("shock", shock_l1, "--", 1.5),
        ("MVA start", t_mva0, "-.", 1.4),
        ("FD min", fd_min, ":", 1.4),
        ("MVA end", t_mva1, "-.", 1.2),
    ]

    for label, t, ls, lw in markers:
        if t_context0 <= t <= t_context1:
            ax.axvline(t, color="black", linestyle=ls, lw=lw, alpha=0.75)


def format_time_axis(ax):
    locator = mdates.HourLocator(interval=1)
    formatter = mdates.DateFormatter("%b %d\n%H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


# ============================================================
# Figure 1: time series only
# ============================================================

fig_ts, axes = plt.subplots(
    6,
    1,
    figsize=(18, 18),
    sharex=True,
    gridspec_kw={
        "height_ratios": [1.0, 1.25, 1.0, 1.25, 1.0, 1.0]
    }
)

axes[0].plot(context_mva["time"], context_mva["B"], lw=1.2, label="ACE |B|, 1-min median")
axes[0].set_ylabel("B [nT]")
axes[0].legend(loc="upper left")
axes[0].grid(True, alpha=0.35)

axes[1].plot(context_mva["time"], context_mva["Bx_GSE"], lw=1.1, label="Bx GSE")
axes[1].plot(context_mva["time"], context_mva["By_GSE"], lw=1.1, label="By GSE")
axes[1].plot(context_mva["time"], context_mva["Bz_GSE"], lw=1.1, label="Bz GSE")
axes[1].axhline(0, color="black", lw=0.8, alpha=0.5)
axes[1].set_ylabel("B GSE [nT]")
axes[1].legend(loc="upper left", ncol=3)
axes[1].grid(True, alpha=0.35)

axes[2].plot(context_mva["time"], context_mva["Bz_GSM"], lw=1.3, label="Bz GSM")
axes[2].axhline(0, color="black", lw=0.8, alpha=0.5)
axes[2].set_ylabel("Bz GSM [nT]")
axes[2].legend(loc="upper left")
axes[2].grid(True, alpha=0.35)

axes[3].plot(context_mva["time"], context_mva["B_L_maxvar"], lw=1.2, label="B_L max var")
axes[3].plot(context_mva["time"], context_mva["B_M_intvar"], lw=1.2, label="B_M int var")
axes[3].plot(context_mva["time"], context_mva["B_N_minvar"], lw=1.2, label="B_N min var")
axes[3].axhline(0, color="black", lw=0.8, alpha=0.5)
axes[3].set_ylabel("B MVA [nT]")
axes[3].legend(loc="upper left", ncol=3)
axes[3].grid(True, alpha=0.35)

axes[4].plot(context_mva["time"], context_mva["B_N_minvar"], lw=1.3, label="B_N min var")
axes[4].axhline(0, color="black", lw=0.8, alpha=0.5)
axes[4].set_ylabel("B_N [nT]")
axes[4].legend(loc="upper left")
axes[4].grid(True, alpha=0.35)

axes[5].plot(context_mva["time"], context_mva["clock_LM_deg"], lw=1.2, label="clock angle in L-M plane")
axes[5].set_ylabel("Clock LM [deg]")
axes[5].set_ylim(-10, 370)
axes[5].legend(loc="upper left")
axes[5].grid(True, alpha=0.35)

for ax in axes:
    add_markers(ax)
    ax.axvspan(t_mva0, t_mva1, alpha=0.08)
    ax.set_xlim(t_context0, t_context1)

# Labels for markers on first panel
for label, t, ls, lw in [
    ("shock", shock_l1, "--", 1.5),
    ("MVA start", t_mva0, "-.", 1.4),
    ("FD min", fd_min, ":", 1.4),
    ("MVA end", t_mva1, "-.", 1.2),
]:
    if t_context0 <= t <= t_context1:
        axes[0].text(
            t,
            axes[0].get_ylim()[1] * 0.92,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=11
        )

format_time_axis(axes[-1])
axes[-1].set_xlabel("Time [UTC]")

fig_ts.suptitle(
    "ACE/MFI Minimum Variance Analysis: candidate flux-rope interval\n"
    f"MVA interval: {t_mva0:%Y-%m-%d %H:%M} – {t_mva1:%Y-%m-%d %H:%M} UTC",
    y=0.995,
    fontsize=20
)

fig_ts.tight_layout(rect=[0, 0, 1, 0.985])

ts_png = outdir / "mva_timeseries_ACE_2026Jan19_20.png"
ts_pdf = outdir / "mva_timeseries_ACE_2026Jan19_20.pdf"

fig_ts.savefig(ts_png, dpi=220, bbox_inches="tight")
fig_ts.savefig(ts_pdf, bbox_inches="tight")
plt.show()


# ============================================================
# Figure 2: isolated hodogram B_L vs B_M
# ============================================================

fig_h1, ax = plt.subplots(figsize=(8, 8))

ax.plot(
    mva_proj["B_L_maxvar"],
    mva_proj["B_M_intvar"],
    lw=1.2,
    marker="o",
    markersize=2,
    label="MVA interval"
)

ax.scatter(
    mva_proj["B_L_maxvar"].iloc[0],
    mva_proj["B_M_intvar"].iloc[0],
    s=90,
    marker="s",
    label="start"
)

ax.scatter(
    mva_proj["B_L_maxvar"].iloc[-1],
    mva_proj["B_M_intvar"].iloc[-1],
    s=90,
    marker="X",
    label="end"
)

# Add arrows every N points to show direction
step = max(1, len(mva_proj) // 20)
for i in range(0, len(mva_proj) - step, step):
    x0 = mva_proj["B_L_maxvar"].iloc[i]
    y0 = mva_proj["B_M_intvar"].iloc[i]
    x1 = mva_proj["B_L_maxvar"].iloc[i + step]
    y1 = mva_proj["B_M_intvar"].iloc[i + step]
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.5)
    )

ax.axhline(0, color="black", lw=0.8, alpha=0.5)
ax.axvline(0, color="black", lw=0.8, alpha=0.5)
ax.set_xlabel("B_L max variance [nT]")
ax.set_ylabel("B_M intermediate variance [nT]")
ax.set_title("MVA hodogram: B_L vs B_M")
ax.legend(loc="best")
ax.grid(True, alpha=0.35)
ax.set_aspect("equal", adjustable="datalim")

fig_h1.tight_layout()

h1_png = outdir / "mva_hodogram_BL_BM_ACE_2026Jan19_20.png"
h1_pdf = outdir / "mva_hodogram_BL_BM_ACE_2026Jan19_20.pdf"

fig_h1.savefig(h1_png, dpi=220, bbox_inches="tight")
fig_h1.savefig(h1_pdf, bbox_inches="tight")
plt.show()


# ============================================================
# Figure 3: isolated hodogram B_M vs B_N
# ============================================================

fig_h2, ax = plt.subplots(figsize=(8, 8))

ax.plot(
    mva_proj["B_M_intvar"],
    mva_proj["B_N_minvar"],
    lw=1.2,
    marker="o",
    markersize=2,
    label="MVA interval"
)

ax.scatter(
    mva_proj["B_M_intvar"].iloc[0],
    mva_proj["B_N_minvar"].iloc[0],
    s=90,
    marker="s",
    label="start"
)

ax.scatter(
    mva_proj["B_M_intvar"].iloc[-1],
    mva_proj["B_N_minvar"].iloc[-1],
    s=90,
    marker="X",
    label="end"
)

for i in range(0, len(mva_proj) - step, step):
    x0 = mva_proj["B_M_intvar"].iloc[i]
    y0 = mva_proj["B_N_minvar"].iloc[i]
    x1 = mva_proj["B_M_intvar"].iloc[i + step]
    y1 = mva_proj["B_N_minvar"].iloc[i + step]
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.5)
    )

ax.axhline(0, color="black", lw=0.8, alpha=0.5)
ax.axvline(0, color="black", lw=0.8, alpha=0.5)
ax.set_xlabel("B_M intermediate variance [nT]")
ax.set_ylabel("B_N minimum variance [nT]")
ax.set_title("MVA hodogram: B_M vs B_N")
ax.legend(loc="best")
ax.grid(True, alpha=0.35)
ax.set_aspect("equal", adjustable="datalim")

fig_h2.tight_layout()

h2_png = outdir / "mva_hodogram_BM_BN_ACE_2026Jan19_20.png"
h2_pdf = outdir / "mva_hodogram_BM_BN_ACE_2026Jan19_20.pdf"

fig_h2.savefig(h2_png, dpi=220, bbox_inches="tight")
fig_h2.savefig(h2_pdf, bbox_inches="tight")
plt.show()


# ============================================================
# Final report
# ============================================================

print("\nSaved files:")
print(ts_png)
print(ts_pdf)
print(h1_png)
print(h1_pdf)
print(h2_png)
print(h2_pdf)
print(out_csv)
print(summary_txt)
print("\nDone.")
