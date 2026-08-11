import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

BASE = "/home/oem/Desktop/interplanetary"

T0 = datetime(2026, 1, 19, 0, 0, 0)
T1 = datetime(2026, 1, 23, 0, 0, 0)

# ============================================================
# OMNI (1-min): Vsw, Np, Tp, beta
# ============================================================
def load_omni_min(fname):
    cols = ["year", "doy", "hour", "minute", "Vsw", "Np", "Tp", "Pdyn", "beta"]
    df = pd.read_csv(fname, sep=r"\s+", header=None, names=cols, engine="python")
    dt = (pd.to_datetime(df["year"].astype(int).astype(str), format="%Y")
          + pd.to_timedelta(df["doy"] - 1, unit="D")
          + pd.to_timedelta(df["hour"], unit="h")
          + pd.to_timedelta(df["minute"], unit="m"))
    df["datetime"] = dt
    thr = {"Vsw": 99999, "Np": 999, "Tp": 9999999, "Pdyn": 99, "beta": 999}
    for c, t in thr.items():
        df.loc[df[c] >= t * 0.999, c] = np.nan
    return df.set_index("datetime").sort_index()

omni = load_omni_min(f"{BASE}/omni_min_2026.dat").loc[T0:T1]


def texp_richardson_cane(vsw):
    """Empirical expected temperature in K (Richardson & Cane).
    Vsw < 500 km/s: Texp = 1000*0.031*(Vsw-258)^2
    Vsw >= 500 km/s: Texp = 1000*(0.51*Vsw-142)
    """
    v = np.asarray(vsw, dtype=float)
    texp = np.full_like(v, np.nan)
    low = np.isfinite(v) & (v < 500)
    high = np.isfinite(v) & (v >= 500)
    texp[low] = 1000.0 * 0.031 * (v[low] - 258.0) ** 2
    texp[high] = 1000.0 * (0.51 * v[high] - 142.0)
    return texp


omni["Texp"] = texp_richardson_cane(omni["Vsw"])
omni["Tp_over_Texp"] = omni["Tp"] / omni["Texp"]

# ============================================================
# ACE MAG (16s): |B|, Bx,By,Bz GSE
# ============================================================
def load_ace_mfi_16s(fname):
    rows = []
    with open(fname) as f:
        for line in f:
            if line.startswith("#") or not line.strip() or line.startswith("EPOCH"):
                continue
            rows.append(line.strip().split(","))
    cols = ["epoch", "Bmag", "Bx_gse", "By_gse", "Bz_gse", "Bx_gsm", "By_gsm", "Bz_gsm"]
    df = pd.DataFrame(rows, columns=cols)
    df["datetime"] = pd.to_datetime(df["epoch"], format="%Y-%m-%dT%H:%M:%S.%fZ")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Bmag", "Bx_gse", "By_gse", "Bz_gse", "Bx_gsm", "By_gsm", "Bz_gsm"]:
        df.loc[df[c].abs() > 1000, c] = np.nan
    return df.set_index("datetime").sort_index()

ACE_L1_TO_EARTH_LAG = pd.Timedelta(minutes=31)

ace = load_ace_mfi_16s(f"{BASE}/AC_H0_MFI_4092410.csv")
ace.index = ace.index + ACE_L1_TO_EARTH_LAG
ace = ace.loc[T0:T1]

# ============================================================
# Cosmic rays: ASSU, MARAMBIO, SAN MARTIN (native 15-min),
# TANCA (30-min, the only one with barometric correction available
# -> interpolated onto the 15-min grid)
# ============================================================
def load_cr_15min(fname, datecol, valcol):
    df = pd.read_csv(fname)
    df["datetime"] = pd.to_datetime(df[datecol], utc=True).dt.tz_localize(None)
    df = df.set_index("datetime").sort_index()
    return df[valcol].loc[T0:T1]

assu = load_cr_15min(f"{BASE}/datos_assu_15min.csv", "data", "Contagem_corr")
marambio = load_cr_15min(f"{BASE}/marambio_15min.csv", "Fecha", "Cuentas_corr")
sanmartin = load_cr_15min(f"{BASE}/san_martin_15min.csv", "Fecha", "Cuentas_corr")
tanca_30 = load_cr_15min(f"{BASE}/tanca_bin30min.csv", "data", "rate_corrigido_Hz")

# common 15-min grid
grid = pd.date_range(T0, T1, freq="15min")
assu = assu.reindex(assu.index.union(grid)).interpolate(limit=2).reindex(grid)
marambio = marambio.reindex(marambio.index.union(grid)).interpolate(limit=2).reindex(grid)
sanmartin = sanmartin.reindex(sanmartin.index.union(grid)).interpolate(limit=2).reindex(grid)
# Tanca/Campinas: plain 15-min resample of the file as-is, no interpolation/padding
tanca = tanca_30.resample("15min").mean().reindex(grid)


def clean_outliers(s, window=5, n_mad=8.0):
    """Remove ONLY isolated point spikes (1-2 samples) via
    comparison with a short-window median + robust MAD.
    Short window (5 samples = 1.25h) and loose threshold (8 MAD) so as
    not to confuse a real sustained change (Forbush decrease) with an
    outlier: in a real decrease, immediate neighbors move
    together, so the residual relative to the local median stays
    small; only a truly isolated point exceeds the threshold."""
    med = s.rolling(window, center=True, min_periods=3).median()
    resid = (s - med).abs()
    mad = resid.rolling(window, center=True, min_periods=3).median() * 1.4826
    # minimum MAD floor to avoid over-flagging outliers in very flat/quiet stretches
    mad_floor = s.std() * 0.05
    mad = mad.clip(lower=mad_floor)
    thresh = n_mad * mad
    out = s.copy()
    out[resid > thresh] = np.nan
    n_removed = int((resid > thresh).sum())
    print(f"  outliers removed: {n_removed} / {s.notna().sum()} valid points")
    return out


def normalize_pct(s, base_start, base_end):
    base = s.loc[base_start:base_end]
    base_mean = base.mean()
    return (s - base_mean) / base_mean * 100.0


baseline_start = T0
baseline_end = datetime(2026, 1, 19, 18, 0)

cr_stations = {}
for name, series in [("Asunción, Paraguay", assu), ("Campinas, Brazil", tanca), ("Marambio, Antarctica", marambio), ("San Martín, Antarctica", sanmartin)]:
    cleaned = clean_outliers(series)
    normed = normalize_pct(cleaned, baseline_start, baseline_end)
    cr_stations[name] = normed

# ============================================================
# style and figure
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 17,
    "axes.linewidth": 2.3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 2.1,
    "ytick.major.width": 2.1,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.labelsize": 21,
    "legend.fontsize": 13,
    "axes.edgecolor": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

ACCENT = "red"
BLUE = "blue"
GREEN = "green"
DARK = "black"

# Boundaries from the ICME catalog entry for this event:
# Disturbance = shock; ICME plasma/field start/end = envelope; MC start/end not given.
shock_time = datetime(2026, 1, 19, 19, 17)
icme_start = datetime(2026, 1, 19, 21, 0)
env_start = icme_start
env_end = datetime(2026, 1, 22, 16, 0)

fig, axes = plt.subplots(7, 1, figsize=(15, 26), sharex=True,
                          gridspec_kw={"hspace": 0.22, "height_ratios": [1, 1, 1, 1, 1, 1, 2.2]})
ax_vsw, ax_tp, ax_np, ax_beta, ax_bmag, ax_bcomp, ax_cr = axes

ax_vsw.plot(omni.index, omni["Vsw"], color=DARK, linewidth=1.6)
ax_vsw.set_ylabel("Vsw\n(km/s)", fontsize=21)

ax_tp.plot(omni.index, omni["Tp"], color=DARK, linewidth=1.6, label=r"$T_p$")
ax_tp.set_ylabel("Tp\n(K)", fontsize=21)
ax_tp.set_yscale("log")
ax_tp.set_ylim(1e3, 1e7)

ax_tp_r = ax_tp.twinx()
ax_tp_r.plot(omni.index, omni["Tp_over_Texp"], color=ACCENT, linewidth=1.6, label=r"$T_p/T_{exp}$")
ax_tp_r.axhline(1.0, color=ACCENT, linewidth=1.6, linestyle=":", alpha=0.8)
ax_tp_r.axhline(0.5, color=ACCENT, linewidth=1.6, linestyle="--", alpha=0.8)
ax_tp_r.set_ylabel(r"$T_p/T_{exp}$", fontsize=21, color=ACCENT)
ax_tp_r.set_yscale("log")
ax_tp_r.set_ylim(1e-2, 1e2)
ax_tp_r.tick_params(axis="y", colors=ACCENT, direction="in", labelsize=16)
ax_tp_r.spines["right"].set_color(ACCENT)
ax_tp_r.spines["right"].set_linewidth(2.3)

lines1, labels1 = ax_tp.get_legend_handles_labels()
lines2, labels2 = ax_tp_r.get_legend_handles_labels()
ax_tp.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=12, frameon=False)

ax_np.plot(omni.index, omni["Np"], color=DARK, linewidth=1.6)
ax_np.set_ylabel("Np\n(N/cm$^3$)", fontsize=21)

ax_beta.plot(omni.index, omni["beta"], color=DARK, linewidth=1.6)
ax_beta.set_yscale("log")
ax_beta.axhline(1.0, color="gray", linewidth=1.5, linestyle=":")
ax_beta.set_ylabel("Beta\n(plasma)", fontsize=21)

ax_bmag.plot(ace.index, ace["Bmag"], color=DARK, linewidth=1.4)
ax_bmag.set_ylabel("|B|\n(nT)", fontsize=21)

ax_bcomp.plot(ace.index, ace["Bx_gse"], color=DARK, linewidth=1.4, label=r"$B_x$")
ax_bcomp.plot(ace.index, ace["By_gse"], color=ACCENT, linewidth=1.4, label=r"$B_y$")
ax_bcomp.plot(ace.index, ace["Bz_gse"], color=BLUE, linewidth=1.4, label=r"$B_z$")
ax_bcomp.axhline(0, color="gray", linewidth=1.5, linestyle=":")
ax_bcomp.set_ylabel("B (GSE)\n(nT)", fontsize=21)
ax_bcomp.legend(loc="upper right", fontsize=13, frameon=False, ncol=3)

cr_colors = {"Asunción, Paraguay": DARK, "Campinas, Brazil": ACCENT, "Marambio, Antarctica": BLUE, "San Martín, Antarctica": GREEN}
for name, series in cr_stations.items():
    valid = series.dropna()
    ax_cr.plot(valid.index, valid.values, color=cr_colors[name], linewidth=1.8, label=name)
ax_cr.axhline(0, color="gray", linewidth=1.5, linestyle=":")
ax_cr.set_ylabel("Cosmic rays\n(% vs. baseline)", fontsize=21)
ax_cr.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.85,
             edgecolor="0.7", ncol=2)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(2.3)
    ax.margins(x=0)
    ax.axvspan(env_start, env_end, color="0.92", zorder=0)
    ax.axvline(icme_start, color="black", linewidth=1.1, linestyle="-.")
    ax.axvline(env_end, color="black", linewidth=1.1, linestyle="-.")
    ax.axvline(shock_time, color="black", linewidth=1.1, linestyle="-.")

ax_vsw.text(shock_time, 0.95, "Shock", transform=ax_vsw.get_xaxis_transform(),
            rotation=90, ha="right", va="top", fontsize=13, color="black", clip_on=True)

axes[-1].set_xlabel("Date and time (2026)", fontsize=21)
axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
fig.autofmt_xdate(rotation=0, ha="center")

fig.suptitle("OMNI (1-min) + ACE MAG (16s, GSE, shifted +31 min L1->Earth) + Cosmic rays (15-min, normalized) - Jan 19 to 23, 2026",
              fontsize=20, y=0.998)

out = f"{BASE}/mc_task14_overview_omni_ace_cr.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved:", out)
