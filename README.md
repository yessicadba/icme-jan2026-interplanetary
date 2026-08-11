# Interplanetary medium — Jan 19, 2026 ICME event

Analysis of the interplanetary coronal mass ejection (ICME) / candidate magnetic
cloud observed on 19 January 2026, combining solar-wind plasma and interplanetary
magnetic field (IMF) measurements with ground-based cosmic-ray count rates from
the LAGO network.

## Contents

### Scripts

- **`mc_task14_overview_omni_ace_cr.py`** — main overview figure. Seven-panel
  plot (Vsw, Tp with Tp/Texp, Np, plasma beta, |B|, Bx/By/Bz GSE, cosmic rays)
  spanning 2026-01-19 to 2026-01-23, with event boundaries (shock, ICME
  start/end) taken from an external ICME catalog entry. Reads OMNI 1-min data,
  ACE/MFI 16s data (shifted +31 min for L1→Earth propagation lag), and four
  LAGO cosmic-ray stations. Outputs `mc_task14_overview_omni_ace_cr.png`.

- **`mva_flux_rope_ace_jan2026.py`** — Minimum Variance Analysis (MVA) on
  ACE/MFI 16s magnetic field data for the event. Produces the field time
  series in the MVA (L, M, N) frame plus BL–BM and BM–BN hodograms, saved
  under `outputs/`.

- **`mva_hodogram_Bxstar_Bystar_only.py`** — paper-style MVA hodogram
  (Bx\* vs By\*, maximum vs. intermediate variance components) for a chosen
  interval. Saved under `outputs/`.

Both MVA scripts read `AC_H0_MFI_2217540.csv` (a separate ACE/MFI 16s
download from the one used by the overview script — kept as originally used
by each script rather than reprocessed to a single file).

### Data

| File | Source | Cadence | Used by |
|---|---|---|---|
| `omni_min_2026.dat` | OMNI | 1 min | overview |
| `AC_H0_MFI_4092410.csv` | ACE/MFI (GSE + GSM) | 16 s | overview |
| `AC_H0_MFI_2217540.csv` | ACE/MFI (GSE + GSM) | 16 s | MVA scripts |
| `datos_assu_15min.csv` | LAGO — Asunción, Paraguay | 15 min | overview |
| `marambio_15min.csv` | LAGO — Marambio, Antarctica | 15 min | overview |
| `san_martin_15min.csv` | LAGO — San Martín, Antarctica | 15 min | overview |
| `tanca_bin30min.csv` | LAGO — Campinas, Brazil | 30 min | overview |

## Running

```bash
python3 mc_task14_overview_omni_ace_cr.py
python3 mva_flux_rope_ace_jan2026.py
python3 mva_hodogram_Bxstar_Bystar_only.py
```

Each script expects the data files above in the same directory (`BASE` /
relative paths hard-coded at the top of each script) and writes its output
figure(s) next to it or into an `outputs/` subdirectory.

## Notes

This repository intentionally tracks only this curated subset — the scripts
and inputs needed to reproduce the overview figure and the MVA hodogram
analysis for the Jan 19, 2026 event. It excludes the many exploratory
scripts, intermediate figures, and large per-window MVA scan outputs
(`outputs_mva_scan*/`) from the broader working directory.
