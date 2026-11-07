# 📊 VoltCompute — Benchmark ML (prognoză surplus solar)

- **Model:** `C:\Users\USER\Desktop\aiurea-work\App\AI-urea\model.joblib` · v1 (main: model t+2h fără calibrare) · `XGBRegressor` · 15 features
- **Date:** 694 ferestre orare reconstruite din CSV · test cronologic 139 ore de la `2026-05-25T03:00:00`
- **Mediu:** Python `3.13.14` · durată benchmark `2.0s` · generat `2026-11-07T19:25:37`

## Verdict: 11/12 verificări trecute

| Verificare | Valoare | Prag | Status |
|---|---|---|:---:|
| Paritate: MAE recalculat independent = MAE raportat în artefact | `0.00e+00` | < 1e-6 | ✅ |
| Bate Persistența pe toate orizonturile | `t+2: 89%` | skill > 0 | ✅ |
| Bate Baseline-ul Sezonier 24h pe toate orizonturile | `t+2: 39%` | skill > 0 | ✅ |
| R² pe test la t+2h | `0.9854` | > 0.90 | ✅ |
| Walk-forward: bate Persistența în toate fold-urile | `4/4` | toate | ✅ |
| Walk-forward: bate Sezonierul în majoritatea fold-urilor | `4/4` | >= 3/4 | ✅ |
| Incertitudine calibrată (interval + confidență) | `indisponibil` | prezent | ⚠️ N/A |
| Latență predict 1 rând (p95) | `1.28 ms` | < 50 ms | ✅ |
| Timp antrenare (aceeași configurație) | `0.07 s` | < 15 s | ✅ |
| Reantrenare deterministă | `0.0e+00` | == 0 | ✅ |
| Zgomot 10% pe senzorii meteo → degradare MAE | `+5.5%` | < 50% | ✅ |
| Fail-closed pe calea de inferență (NaN / Inf / lipsă / limite fizice) | `7/7` | toate | ✅ |
| Agentul folosește efectiv modelul ML la runtime | `None` | xgboost | ❌ |

## 1. Acuratețe vs. baseline (set test cronologic)

| Orizont | Rânduri | Model MAE | RMSE | R² | Persistență MAE | Sezonier MAE | Skill vs Pers. | Skill vs Sez. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t+2h | 139 | **0.00650** | 0.01408 | 0.9854 | 0.05828 | 0.01057 | 88.9% | 38.5% |

> Skill = 1 − MAE_model / MAE_baseline. Predicțiile sunt tăiate la 0 MW (export negativ imposibil).

## 2. Erori pe regimuri (după ora livrării t+2)

| Regim | Rânduri | Export mediu MW | Model MAE | Persistență MAE | Sezonier MAE |
|---|---:|---:|---:|---:|---:|
| varf_solar_11_15 | 30 | 0.2647 | **0.01850** | 0.12952 | 0.03303 |
| rampe_8_10_16_18 | 36 | 0.0536 | **0.00932** | 0.10277 | 0.01323 |
| noapte_19_7 | 73 | 0.0000 | **0.00017** | 0.00707 | 0.00002 |

Cele mai dificile ore de livrare: `15:00` (0.0277 MW), `14:00` (0.0270 MW), `16:00` (0.0233 MW)

## 3. Detecția depășirilor (surplus > prag la t+2)

| Prag | MW | Evenimente reale | Prezise | Precision | Recall | F1 | F1 baseline sezonier |
|---|---:|---:|---:|---:|---:|---:|---:|
| limita_retea_0.35MW | 0.3500 | 4 | 3 | 0.333 | 0.250 | 0.286 | 0.250 |
| p75_export_pozitiv | 0.2857 | 16 | 12 | 1.000 | 0.750 | 0.857 | 0.875 |

## 4. Calibrarea incertitudinii & confidența

> ⚠️ Modelul nu expune interval de predicție / confidență calibrată (lipsește `calibration`). Gate-ul de siguranță de 75% nu poate fi fundamentat statistic pe acest model.

## 5. Backtest walk-forward (expanding window, reantrenare per fold)

| Fold | Train | Test | Perioadă test | Model MAE | Persistență MAE | Sezonier MAE |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 345 | 86 | 2026-05-16T11 → 2026-05-20T00 | **0.00882** | 0.06191 | 0.01037 |
| 2 | 431 | 86 | 2026-05-20T01 → 2026-05-23T14 | **0.00569** | 0.05980 | 0.00773 |
| 3 | 517 | 86 | 2026-05-23T15 → 2026-05-27T04 | **0.00605** | 0.05260 | 0.00673 |
| 4 | 603 | 89 | 2026-05-27T05 → 2026-05-30T21 | **0.00629** | 0.06129 | 0.01229 |
| **Medie** | | | | **0.00671 ± 0.00124** | 0.05890 | 0.00928 |

## 6. Robustețe

MAE referință: `0.00650` MW · cale inferență testată: model.predict (nevalidat)

| Zgomot multiplicativ pe senzorii meteo | MAE | Degradare |
|---|---:|---:|
| σ = 5% | 0.00644 | -0.9% |
| σ = 10% | 0.00685 | +5.5% |
| σ = 20% | 0.00745 | +14.7% |

| Test fail-closed | Rezultat |
|---|:---:|
| valid_input_accepted | ✅ |
| missing_feature_rejected | ✅ |
| nan_rejected | ✅ |
| inf_rejected | ✅ |
| out_of_bounds_radiatie_solara_wm2_rejected | ✅ |
| out_of_bounds_temp_c_rejected | ✅ |
| out_of_bounds_viteza_vant_kmh_rejected | ✅ |

## 7. Importanța atributelor

| # | Permutation (Δ MAE pe test, MW) | | TreeSHAP global pe train (mean abs, MW) | |
|---:|---|---:|---|---:|
| 1 | `hour_cos` | 0.08002 | `hour_cos` | 0.07652 |
| 2 | `radiatie_solara_wm2` | 0.00753 | `radiatie_solara_wm2` | 0.00994 |
| 3 | `hour` | 0.00443 | `hour` | 0.00480 |
| 4 | `surplus_lag_24` | 0.00211 | `surplus_lag_2` | 0.00157 |
| 5 | `Surplus_National_MW` | 0.00088 | `pret_pzu_eur_mwh` | 0.00139 |
| 6 | `temp_c` | 0.00078 | `surplus_lag_24` | 0.00130 |
| 7 | `hour_sin` | 0.00074 | `pret_lag_1` | 0.00127 |
| 8 | `pret_lag_1` | 0.00072 | `hour_sin` | 0.00112 |
| 9 | `surplus_lag_2` | 0.00067 | `surplus_lag_1` | 0.00109 |
| 10 | `pret_pzu_eur_mwh` | 0.00050 | `Surplus_National_MW` | 0.00089 |

## 8. Performanță & reproductibilitate

| Metrică | mean / p50 / p95 / p99 |
|---|---:|
| Predict 1 rând | 1.26 / 1.16 / 1.28 / 1.42 ms |
| TreeSHAP 1 rând | 1.60 / 1.58 / 1.65 / 2.29 ms |

| Metrică | Valoare |
|---|---:|
| Throughput batch | 102,820 rânduri/s |
| Timp antrenare (aceeași configurație) | 0.07 s |
| Diferență între 2 reantrenări | 0.00e+00 |
| Diferență reantrenare vs. artefact salvat | 0.00e+00 |

## 9. Integrare cu agentul VoltCompute

| Metrică | Valoare |
|---|---|
| Model încărcat de agent | nu |
| Apeluri reușite către model | 0 |
| Sursa predicției în agent | `None` |
| Predicție directă model / agent (MW) | — / — |

> ❌ **Agentul NU folosește modelul ML** — deciziile se bazează pe fallback-ul fizic. Cauză tipică: nume de features diferite între artefact și `agent.py`.
