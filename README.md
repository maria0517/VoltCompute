# ⚡ VoltCompute — AI Arbitrage & Virtual Academic Battery for PPC

**Smarthack XI — Tema PPC: „24h to Power PPC”**

---

## 🎯 Ideea Proiectului pe Scurt

La orele prânzului (11:00–15:00), parcurile solare PPC produc frecvent mai multă energie decât poate prelua în siguranță rețeaua de transport locală (peste limita de **10.0 MW**), riscând supraîncălzirea liniilor și forțând oprirea panourilor (**curtailment / risipă de energie verde**) sau vânzarea la prețuri PZU aproape de zero / negative.

**VoltCompute** este un agent AI decizional autonom care intervine în timp real și arbitrează acest surplus energetic, redirecționându-l către **centre de cercetare universitare și noduri de calcul academic** (ex. RoEduNet, clustere HPC din universități, institute AI). Transferul se realizează la un **tarif verde fix preferențial de 30 EUR/MWh**, declanșând automat execuția sarcinilor masive de calcul (antrenare LLM, simulări fizice/climatice) din coada Slurm/Kubernetes și transformând infrastructura universitară într-o **„baterie virtuală”**.

---

## 🏗️ Arhitectura Proiectului & Împărțirea pe Roluri (Echipă de 5)

```text
VoltCompute/
├── schemas.py                 # [Rol 1: Arhitectură & Date] Contracte stricte Pydantic v2
├── train.py                   # [Rol 2: Data Science / ML] Generator date + Baseline vs XGBRegressor
├── agent.py                   # [Rol 3: AI Agent Core] Ciclul perceive() -> estimate() -> decide() -> act()
├── simulator.py               # [Rol 4: Simulare & Business KPI] Motor 24h pe 3 scenarii + impact financiar/CO2
├── services/router.py         # [Backend & Router] Clasament determinist (Match Score) + pool routing la tarif fix
├── server.py                  # [Rol 4/5: Backend API] FastAPI (/health, /api/evaluate, /api/dispatch, /api/simulation, /api/router/rank)
├── app.py                     # [Rol 5: Frontend & Pitch] Dashboard Streamlit în 3 tab-uri + grafice Plotly
├── tests/
│   ├── test_robustness.py     # Suite Pytest (date corupte, >3σ UNCERTAIN, confidență < 0.75, HVAC > 26°C, PZU >= 25)
│   └── test_router.py         # Match Score, filtru HVAC, refuz -> următorul DC la același preț, ALL_DCS_EXHAUSTED
├── data/
│   ├── energy_datacenter_data.csv  # Set de date generat automat la rularea train.py
│   └── dc_profiles_and_history.json # Profiluri DC + istoric oferte (input pentru Match Score)
├── models/
│   ├── model_surplus.pkl      # Modelul XGBRegressor antrenat
│   └── metrics.json           # Comparația MAE / RMSE (DummyRegressor vs XGBRegressor)
├── Dockerfile                 # Configurație container Docker
├── .dockerignore              # Fișiere excluse din build-ul Docker
└── requirements.txt
```

---

## 🛡️ Ciclul Decizional al Agentului (`VoltComputeAgent` + `services/router.py`)

1. **`perceive()`**: Validează telemetria (`GridTelemetry`, `ResearchCenterTelemetry`) cu Pydantic v2. Valorile corupte → `DATA_ERROR`, 0 MW alocați.
2. **`estimate()`**: XGBoost prezice surplusul (MW) pentru ora țintă (anticipare t+2h din prognoza meteo/PZU). Explicabilitate prin **TreeSHAP exact** (`pred_contribs` din XGBoost, echivalent `shap.TreeExplainer`). Senzorii cu valori **> 3σ** față de setul de antrenare → **`UNCERTAIN`** + notificare operator.
3. **`decide()`** — reguli deterministe, fără al doilea model ML (evităm compunerea erorilor):
   - `confidence_score < 0.75` → `ESCALATE_TO_HUMAN`;
   - temperatura HVAC a DC-ului **> 26°C** → nod exclus (filtru siguranță termică);
   - **orice depășire prezisă a liniei de 10 MW** → `OFFER_GENERATED` la **30 EUR/MWh**, doar pentru surplusul prezis (oferta se oprește când estimarea ajunge la limita liniei; nu vindem energie de sub linie).
   - Clasament DC: `Score = 0.4·VolumFit + 0.4·RatăAcceptareIstorică − 0.2·PenalizareTermică`.
4. **`act()`** — pool routing Human-in-the-Loop: oferta pleacă la primul clasat; la **Refuză** tariful **nu scade**, nodul devine inactiv și oferta sare la următorul DC la același preț; dacă refuză toți → **`ALL_DCS_EXHAUSTED`**. La **Aprobă** nodul trece în `COMPUTING`, iar surplusul scade pe grafic.

**Evaluare ML:** split temporal strict 80/20 (1152h train / 288h test, fără shuffle). `python train.py` afișează în consolă MAE/RMSE pentru **baseline-ul de persistență ŷ(t+2) = y(t)** vs XGBoost.

**Interfață:** un singur control temporal — **[ ⏭ Simulează Ora Următoare ]**; datele vin din fișierele de telemetrie, fără inputuri manuale. Dashboard-ul este organizat pe etapele Perceive → Estimate (grafic + bare SHAP) → Decide (tabel Match Score) → Act (carduri DC cu **Aprobă Tranzacția** / **Refuză Oferta**).

---

## 🚀 Pași de Reproducere (Quickstart)

### 1. Instalare dependențe
```bash
pip install -r requirements.txt
```

### 2. Antrenarea modelului ML și generarea datelor (`train.py`)
Generează automat `data/energy_datacenter_data.csv` (dacă lipsește), antrenează `DummyRegressor` vs `XGBRegressor` cu `train_test_split(shuffle=False)` și salvează `models/model_surplus.pkl` + `models/metrics.json`:
```bash
python train.py
```

### 3. Rularea testelor unitare de robustețe (`pytest`)
Verifică detecția datelor corupte, pragul de confidență scăzută (`< 0.75`), protecția termică (`> 25.0°C`) și simulatorul pe 24h:
```bash
pytest -v
```

### 4. Pornirea aplicației de Demo & Pitch (`Streamlit`)
```bash
streamlit run app.py
```

#### Conturi demo (pagina de logare)
| Rol | Utilizator | Parolă | Ce vede |
|---|---|---|---|
| Dispecerat PPC | `ppc` | `ppc123` | grafice, metrici, tabelul de arbitraj, control temporal (Pas / Play / timeline) |
| Data Center | `bucuresti`, `cluj`, `timisoara` | `dc123` | doar ofertele adresate lui (Aprobă / Refuză), telemetria proprie și contractele lui |

PPC și DC-urile împart aceeași stare pe server: ora o controlează PPC, iar o ofertă aprobată de un DC apare imediat la PPC (ecranele se actualizează automat la 2–3 s). Pentru demo, deschide PPC și un DC în două ferestre diferite (sau una în mod incognito).

### 5. (Opțional) Pornirea serverului FastAPI
```bash
uvicorn server:app --reload --port 8000
```
*(Documentația interactivă Swagger UI este disponibilă la `http://localhost:8000/docs`)*

---

## 🐳 Rulare cu Docker

Proiectul include un `Dockerfile` gata configurat (cu Python 3.11, biblioteca `libgomp1` pentru XGBoost, antrenarea automată a modelului și expunerea porturilor `8501` și `8000`):

```bash
# 1. Construiește imaginea Docker
docker build -t voltcompute .

# 2. Pornește containerul cu dashboard-ul Streamlit (port 8501)
docker run --rm -p 8501:8501 --name voltcompute-app voltcompute
```

După pornire, deschide în browser: **`http://localhost:8501`**

*(Opțional) Comenzi suplimentare cu Docker:*
```bash
# Rulare server FastAPI pe portul 8000
docker run --rm -p 8000:8000 voltcompute uvicorn server:app --host 0.0.0.0 --port 8000

# Rulare teste unitare pytest în container
docker run --rm voltcompute pytest -v
```
