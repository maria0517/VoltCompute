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
├── server.py                  # [Rol 4/5: Backend API] FastAPI (/health, /api/evaluate, /api/dispatch, /api/simulation)
├── app.py                     # [Rol 5: Frontend & Pitch] Dashboard Streamlit în 3 tab-uri + grafice Plotly
├── tests/
│   └── test_robustness.py     # Suite Pytest (date corupte, confidență < 0.75, protecție termică > 25°C)
├── data/
│   └── energy_datacenter_data.csv  # Set de date generat automat la rularea train.py
├── models/
│   ├── model_surplus.pkl      # Modelul XGBRegressor antrenat
│   └── metrics.json           # Comparația MAE / RMSE (DummyRegressor vs XGBRegressor)
├── Dockerfile                 # Configurație container Docker
├── .dockerignore              # Fișiere excluse din build-ul Docker
└── requirements.txt
```

---

## 🛡️ Ciclul Decizional al Agentului (`VoltComputeAgent`)

1. **`perceive()`**: Validează telemetria primită de la rețeaua PPC (`GridTelemetry`) și de la centrul universitar (`ResearchCenterTelemetry`) folosind Pydantic v2. Orice valoare coruptă declanșează imediat `status="DATA_ERROR"` cu `0.0 MW` alocați.
2. **`estimate()`**: Folosește modelul `models/model_surplus.pkl` (cu ancoră fizică și fallback determinist dacă modelul nu e încă compilat) pentru a prezice surplusul peste limita de 10 MW, scorul de confidență (`0.0 - 1.0`) și contribuțiile explicabile (`top_shap_features`).
3. **`decide()`**: Aplică regulile stricte de siguranță industrială:
   - **Dacă `confidence_score < 0.75`** $\rightarrow$ `ESCALATE_TO_HUMAN` (oprire alocare automată, escaladare către dispecerul PPC).
   - **Dacă `cooling_temp_celsius > 25.0°C`** $\rightarrow$ `NORMAL_GRID_OPERATION` (protecție termică pentru serverele universitare).
   - **Dacă `surplus_predicted_mw > 2.0 MW`** $\rightarrow$ `OFFER_GENERATED` la **30.0 EUR/MWh** pentru `min(surplus_predicted_mw, max_absorb_capacity_mw)`.
4. **`act()`**: Formatează payload-ul de execuție pentru SCADA PPC și orchestratorul HPC, incluzând suport pentru confirmarea **Human-in-the-Loop** (`APPROVED` / `REJECTED`).

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
