# Synthetic Healthcare Data Generation System (SDV + Fairness + Privacy)

## Overview

A microservice-based platform for generating, evaluating, and comparing synthetic healthcare datasets with integrated **quality**, **fairness**, and **privacy** assessment.

This system addresses a critical gap in healthcare data science: researchers need realistic synthetic datasets that eliminate privacy risks while preserving statistical properties and demographic fairness. The platform generates synthetic patient data using three state-of-the-art algorithms (GaussianCopula, CTGAN, TVAE) and rigorously evaluates each generator across three evaluation dimensions:

- **Quality:** Statistical fidelity via SDMetrics (column shapes, pair correlations, data validity)
- **Fairness:** Demographic parity and per-group performance via fairlearn
- **Privacy:** Membership inference risk, exact duplicate detection, and NN leakage

Users interact via a natural language goal ("Generate data balanced across gender with strong privacy") that is auto-translated to optimized parameters via LLM, then visualize results in an interactive web dashboard.

---

## Target Users

1. **Healthcare Researchers** – Academics developing fairness-aware ML models who need synthetic data without privacy constraints
2. **Data Engineers** – Practitioners building data pipelines requiring safe data sharing across teams/organizations
3. **ML Engineers** – AI practitioners evaluating model robustness and generator performance
4. **Privacy/Compliance Officers** – Organizational stakeholders verifying synthetic data safety before deployment

---

## Key Features Implemented

✅ **Multi-Algorithm Benchmarking** – Compare GaussianCopula, CTGAN, TVAE on identical metrics  
✅ **LLM-Guided Planning** – Natural language goals auto-generate optimized parameters  
✅ **Quality Evaluation** – SDMetrics: column shapes, pair correlations, data validity  
✅ **Fairness Assessment** – Demographic parity, per-group metrics (TPR, FPR) across race/gender/intersections  
✅ **Privacy Quantification** – Exact duplicates, NN leakage distance, membership inference AUC  
✅ **Downstream Performance Validation** – Train-on-Synthetic-Test-on-Real (TSTR) model evaluation  
✅ **Interactive Dashboard** – Streamlit UI with tabs for config, metrics, visualizations, exports  
✅ **Real-Time Progress Monitoring** – Live job status updates via Redis polling  
✅ **Asynchronous Job Queue** – Redis-backed worker queue for scalable processing  
✅ **Artifact Versioning** – Each run stored with unique ID; reproducible configs and results  

---

## Installation

### Prerequisites

- **Docker & Docker Compose** (version 3.8+)
  - [Install Docker Desktop](https://www.docker.com/products/docker-desktop) (includes Compose)
  - Or: [Install Docker Engine](https://docs.docker.com/engine/install/) + [Docker Compose](https://docs.docker.com/compose/install/)

- **macOS/Linux/Windows with WSL2** (Docker requirement)

- **Git** (to clone repository)

### Clone Repository

```bash
git clone <repository-url>
cd syn_microservice
```

### Download Pre-trained Models & Dataset

The system requires three pre-trained synthetic generator models and the base healthcare dataset. These are stored in the `artifacts/` directory:

```bash
# Create artifacts directory if it doesn't exist
mkdir -p artifacts

# Download pre-trained models and dataset
# (These should be provided separately or generated via external training pipeline)
# Expected files:
# - artifacts/gaussian_copula_diabetes.pkl          (~50 MB)
# - artifacts/ctgan_diabetes.pkl                    (~100 MB)
# - artifacts/tvae_diabetes.pkl                     (~150 MB)
# - artifacts/fairlearn_diabetes_hospital.pkl       (~20 MB, real dataset)
# - artifacts/diabetes_metadata.json                (~5 KB)

# Verify files exist:
ls -lh artifacts/*.pkl artifacts/*.json
```

---

## Running Locally

### Quick Start

```bash
# Start all services (api, worker, ui, redis, ollama)
docker-compose up --build

# Wait for startup (~30-60 seconds for Ollama to download llama3.1 model first run)
# Services will be available at:
#   - UI Dashboard:  http://localhost:8501
#   - API Server:    http://localhost:8000
#   - Redis:         http://localhost:6379
#   - Ollama LLM:    http://localhost:11434
```

### Manual Service Startup (Advanced)

```bash
# Terminal 1: Start infrastructure (Redis, Ollama)
docker-compose up redis ollama

# Terminal 2: Start API service
docker-compose up api

# Terminal 3: Start worker service
docker-compose up worker

# Terminal 4: Start UI dashboard
docker-compose up ui

# Verify all services running:
docker-compose ps
```

### Verify Services

```bash
# Check API health
curl http://localhost:8000/health
# Expected response: {"ok": true}

# Check dataset preview
curl http://localhost:8000/dataset/preview
# Expected response: JSON with dataset metadata, columns, preview rows

# Open UI dashboard
open http://localhost:8501
# Or navigate in browser
```

---

## Environment Variables & Configuration

### Default Configuration (via docker-compose.yml)

| Variable | Default | Purpose |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection for job queue |
| `ARTIFACT_DIR` | `/artifacts` | Shared volume for outputs, models, dataset |
| `OLLAMA_BASE_URL` | `http://ollama:11434/api` | LLM service endpoint |
| `OLLAMA_MODEL` | `llama3.1` | LLM model name |
| `API_URL` | `http://api:8000` | API server endpoint (used by UI) |

### Customizing Configuration

Edit `docker-compose.yml` to modify:
- Port mappings (e.g., change `8501:8501` to `9000:8501` for UI)
- Volume mounts (e.g., use different artifact directory)
- Environment variables (e.g., different LLM model)

```yaml
# Example: Change UI port to 9000
ui:
  ports:
    - "9000:8501"  # Old: 8501:8501
```

Restart services:
```bash
docker-compose down
docker-compose up --build
```

---

## Project Structure

```
syn_microservice/
├── docker-compose.yml              # 5-service orchestration config
├── README.md                        # This file
│
├── services/
│   ├── api/                        # FastAPI backend
│   │   ├── main.py                # Endpoints: /runs, /llm/plan, /llm/explain, /artifact, /dataset/preview
│   │   ├── llm_service.py         # Ollama LLM integration
│   │   ├── requirements.txt       # Python dependencies
│   │   └── Dockerfile             
│   │
│   ├── worker/                     # Async job processor
│   │   ├── worker.py              # Synthetic generation + evaluation pipeline
│   │   ├── requirements.txt       
│   │   └── Dockerfile             
│   │
│   └── ui/                         # Streamlit dashboard
│       ├── app.py                 # Interactive web UI
│       ├── requirements.txt       
│       └── Dockerfile             
│
└── artifacts/                      # Persistent shared volume
    ├── diabetes_metadata.json     
    ├── fairlearn_diabetes_hospital.pkl  # Real dataset
    ├── gaussian_copula_diabetes.pkl     # Pre-trained model
    ├── ctgan_diabetes.pkl               
    ├── tvae_diabetes.pkl                
    ├── [run_id_1]/                # Output from each experiment run
    │   ├── config.json            
    │   ├── metrics.json           
    │   ├── real_reference.csv     
    │   ├── synthetic/             
    │   ├── fairness/              
    │   ├── privacy/               
    │   └── reports/               
    └── [run_id_2]/, [run_id_3]/, ...
```

---

## Usage Workflow

### 1. Access Dashboard

**Local:**
```
http://localhost:8501
```

**Remote (UCI VPN required):**
```
http://172.27.135.4:8501/
```

### 2. Create Experiment (LLM-Guided)

1. Open Streamlit dashboard
2. Left sidebar: Enter natural language goal
   - Example: *"Generate balanced synthetic data with strong privacy protection"*
3. Click **"Suggest config"** button
4. LLM auto-generates optimized parameters:
   - Which generators to use (GaussianCopula, CTGAN, TVAE)
   - Number of synthetic rows
   - Privacy/fairness settings
5. Review and click **"Run Experiment"**

### 3. Monitor Job Progress

- Dashboard displays live status: `stage`, `current_model`, `progress`
- Real-time JSON status updates every 0.5 seconds
- View console logs for worker details

### 4. Explore Results

Once job completes, navigate to **"Results Explorer"** tab:

**Tabs Available:**
- **Overview** – Key metrics (Quality, Diagnostic, TSTR AUC, DP max, MIA AUC)
- **Synthetic Data** – Side-by-side real vs. synthetic preview with adjustable row count
- **Visualize** – Interactive Plotly charts (distributions, correlations, trends)
- **Trend Story** – Relationship analysis (e.g., Age → Num Medications)
- **Evaluate** – Downstream model training/testing on synthetic data
- **Raw JSON** – Detailed metrics export

### 5. Download Results

Click **"Download Results"** to export:
- Synthetic sample CSV (use for downstream analysis)
- Metrics JSON (quality, fairness, privacy scores)
- Fairness breakdown CSVs (per-group metrics)
- Privacy report JSON
- Interactive HTML plots (data validity, distributions, correlations)

---

## API Reference

### Key Endpoints

#### `POST /runs` – Create Experiment Job

**Request:**
```json
{
  "generators": ["gaussian_copula", "ctgan", "tvae"],
  "num_rows": 5000,
  "subset_rows": 10000,
  "pair_metric_rows": 5000,
  "privacy_max_n": 5000,
  "privacy_percentile": 1.0,
  "save_plots": true
}
```

**Response:**
```json
{
  "run_id": "daf9dbdd-5ba0-451e-9fb6-3d0b064037f8"
}
```

#### `GET /runs/{run_id}` – Get Job Status

**Response:**
```json
{
  "status": "running",
  "stage": "evaluating_model",
  "current_model": "GaussianCopula",
  "progress": 0.33,
  "models_done": 1,
  "models_total": 3
}
```

#### `GET /runs/{run_id}/results` – Get Completed Results

**Response:**
```json
{
  "run_id": "daf9dbdd-5ba0-451e-9fb6-3d0b064037f8",
  "full": {
    "baseline": {...},
    "models": {
      "GaussianCopula": {...},
      "CTGAN": {...},
      "TVAE": {...}
    }
  }
}
```

#### `POST /llm/plan` – Auto-Generate Parameters from Goal

**Request:**
```json
{
  "user_goal": "Generate synthetic data balanced across gender and race, minimize privacy leakage"
}
```

**Response:**
```json
{
  "plan": {
    "generators": ["gaussian_copula", "ctgan", "tvae"],
    "num_rows": 5000,
    "privacy_percentile": 0.5,
    ...
  }
}
```

#### `GET /dataset/preview` – Preview Real Data

**Response:**
```json
{
  "meta": {
    "dataset_name": "fairlearn_diabetes_hospital",
    "n_rows": 101766,
    "n_columns": 23,
    "target_column": "readmit_binary",
    "sensitive_columns": ["race", "gender"]
  },
  "columns": [...],
  "preview": [...],
  "target_distribution": {"0": 0.72, "1": 0.28}
}
```

---

## Deployment

### Local Deployment

System is designed for **local laptop/workstation** development and research use:

```bash
docker-compose up --build
# Services available at localhost:8501 (UI), localhost:8000 (API)
```

### Remote Deployment (UCI VPN)

System is hosted at **http://172.27.135.4:8501/** (requires UCI VPN access)

To deploy to remote server:

1. SSH into server
2. Clone repository
3. Copy pre-trained models and dataset to `/artifacts/`
4. Run `docker-compose up -d` (background mode)
5. Configure firewall/reverse proxy for external access

**Note:** Current setup is **single-user, no authentication**. Not recommended for multi-tenant or production healthcare environments.

---

## Known Issues & Limitations

### Known Bugs

| Issue | Severity | Workaround |
|---|---|---|
| Ollama cold start timeout on first run | Medium | Increase timeout to 120s; restart container |
| Streamlit session state lost across browser tabs | Low | Use single tab per experiment |
| Redis connection timeout if service fails | High | Restart Docker Compose; add retry logic |
| Large artifact exports (>100MB) slow | Medium | Subsample rows or implement chunked export |
| Privacy metrics all-zero for certain data distributions | Low | Verify data balance; check metric assumptions |

### System Limitations

1. **Single Dataset** – Hard-coded to diabetes hospital readmission; cannot generalize without retraining models (30–60 min on GPU)
2. **Pre-Trained Models Only** – Cannot fine-tune on user data; must accept diabetes-trained generators
3. **Binary Classification Only** – Fairness evaluation limited to binary target (readmit: yes/no)
4. **Two Sensitive Attributes** – Fairness supports only race + gender; intersections limited to race × gender
5. **Privacy Threat Model** – Metrics are relative risk indicators, not formal differential privacy guarantees
6. **Single Worker** – Jobs processed sequentially; no horizontal scaling
7. **No Authentication** – Single-user only; not suitable for multi-tenant deployment

---

## Future Work

### Near-Term (1–2 quarters)
- [ ] Multi-dataset support with model retraining pipeline
- [ ] Custom fairness attributes (age ranges, comorbidities, disability)
- [ ] Extended privacy metrics (differential privacy, linkage attacks)
- [ ] Temporal/time-series data support

### Medium-Term (2–3 quarters)
- [ ] Privacy-utility tradeoff Pareto frontier visualization
- [ ] Enterprise features (authentication, audit logging, RBAC, multi-tenancy)
- [ ] Kubernetes scaling (horizontal worker scaling)
- [ ] Alternative LLM backends (GPT-4, Claude)

### Long-Term (3+ quarters)
- [ ] Integration with Databricks, Snowflake, Kaggle
- [ ] Public benchmarking leaderboard vs. competitors
- [ ] HIPAA/GDPR compliance reporting
- [ ] Active learning for fairness optimization

---

## Troubleshooting

### Services Won't Start

```bash
# Check Docker running
docker --version

# View service logs
docker-compose logs -f api
docker-compose logs -f worker
docker-compose logs -f ui

# Restart all services
docker-compose restart
```

### UI Dashboard Not Accessible

```bash
# Verify UI container running
docker-compose ps ui

# Check port mapping
netstat -an | grep 8501  # macOS/Linux
Get-NetTCPConnection -LocalPort 8501  # Windows

# Try accessing from different port
docker-compose exec ui streamlit run app.py --server.port 9000
```

### Job Stuck in "Running" State

```bash
# Check worker logs
docker-compose logs worker

# Check Redis queue
docker-compose exec redis redis-cli LLEN jobs
docker-compose exec redis redis-cli KEYS "run:*"

# Restart worker
docker-compose restart worker
```

### Memory Issues (Large Datasets)

```bash
# Reduce synthetic row count
# Edit UI config: num_rows → 2000 (instead of 5000)

# Check Docker memory limit
docker stats

# Increase Docker memory in Desktop settings: Preferences → Resources
```

---
