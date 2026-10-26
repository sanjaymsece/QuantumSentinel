# QuantumSentinel

BB84 quantum-key-distribution simulator with an ML intrusion detector and a
browser operations console.

The project is a complete local stack: a vectorised BB84 simulation, three
calibrated noise channels, intercept-resend eavesdropping, a trained
scikit-learn detector, a FastAPI service, and a single-page dashboard that
plots real per-block QBER. The detector never receives eavesdropper ground
truth; labels are returned separately only so the demo can show when the
traditional 11% rule missed an attack.

![QuantumSentinel dashboard](docs/image.png)

## Quick start

Requirements: Python 3.12, pip, a running PostgreSQL server, and internet
access for the Chart.js and IBM Plex CDN assets used by the page.

Create the archive database once (the API creates its table automatically):

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h 127.0.0.1 `
  -c "CREATE DATABASE quantum_sentinel"
```

The default connection string is
`postgresql://postgres@127.0.0.1:5432/quantum_sentinel`. Override it with the
`QS_DATABASE_URL` environment variable when your PostgreSQL user has a
password or lives on another host:

```powershell
$env:QS_DATABASE_URL = "postgresql://postgres:secret@127.0.0.1:5432/quantum_sentinel"
```

Then start the stack:

```powershell
cd "C:\Users\Sanjay M\Desktop\Quantum Sentinel"

# The workspace already contains .venv. To recreate it:
# py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt

cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

If PowerShell blocks `Activate.ps1`, call `..\.venv\Scripts\python.exe`
directly instead of activating the environment.

Open <http://127.0.0.1:8010>. The backend serves the dashboard and the API
from the same origin. Interactive API documentation is at
<http://127.0.0.1:8010/docs>.

If port 8010 is also in use, pick another free port and use the same URL for
the dashboard. No frontend configuration is needed because the dashboard uses
same-origin requests when served by the backend.

### Open the HTML directly

`index.html` can also be opened from disk (`file://`). It then defaults to the
API at `http://localhost:8000`. If the API runs on another port, append an
override:

```text
file:///C:/Users/Sanjay%20M/Desktop/Quantum%20Sentinel/index.html?api=http://127.0.0.1:8010
```

CORS is enabled for `localhost`/`127.0.0.1` origins and for the `null` origin
sent by local files.

## What was fixed

The original page contained a placeholder "ML" branch that simply read the
selected Eve mode, and a random chart generator. Both are gone.

- The backend now runs a real BB84 simulation with per-qubit state
  probabilities, basis sifting, calibrated channel noise, and two Eve
  strategies.
- The detector is a trained, calibrated classifier over post-sift
  observations. It has no access to Eve mode, Eve strength, or the attack
  label.
- The chart plots the session's actual 20-block QBER series.
- `channel` is now wired through to the backend (it was previously unused).
- The 11% rule and the ML verdict are computed independently. The UI can show
  both a rule false positive (benign high noise) and a rule false negative
  (burst attack with low session-average QBER).
- The backend serves the dashboard, exposes OpenAPI docs, validates inputs,
  and returns structured 4xx/5xx errors instead of silently mocking a result.

## Dashboard behaviour

- **Eve Off / Steady / Burst** — selects the simulator's eavesdropper.
- **Eve strength** — steady mode: per-qubit interception probability. Burst
  mode: burst length (roughly 3%–30% of the session).
- **Noise channel** — depolarizing, amplitude damping, or phase damping.
- **Noise level** — nominal post-sift QBER in percent; the channel parameter is
  calibrated to that value.
- **Qubits per session** — 100–5000 transmitted qubits.
- **Engine** — `Classical` uses the in-process NumPy channel model;
  `Qiskit` derives the same channel error probabilities with Qiskit density
  matrices and Kraus operators.
- **Run Session** — POSTs the configuration to `/run`, renders the QBER,
  standard-rule card, ML card, block chart, and event log.

Card semantics:

| Observation | Standard-rule card | ML card |
|---|---|---|
| `standard_detected = true` | ATTACK DETECTED | classifier verdict |
| rule misses, `attack_present = true` | SAFE (VULNERABLE) | classifier verdict |
| rule misses, no attack | SAFE (SECURE) | classifier verdict |

The ML card is driven only by `ml_detected` and `ml_confidence` from the
classifier. `attack_present` is simulator ground truth and is used only for
the vulnerability annotation on the standard-rule card.

## Session archive

Every successful `POST /run` is inserted into PostgreSQL before the response is
returned. A run that cannot be archived returns HTTP 503 instead of pretending
to succeed. The dashboard shows the eight newest records in **Session archive**
and refreshes the table after each run.

The `run_sessions` table stores the request configuration, the simulator result,
both detector verdicts, the 20-block QBER series, model identity, and all 33
detector features. The table and indexes are created automatically on first
startup. The archive is append-only and has no automatic retention limit; prune
old rows manually if needed.

The API binds to localhost by default. PostgreSQL credentials belong in
`QS_DATABASE_URL`; they are not written to logs or exposed by `/health`.

## API

### `GET /health`

```json
{
  "status": "ok",
  "version": "1.1.0",
  "model_loaded": true,
  "qiskit_available": true,
  "database_connected": true,
  "database_backend": "postgresql",
  "database_name": "quantum_sentinel",
  "model": {
    "name": "BB84 QuantumSentinel Detector",
    "version": "1.0.0",
    "trained_at": "2026-10-26T04:07:45+00:00",
    "threshold": 0.66
  }
}
```

`status` is `ok` only when both the detector and PostgreSQL are available.
`threshold` here is the classifier's probability threshold. The dashboard's
11% QBER threshold is returned as `threshold` in each `/run` response.

### `POST /run`

```powershell
$body = @{
  eve = "burst"
  eve_strength = 50
  noise = 4
  qubits = 2000
  channel = "depolarizing"
  engine = "js"
  seed = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/run `
  -Method Post -ContentType "application/json" -Body $body
```

```bash
curl -s http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"eve":"burst","eve_strength":50,"noise":4,"qubits":2000,"channel":"depolarizing","engine":"js","seed":1}'
```

Request fields:

| Field | Type | Notes |
|---|---|---|
| `eve` | `off`, `steady`, `burst` | required |
| `eve_strength` | integer 1–100 | ignored when `eve=off` |
| `noise` | integer 0–20 | nominal QBER percent |
| `qubits` | integer 100–5000 | session length |
| `channel` | `depolarizing`, `amplitude_damping`, `phase_damping` | required |
| `engine` | `js`, `qiskit` | `js` is the classical NumPy engine |
| `seed` | integer 0–2³²−1, optional | reproducible sessions |

Response fields include `qber`, `threshold`, `standard_detected`,
`ml_detected`, `ml_confidence`, `attack_probability`, `attack_present`,
`attack_mode`, the 20-value `block_qber` series, `sifted_bits`,
`sifted_rate`, `x_qber`, `z_qber`, `expected_qber`, `elapsed_ms`, `seed`, and
model metadata.

The session is archived before the response is sent. If PostgreSQL is down,
`/run` returns HTTP 503 and no successful response is produced.

### `GET /sessions`

```powershell
Invoke-RestMethod "http://127.0.0.1:8010/sessions?limit=20&offset=0"
Invoke-RestMethod "http://127.0.0.1:8010/sessions?ml_detected=true"
```

Returns `items`, `total`, `limit`, and `offset`, newest first. `limit` is 1–100
(default 20), `offset` is zero-based, and `ml_detected` optionally filters by
the ML verdict.

### `GET /sessions/{session_id}`

Returns the full archived record for one UUID: request fields, verdicts,
`block_qber`, model identity, and the 33-feature vector used by the detector.
Unknown IDs return 404; malformed IDs return 422.

`GET /model` returns the full model card. `GET /docs` provides the generated
OpenAPI UI. `POST /api/v1/run` and `GET /api/v1/sessions[...]` are aliases of
the unversioned endpoints.

## Simulation model

- Alice prepares random bits in randomly chosen Z or X bases.
- Bob measures in an independent random basis; only matching-basis bits are
  sifted.
- Eve, if present, performs intercept-resend. Averaged over her random basis
  and outcome, an intercepted qubit becomes
  `(1 − p/2)ρ + (p/2)I/2`, which contributes a 25% error rate at `p = 1`.
- **Steady** Eve intercepts every qubit independently with probability
  `strength/100`.
- **Burst** Eve intercepts a contiguous window at 100%; the window length is
  `3% + 27% × strength/100` of the session. A burst can therefore keep the
  session-average QBER near 8% while a few 100-qubit blocks show 20–30% QBER.
  That is the evasion path the standard rule cannot see and the ML detector is
  trained to catch.
- Channel parameters are calibrated so the average no-Eve QBER equals the
  requested noise level:
  - depolarizing: `q = 2n`
  - amplitude damping: solved from the Z/X error average
  - phase damping: `λ = 8n(1 − 2n)` (Z basis unaffected, X basis errors)
- The Qiskit engine uses `qiskit.quantum_info.DensityMatrix`, `Kraus`, and
  `Operator` to derive the per-state channel error probabilities. The random
  sampling layer is shared with the classical engine. Both engines agree to
  floating-point precision in the test suite.

## Detector

Artifact: `backend/artifacts/bb84_detector.joblib`  
Model card: `backend/artifacts/model_card.json`

- Estimator: `HistGradientBoostingClassifier` wrapped in
  `CalibratedClassifierCV` (sigmoid, 3 folds).
- 33 features: QBER, nominal noise, excess over the channel baseline, X/Z
  basis error rates and excesses, sifted count/rate, block-QBER statistics
  (mean, spread, min/max, median, p90/p95, excess, z-scores, autocorrelation,
  excess kurtosis), and channel one-hot indicators.
- Training data: 24,000 simulated sessions across all three channels, 0–20%
  noise, 100–5,000 qubits, all Eve modes, and strengths 1–100. Split
  16,800 / 3,600 / 3,600; the final model is fitted on 20,400 sessions after
  threshold selection on the validation split.
- Decision threshold: 0.66, selected on validation under a ≤2% false-positive
  budget, then evaluated once on the untouched test split.

### Held-out test performance (3,600 sessions)

| Metric | ML detector | 11% QBER rule |
|---|---:|---:|
| Accuracy | 94.53% | 66.08% |
| Precision | 99.17% | 67.31% |
| Recall | 90.78% | 74.23% |
| F1 | 94.79% | 70.60% |
| False-positive rate | 0.92% | 43.82% |
| ROC AUC | 0.9877 | — |
| Brier score | 0.0377 | — |

The ML detector caught **451 attacks the rule missed** and avoided **702 false
alarms** on the test split.

Attacks with QBER ≤ 11%: 509 sessions, **88.61% recall**, 100% precision.

Breakdown:

| Segment | ML recall | ML FPR |
|---|---:|---:|
| Steady Eve | 92.05% | — |
| Burst Eve | 89.54% | — |
| No Eve | — | 0.92% |
| 100–499 qubits | 65.81% | 2.24% |
| 500–999 qubits | 83.77% | 3.78% |
| 1,000–2,499 qubits | 91.35% | 0.61% |
| 2,500–5,000 qubits | 95.57% | 0.25% |
| Eve strength 1–10% | 55.56% | — |
| Eve strength 11–25% | 84.21% | — |
| Eve strength 26–50% | 94.63% | — |
| Eve strength 51–100% | 98.16% | — |

Detection power scales with session length and attack strength, as expected:
100-qubit sessions contain too few sifted bits to resolve a short burst
reliably. The default 2,000-qubit demo and sessions above 1,000 qubits are the
intended operating range.

### Retraining

From `backend/`:

```powershell
..\.venv\Scripts\python.exe -m app.detection.train --sessions 24000 --seed 20261026
```

The script regenerates the training data deterministically, evaluates on a
held-out split, writes both artifacts, and prints the model card summary.
Restart the API after retraining so it reloads the artifact.

## Project layout

```text
Quantum Sentinel/
├── index.html                     # dashboard (single file, no build step)
├── README.md
├── docs/
│   └── dashboard.png
├── .venv/                         # local Python 3.12 environment
└── backend/
    ├── app/
    │   ├── main.py                # FastAPI application and routes
    │   ├── config.py              # paths, thresholds, CORS, database DSN
    │   ├── db.py                  # PostgreSQL session archive
    │   ├── schemas.py             # Pydantic contracts
    │   ├── simulation/
    │   │   ├── bb84.py            # BB84 session simulation
    │   │   └── channels.py        # channel calibration + Qiskit path
    │   └── detection/
    │       ├── features.py        # 33-feature extractor
    │       ├── detector.py        # model loading and inference
    │       └── train.py           # deterministic training/evaluation
    ├── artifacts/
    │   ├── bb84_detector.joblib
    │   └── model_card.json
    ├── tests/
    ├── requirements.txt
    └── requirements-dev.txt
```

## Testing

From `backend/`:

```powershell
..\.venv\Scripts\python.exe -m pytest
```

API tests use a separate PostgreSQL database,
`quantum_sentinel_test` (override with `QS_TEST_DATABASE_URL`). It is created
automatically when the configured PostgreSQL role has `CREATEDB`; otherwise
create it once with `createdb` or `psql`. The suite truncates that table at the
start of the API tests.

77 tests cover channel calibration, analytic/Qiskit agreement, BB84
determinism and physics, feature leakage guards, detector behaviour, API
contracts, validation, CORS, the missing-model 503 path, PostgreSQL archive
writes/reads/filters/pagination, archive-outage handling, dashboard serving,
and the dashboard's required controls/API wiring.

## Limitations and operational notes

- This is a simulator, not a deployed QKD link. The detector is trained on
  simulated channels; recalibrate and retrain on measured telemetry before
  operational use.
- The nominal noise level and channel model are inputs to the detector. A
  mischaracterised channel will change its false-positive and false-negative
  behaviour.
- Very weak attacks (roughly <10% Eve strength at 2,000 qubits) are close to
  the statistical noise floor and are not always detectable.
- The Qiskit engine computes channel error probabilities with density
  matrices; it is not a full Aer circuit simulation of every qubit. The
  interface labels it as the Qiskit density-matrix engine.
- The API has no authentication and is intended to bind to `127.0.0.1` for
  local use. Add TLS, authentication, and rate limiting before exposing it.
- The page loads Chart.js and IBM Plex from CDNs. Offline use falls back to
  system fonts, but the chart requires Chart.js to be reachable.
