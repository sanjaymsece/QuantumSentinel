https://sanjaymsece.github.io/QuantumSentinel/
# QuantumSentinel - BB84 Eavesdropper Detector

Q-Hack India 2026 (IBM Qiskit Fall Fest) | Team Quantum Syndicate | Quantum Security & Cryptography track

Detects eavesdroppers on BB84 quantum key-distribution channels with an ML classifier
that beats the standard 11%-QBER abort rule, even when Eve attacks in bursts or the
channel is noisy. **Every session runs as real quantum circuits on IBM Qiskit Aer.**

## Architecture

```
frontend/index.html   single-page UI (no build step - just open it)
      |  HTTP POST /run  {mode, strength, noise_type, noise_level, n, engine}
      v
backend/main.py       FastAPI
      |-- run_qiskit()     BB84 on Aer stabilizer simulator + Pauli noise models
      |-- run_classical()  NumPy Monte-Carlo twin (baseline / no-Qiskit fallback)
      |-- ml_infer()       logistic-regression detector (backend/model.json)
```

## Quickstart

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000
```
Then open `frontend/index.html` in a browser.

- Engine **Qiskit (Aer)** (default): real BB84 circuits - X/H encoding, identity-delay
  channel carrying Pauli noise, Eve intercept-resend, executed in batches of 512 qubits
  with the stabilizer method, 1 shot per batch.
- Engine **Classical (JS)**: the original in-browser Monte-Carlo twin, kept as a
  classical baseline for the demo video comparison.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | status, qiskit version, detector metrics |
| `POST /run` | one BB84 session -> `{q, qz, qx, bq[8], pe, truth, engine, ...}` |

## Retrain the detector on real Qiskit data

The shipped `backend/model.json` was trained on the statistically identical NumPy twin
(accuracy 0.980, detection 0.986, false-alarm 0.030 on 3,000 unseen sessions).
To retrain on genuine Qiskit output, run `notebooks/02_train_ml_detector.ipynb`
(ENGINE='aer') and restart uvicorn.

## Round 1 submission mapping (Q-Hack India)

| Requirement | Where |
|---|---|
| Minimal prototype | this repo: real Qiskit backend + working frontend UI |
| Frontend | `frontend/index.html` |
| Backend | `backend/main.py` (FastAPI + Qiskit Aer) |
| GitHub Repository | push this folder |
| Demo Video | screen-record: burst attack -> standard rule misses, QuantumSentinel catches it |
| Idea PPT | use the official template; numbers above are reproducible from notebook 02 |
