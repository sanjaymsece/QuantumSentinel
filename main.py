"""
QuantumSentinel - BB84 Eavesdropper Detector (backend)
Q-Hack India 2026 | Team Quantum Syndicate

FastAPI backend that runs REAL BB84 quantum-circuit sessions on the Qiskit Aer
stabilizer simulator (with Pauli noise models) plus an ML eavesdropper detector.
A NumPy twin engine is included as fallback / classical baseline.
"""

import json
import pathlib

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = pathlib.Path(__file__).parent
MODEL = json.loads((BASE_DIR / "model.json").read_text())

app = FastAPI(title="QuantumSentinel API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RunRequest(BaseModel):
    mode: str = "off"          # off | steady | burst
    strength: float = 0.5      # Eve strength (0.15 - 1.0)
    noise_type: str = "depol"  # none | bitflip | dephase | depol
    noise_level: float = 0.04  # channel noise probability
    n: int = 2000              # qubits per session
    engine: str = "aer"        # aer (Qiskit) | classical (NumPy twin)
    seed: int | None = None


# ------------------------------------------------------------ shared helpers
def eve_mask(rng, mode, f, n):
    """Positions where Eve intercepts: steady = iid, burst = one contiguous window."""
    m = np.zeros(n, dtype=bool)
    if mode == "steady":
        m = rng.random(n) < f
    elif mode == "burst":
        L = int(n * f)
        if L > 0:
            s = rng.integers(0, n - L + 1)
            m[s:s + L] = True
    return m


def sift_and_features(a, bob, ba, bb, rng):
    """Basis sifting -> random half-sample -> QBER stats -> 6 ML features."""
    sift = ba == bb
    A, Bo, Bs = a[sift], bob[sift], ba[sift]
    m = len(A)
    sel = np.sort(rng.choice(m, size=m // 2, replace=False))
    er = (A[sel] != Bo[sel]).astype(float)
    bs = Bs[sel]
    q = float(er.mean())
    qz = float(er[bs == 0].mean()) if (bs == 0).any() else 0.0
    qx = float(er[bs == 1].mean()) if (bs == 1).any() else 0.0
    k = len(er)
    base, ex = divmod(k, 8)
    bq = np.array([er[i * base + min(i, ex):(i + 1) * base + min(i + 1, ex)].mean()
                   for i in range(8)])
    feats = np.array([q, qz, qx, abs(qz - qx), float(bq.std()), float(bq.max())])
    return feats, bq


def ml_infer(feats):
    z = MODEL["b"] + float(np.dot(MODEL["coef"],
                                  (feats - MODEL["mean"]) / MODEL["scale"]))
    return float(1.0 / (1.0 + np.exp(-z)))


# ------------------------------------------------------------ engines
def run_classical(req: RunRequest, rng):
    """NumPy Monte-Carlo twin of the Qiskit circuit (identical statistics)."""
    n = req.n
    a = rng.integers(0, 2, n)
    ba = rng.integers(0, 2, n)
    bb = rng.integers(0, 2, n)
    mask = eve_mask(rng, req.mode, req.strength, n)
    flip = mask & (rng.random(n) < 0.5)   # intercept-resend: random-basis measure == flip w.p. 1/2
    lv = req.noise_level
    px = lv if req.noise_type == "bitflip" else 0.0
    pz = lv if req.noise_type == "dephase" else 0.0
    pd = lv if req.noise_type == "depol" else 0.0
    xe = rng.random(n) < px
    ze = rng.random(n) < pz
    wd = rng.random(n) < pd
    xe = xe ^ (wd & (rng.random(n) < 2 / 3))  # depolarizing: X, Y, Z each w.p. pd/3
    ze = ze ^ (wd & (rng.random(n) < 2 / 3))
    err = ((ba == 0) & xe) | ((ba == 1) & ze)
    bob = a ^ flip ^ err
    return sift_and_features(a, bob, ba, bb, rng)


_simulator = None


def get_simulator():
    global _simulator
    if _simulator is None:
        from qiskit_aer import AerSimulator
        _simulator = AerSimulator(method="stabilizer")
    return _simulator


def run_qiskit(req: RunRequest, rng):
    """Real BB84 on the Aer stabilizer simulator with Pauli noise models."""
    try:
        from qiskit import transpile, QuantumCircuit
        from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Qiskit not installed: pip install qiskit qiskit-aer (or use engine='classical')",
        ) from e

    n = req.n
    a = rng.integers(0, 2, n)
    ba = rng.integers(0, 2, n)
    bb = rng.integers(0, 2, n)
    mask = eve_mask(rng, req.mode, req.strength, n)
    flip = mask & (rng.random(n) < 0.5)

    nm = NoiseModel()
    p = req.noise_level
    if p > 0:
        if req.noise_type == "bitflip":
            nm.add_all_qubit_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["id"])
        elif req.noise_type == "dephase":
            nm.add_all_qubit_quantum_error(pauli_error([("Z", p), ("I", 1 - p)]), ["id"])
        elif req.noise_type == "depol":
            nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["id"])

    sim = get_simulator()
    bob = np.zeros(n, dtype=int)
    BATCH = 512
    for start in range(0, n, BATCH):
        k = min(BATCH, n - start)
        qc = QuantumCircuit(k, k)
        for i in range(k):
            j = start + i
            if a[j]:
                qc.x(i)             # Alice bit 1
            if ba[j]:
                qc.h(i)             # Alice encodes in X basis
            qc.id(i)                # channel: every noise model attaches here
            if flip[j]:
                qc.x(i)             # Eve intercept-resend flip
            if bb[j]:
                qc.h(i)             # Bob decodes
            qc.measure(i, i)
        tqc = transpile(qc, sim)
        counts = sim.run(tqc, noise_model=nm, shots=1,
                         seed_simulator=int(rng.integers(0, 2**31))).result().get_counts()
        bitstr = next(iter(counts))  # shots=1 -> single outcome for all k qubits
        for i in range(k):
            bob[start + i] = int(bitstr[k - 1 - i])
    return sift_and_features(a, bob, ba, bb, rng)


def run_session(req: RunRequest):
    rng = np.random.default_rng(req.seed)
    if req.engine == "classical":
        feats, bq = run_classical(req, rng)
        engine = "classical (numpy twin)"
    else:
        feats, bq = run_qiskit(req, rng)
        engine = "qiskit-aer (stabilizer)"
    return {"engine": engine, "q": float(feats[0]), "qz": float(feats[1]),
            "qx": float(feats[2]), "bq": [float(x) for x in bq],
            "pe": ml_infer(feats), "truth": req.mode != "off",
            "mode": req.mode, "n": req.n}


def generate_dataset(count, seed=0, engine="aer", n=1000):
    """Used by notebooks/02 to retrain the detector on real Qiskit data."""
    rng = np.random.default_rng(seed)
    X = np.zeros((count, 6))
    y = np.zeros(count)
    for i in range(count):
        req = RunRequest(mode=str(rng.choice(["off", "steady", "burst"])),
                         strength=float(rng.uniform(0.15, 1.0)),
                         noise_type=str(rng.choice(["none", "bitflip", "dephase", "depol"])),
                         noise_level=float(rng.uniform(0.0, 0.15)),
                         n=n, engine=engine)
        r2 = np.random.default_rng(rng.integers(0, 2**31))
        feats, _ = run_classical(req, r2) if engine == "classical" else run_qiskit(req, r2)
        X[i], y[i] = feats, req.mode != "off"
    return X, y


# ------------------------------------------------------------ API
@app.get("/health")
def health():
    try:
        import qiskit
        qver = qiskit.__version__
    except ImportError:
        qver = None
    return {"status": "ok", "qiskit": qver,
            "model": {k: MODEL["metrics"][k] for k in ("accuracy", "detection", "false_alarm")}}


@app.post("/run")
def run(req: RunRequest):
    return run_session(req)
