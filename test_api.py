"""Quick API smoke test (uses the classical engine, no Qiskit needed)."""
from fastapi.testclient import TestClient
import main

c = TestClient(main.app)

h = c.get("/health").json()
print("health:", h)

for mode in ["off", "steady", "burst"]:
    r = c.post("/run", json={"mode": mode, "strength": 0.5, "noise_type": "depol",
                             "noise_level": 0.04, "n": 2000, "engine": "classical",
                             "seed": 7}).json()
    print(f"{mode:7s} qber={r['q']:.3f} pe={r['pe']:.3f} truth={r['truth']} engine={r['engine']}")
    assert len(r["bq"]) == 8 and all(k in r for k in ("q", "qz", "qx", "pe", "truth"))

print("ALL TESTS PASSED")
