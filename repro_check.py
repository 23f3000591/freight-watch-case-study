import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["output.csv", "output_all_weeks.csv", "cost_log.json"]
hashes = []
for i in (1, 2, 3):
    out = ROOT / "outputs" / f"repro_run{i}"
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "freight_watch.pipeline", "--out", str(out)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    hashes.append({f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in FILES})

ok = all(h == hashes[0] for h in hashes)
for f in FILES:
    print(f"{f:22s}", " ".join(h[f][:12] for h in hashes))
print("REPRODUCIBLE: 3 runs byte-identical" if ok else "MISMATCH between runs")
sys.exit(0 if ok else 1)
