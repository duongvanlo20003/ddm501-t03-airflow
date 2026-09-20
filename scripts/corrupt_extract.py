"""
Damage the extract, so you can watch the pipeline refuse it.

The validate task quarantines bad rows and fails the run only when more than
5% of them are bad. This script pushes the extract past that line.

Run:  python scripts/corrupt_extract.py          # break 12% of the rows
      python scripts/corrupt_extract.py --repair # put the original back
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "wdbc.csv"
BACKUP = RAW.with_suffix(".csv.orig")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--fraction", type=float, default=0.12)
    args = ap.parse_args()

    if args.repair:
        if BACKUP.exists():
            # copyfile does not try to chmod a Windows bind mount.
            shutil.copyfile(BACKUP, RAW)
            print(f"restored {RAW.name} from {BACKUP.name}")
        else:
            print("no backup found; nothing to restore")
        return

    if not BACKUP.exists():
        shutil.copyfile(RAW, BACKUP)
        print(f"kept a copy at {BACKUP.name}")

    frame = pd.read_csv(RAW)
    n = int(len(frame) * args.fraction)
    hit = frame.sample(n, random_state=1).index
    frame.loc[hit, "mean_radius"] = np.nan
    frame.to_csv(RAW, index=False)
    print(f"blanked mean_radius on {n} of {len(frame)} rows "
          f"({n / len(frame):.1%}) -- above the 5% limit")
    print("now trigger the DAG again and read the validate task's log")


if __name__ == "__main__":
    main()
