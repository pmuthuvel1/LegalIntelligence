"""Quick validator for input_examples/ and output_examples/ JSON files."""
import json
import os
import sys

ok = True
for folder in ["input_examples", "output_examples"]:
    files = sorted(os.listdir(folder))
    print(f"\n=== {folder} ({len(files)} files) ===")
    for name in files:
        if not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = list(data.keys())
            first = keys[0] if keys else None
            uc = data.get("use_case_id")
            good = first == "use_case_id" and uc == 21
            mark = "OK " if good else "BAD"
            if not good:
                ok = False
            print(f"  {mark}  {name:50s}  first_key={first}  use_case_id={uc}")
        except Exception as e:
            ok = False
            print(f"  ERR {name}: {e}")

sys.exit(0 if ok else 1)

