#!/usr/bin/env python3
"""
Self‑contained Lean‑mathlib compilation harness for the Herald dataset.

• Creates/refreshes a project `TmpProj/`
• Reuses the same project to recompile many candidate theorems quickly
"""

from datasets import load_dataset
import subprocess, os, shutil, sys
from pathlib import Path
import textwrap
import re
#from git import Repo
#from lean_dojo import LeanGitRepo, trace
import pandas as pd

# ----------------- CONFIG -----------------
HF_DATASET          = "FrenzyMath/Herald_proofs"
SPLIT               = "train"
MAX_ROWS            = None          # None i.e., all rows, select 10 for debugging
TIMEOUT_SECS        = 300
PROJECT_NAME        = "TmpProjDir"   # also the Lean namespace
LEAN_VERSION        = "v4.11.0"
REPL_VERSION        = "adbbfcb9d4e61c12db96c45d227de92f21cc17dd" #https://github.com/leanprover-community/repl/tree/adbbfcb9d4e61c12db96c45d227de92f21cc17dd (for v.4.11.0)
# Set to True if you want a completely fresh project each run
FORCE_CLEAN_SETUP   = True
rows_writeCSV       = []
# ------------------------------------------

PROJECT_DIR  = Path(PROJECT_NAME)
SRC_DIR      = PROJECT_DIR / PROJECT_NAME
LEAN_SRC     = SRC_DIR / "Basic.lean"
LEAN_SRC_REL = LEAN_SRC.relative_to(PROJECT_DIR).as_posix()

# ---------- helpers ----------
def run_cmd(cmd, cwd="."): #, timeout=TIMEOUT_SECS
    """Run a shell command, return (ok:bool, combined_output:str)."""
    try:
        completed = subprocess.run(
            cmd, cwd=cwd, text=True,
            capture_output=True  #, timeout=timeout
        )
        return completed.returncode == 0, completed.stdout + completed.stderr
    except Exception as exc:
        return False, str(exc)

def bootstrap_project():
    """Create project (or refresh if FORCE_CLEAN_SETUP)."""
    if FORCE_CLEAN_SETUP and PROJECT_DIR.exists():
        print("**  Removing old project...")
        shutil.rmtree(PROJECT_DIR)

    if not PROJECT_DIR.exists():
        print("**  Initialising empty Lean project...")
        ok, out = run_cmd(
            ["lake", "new", PROJECT_NAME]
        )
        if not ok:
            sys.exit(f"X  lake new failed:\n{out}")

    # Content you want to write
    content = f"import Lake\nopen Lake DSL\n\npackage \"{PROJECT_NAME}\" where\n  -- add package configuration options here" +\
                f"\n\n@[default_target]\nlean_lib «{PROJECT_NAME}» where\n  -- add library configuration options here" +\
                f"\n\nrequire mathlib from git \"https://github.com/leanprover-community/mathlib4\" @\"{LEAN_VERSION}\"" +\
                f"\nrequire \"REPL\" from git \"https://github.com/leanprover-community/repl.git\" @ \"{REPL_VERSION}\""

    # Open the file in write mode and write the content
    with open(PROJECT_DIR / "lakefile.lean", "w") as f:
        f.write(content)

    # Ensure dependencies & cache
    for cmd in (["lake", "update"], ["lake", "exe", "cache", "get"]):
        ok, out = run_cmd(cmd, cwd=PROJECT_DIR)
        if not ok:
            sys.exit(f"X  {' '.join(cmd)} failed:\n{out}")

    with open(PROJECT_DIR / "lean-toolchain", "r") as f:
        toolchainVersion = f.read().strip()

    try:
        expectedToolchainVer = f"leanprover/lean4:{LEAN_VERSION}"
        assert toolchainVersion == expectedToolchainVer
        print (f"✓ Lean version verified to be {expectedToolchainVer}")
    except:
        sys.exit(f"✗ Unexpected version: {toolchainVersion}...should be {expectedToolchainVer}")

def write_basic_lean(row):
    """Overwrite TmpProj/Basic.lean with header + formal_proof."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    # Dedent the header and proof to avoid indentation issues
    header = (row["header"]).strip()
    proof = (row["formal_proof"]).strip()

    # Separate import lines, and everything else (including open and variable)
    import_lines = []
    other_header_lines = []
    for line in header.splitlines():
        if re.match(r'^\s*import\s+', line):
            import_lines.append(line.strip())
        else:
            other_header_lines.append(line)

    # Wrap everything in a namespace to avoid name clashes
    lean_src = "\n".join([
        *import_lines,
        "",
        "namespace myNameSpace",
        "",
        *other_header_lines,
        "",
        proof,
        "",
        "end myNameSpace"
    ])
    LEAN_SRC.write_text(lean_src)
    return lean_src

def check_compiles():
    """Run lake build; return (success, truncated_error_or_None)."""
    ok, out = run_cmd(["lake", "build"], cwd=PROJECT_DIR)
    return ok, (None if ok else "\n".join(out.splitlines()[:12]))

def add_csv_row(
    herald_id: str,
    lean_source: str,
    lean_compilation_status: str,
    lean_compilation_output: str,
    REPL_execution_status: str,
    REPL_output: str
):
    global rows_writeCSV
    row = {
        "Herald ID": herald_id,
        "LEAN Source": lean_source,
        "LEAN Compilation Status": lean_compilation_status,
        "LEAN Compilation Output": lean_compilation_output,
        "REPL Execution Status": REPL_execution_status,
        "REPL Output": REPL_output
    }
    rows_writeCSV.append(row)

# -------------- main --------------
def main():
    bootstrap_project()

    ds = load_dataset(HF_DATASET, split=SPLIT)
    if MAX_ROWS is not None:
        ds = ds.select(range(MAX_ROWS))

    # Counters
    passed_compile = 0
    passed_dag = 0
    # List to store row dictionaries

    for idx, row in enumerate(ds):
        ok_compile, ok_repl = False, False
        herald_id = str(row["id"])
        lean_src = write_basic_lean(row)  # get the Lean source string
        ok_compile, out_compile = check_compiles()
        status = "✓" if ok_compile else "✗"
        print("\n\n\n   " + "=" * 80)
        print(f"[{status}] row {idx}")

        # Print the Lean source always
        print("   " + "-" * 40)
        print("   >> Lean source:")
        print("   " + "-" * 40)
        print(textwrap.indent(lean_src.strip(), "   "))
        print("   " + "-" * 40)

        if not ok_compile:
            print("   ‣ error:", out_compile.replace("\n", "\n     "))
            ok_repl, out_repl = False, "Skipped as LEAN compilation failed"
        else:
            cmd = [
                "sh",
                "-c",
                f"echo '{{\"path\": \"{LEAN_SRC_REL}\", \"allTactics\": true}}' | lake exe repl"
            ]
            ok_repl, out_repl = run_cmd(cmd, cwd=PROJECT_DIR)
            if not ok_repl:
                print ("   >> REPL failed!")
            else:
                print ("   >> REPL ran successfully!")
                print("   " + "-" * 40)
                print(textwrap.indent(out_repl.strip(), "   "))
                print("   " + "-" * 40)
                # Save only successful DAG rows
                
        add_csv_row(herald_id, lean_src, str(ok_compile), out_compile, str(ok_repl), out_repl)

        passed_compile += (ok_compile == True)
        passed_dag += (ok_repl == True)

    total = len(ds)
    print(f"\n*** Compiled {passed_compile}/{total} successfully.")
    print(f"\n*** Computed DAG for {passed_dag}/{total} successfully.")

    # Construct DataFrame and save as CSV
    df = pd.DataFrame(rows_writeCSV)

    csv_path = "herald_lean_dag.csv"
    df.to_csv(csv_path, encoding="utf-8", index=False)
    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()
