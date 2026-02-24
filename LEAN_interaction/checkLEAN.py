#!/usr/bin/env python3
"""
Self‑contained Lean‑mathlib compilation harness for the Herald dataset.

• Creates/refreshes a project `TmpProj/`
• Reuses the same project to recompile many candidate theorems quickly
"""

import subprocess, os, shutil, sys
from pathlib import Path
import textwrap
import re
import uuid
#from git import Repo
#from lean_dojo import LeanGitRepo, trace
import pandas as pd
import json

# ----------------- CONFIG -----------------
TIMEOUT_SECS        = 300
PROJECT_NAME        = "TmpProjDir"   # also the Lean namespace
LEAN_VERSION        = "v4.15.0"
REPL_VERSION        = "21966799da3691a0912b5a15193585bd2dd7165d" #https://github.com/leanprover-community/repl/tree/21966799da3691a0912b5a15193585bd2dd7165d (for v.4.15.0)
# Set to True if you want a completely fresh project each run
FORCE_CLEAN_SETUP   = False
rows_writeCSV       = []
# ------------------------------------------

# Base tmp folder
TMP_BASE = Path("tmpFolder")
# Unique run folder
if not FORCE_CLEAN_SETUP:
    RUN_UUID = "8d9a729d-dec6-452d-bb90-d63be139ee52"
else:
    RUN_UUID = str(uuid.uuid4())
RUN_DIR  = TMP_BASE / RUN_UUID

PROJECT_DIR  = RUN_DIR / PROJECT_NAME
SRC_DIR      = PROJECT_DIR / PROJECT_NAME
LEAN_SRC     = SRC_DIR / "Basic.lean"

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
    if FORCE_CLEAN_SETUP and RUN_DIR.exists():
        print("**  Removing old project...")
        shutil.rmtree(RUN_DIR)

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not PROJECT_DIR.exists():
        print("** Initialising empty Lean project in", PROJECT_DIR)
        ok, out = run_cmd(
            ["lake", "new", PROJECT_NAME],
            cwd=RUN_DIR
        )
        if not ok:
            sys.exit(f"X  lake new failed:\n{out}")

    # Content you want to write
    content = f"""
import Lake
open Lake DSL

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "{LEAN_VERSION}"

require REPL from git
  "https://github.com/leanprover-community/repl.git" @ "{REPL_VERSION}"

package «{PROJECT_NAME}» where
-- add package configuration options here

@[default_target]
lean_lib «{PROJECT_NAME}» where
-- add library configuration options here"""

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
    return RUN_DIR, PROJECT_DIR, LEAN_SRC #the unique run directory, the project directory, and the lean file path

def write_basic_lean(header, proof, leanFile):
    """Overwrite TmpProj/Basic.lean with header + formal_proof."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    # Separate import lines, and everything else (including open and variable)
    import_lines = []
    other_header_lines = []
    for line in header.splitlines():
        if re.match(r'^\s*import\s+', line):
            import_lines.append(line.strip())
        else:
            other_header_lines.append(line)

    # Wrap everything in a namespace to avoid name clashes
    lean_code = "\n".join([
        *import_lines,
        *other_header_lines,
        proof
    ])
    leanFile.write_text(lean_code)
    return lean_code

def check_compiles(project_dir):
    """Run lake build; return (success, truncated_error_or_None)."""
    ok, out = run_cmd(["lake", "build"], cwd=project_dir)
    return ok, (None if ok else "\n".join(out.splitlines()[:12]))

def check_repl(lean_file_path, project_dir):
    """Run REPL"""
    lean_file_path_relative = lean_file_path.relative_to(project_dir).as_posix()
    cmd = [
        "sh",
        "-c",
        f"echo '{{\"path\": \"{lean_file_path_relative}\", \"allTactics\": true}}' | lake exe repl"
    ]
    ok_repl, out_repl = run_cmd(cmd, cwd=project_dir)

    # Find the first { ... } JSON block in the output
    json_match = re.search(r"\{.*\}", out_repl, re.DOTALL)
    if not json_match:
        return False, "[No JSON found in REPL output]" + out_repl

    json_str = json_match.group()
    try:
        repl_json = json.loads(json_str)
        messages = repl_json.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        ok_repl = len(errors) == 0
        return ok_repl, "[JSON parsed correctly]" + str(repl_json)
    except json.JSONDecodeError:
        return False, "[Failed to parse JSON from REPL output]" + out_repl

# -------------- main --------------
def main():
    run_dir, project_dir, lean_file_path = bootstrap_project()

    for i in range(0, 300001, 1000):
        fname = f"Basic_{i:06d}.lean"
        fpath = SRC_DIR / fname
        fpath.touch(exist_ok=True)   # create empty file (does nothing if it exists)
        print(f"Created: {fpath}")

    '''
    for idx, row in enumerate(ds):
        ok_compile, ok_repl = False, False
        herald_id = str(row["id"])
        # Dedent the header and proof to avoid indentation issues
        proof_header = (row["header"]).strip()
        proof_body = (row["formal_proof"]).strip()
        lean_code = write_basic_lean(proof_header, proof_body, lean_file_path)
        ok_compile, out_compile = check_compiles(project_dir)
        status = "✓" if ok_compile else "✗"
        print("\n\n\n   " + "=" * 80)
        print(f"[{status}] row {idx}")

        # Print the Lean source always
        print("   " + "-" * 40)
        print("   >> Lean source:")
        print("   " + "-" * 40)
        print(textwrap.indent(lean_code.strip(), "   "))
        print("   " + "-" * 40)

        if not ok_compile:
            print("   ‣ error:", out_compile.replace("\n", "\n     "))
            ok_repl, out_repl = False, "Skipped as LEAN compilation failed"
        else:
            ok_repl, out_repl = check_repl(lean_file_path, project_dir)
            if not ok_repl:
                print ("   >> REPL failed!")
            else:
                print ("   >> REPL ran successfully!")
                print("   " + "-" * 40)
                print(textwrap.indent(out_repl.strip(), "   "))
                print("   " + "-" * 40)
                # Save only successful DAG rows
                
        add_csv_row(herald_id, lean_code, str(ok_compile), out_compile, str(ok_repl), out_repl)

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
    '''

if __name__ == "__main__":
    main()
