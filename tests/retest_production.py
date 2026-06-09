"""
Full Production Retest — Combined Runner
Executes all 3 phases in order:
  Phase 1: 50-query Production Benchmark (with all fixes applied)
  Phase 2: 14-Agent Research Paper + Implementation Guide
  Phase 3: Generate production1.md final report

Run with: python tests/retest_production.py
"""
import sys
import os
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

def banner(msg):
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)

def run(script_name):
    script = os.path.join(ROOT, "tests", script_name)
    banner(f"RUNNING: {script_name}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([PYTHON, script], env=env, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} exited with code {result.returncode}")
    else:
        print(f"\n[OK] {script_name} completed successfully.")
    return result.returncode

if __name__ == "__main__":
    start = time.time()

    banner("PRODUCTION RETEST — ALL PHASES")
    print("  Phase 1: 50-Query Benchmark (fixed pipeline_failure, hallucination, navigator)")
    print("  Phase 2: 14-Agent Research Paper + Implementation Guide")
    print("  Phase 3: Generate production1.md final report")

    # Phase 1
    rc1 = run("run_production50.py")

    # Phase 2
    rc2 = run("generate_paper.py")

    # Phase 3
    rc3 = run("generate_production1_report.py")

    elapsed = time.time() - start
    banner(f"RETEST COMPLETE — Total time: {elapsed/60:.1f} min")
    print(f"  Phase 1 (Benchmark)  : {'OK' if rc1 == 0 else 'ERROR'}")
    print(f"  Phase 2 (Paper)      : {'OK' if rc2 == 0 else 'ERROR'}")
    print(f"  Phase 3 (Report)     : {'OK' if rc3 == 0 else 'ERROR'}")
    print()
    print("  Output files:")
    print("    outputs/production50_results.json")
    print("    outputs/novel_paper.md")
    print("    outputs/novel_implementation_guide.md")
    print("    production1.md  <-- FINAL REPORT")
    sys.exit(max(rc1, rc2, rc3))
