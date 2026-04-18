import subprocess
import sys


def run_step(name, cmd):
    print(f"[RUN] {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with code {result.returncode}")


def main():
    py = sys.executable
    run_step("live_signal", [py, "live_signal.py"])
    run_step("execute_signal", [py, "execute_signal.py"])


if __name__ == "__main__":
    main()
