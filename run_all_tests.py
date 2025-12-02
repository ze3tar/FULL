import os
import sys


def run(command: str) -> bool:
    print(f"\n➡️ Running: {command}")
    result = os.system(command)
    if result == 0:
        print("✅ PASS")
        return True
    print("❌ FAIL")
    return False


def main() -> None:
    tests = [
        "python tests/test_dependencies.py",
        "python tests/test_baseline.py",
        "python tests/test_pso.py",
        "python tests/test_rl_env.py",
        "python tests/test_lstm.py",
        "python tests/test_comparison.py",
        "python tests/test_hardware.py",
    ]

    results = {command: run(command) for command in tests}

    passed = sum(results.values())
    total = len(results)
    print("\n==============================")
    print(f"Results: {passed}/{total} tests passed")
    status = "SUCCESS" if passed == total else "FAILED"
    print(f"Status: {status}")
    print("==============================")

    if passed != total:
        sys.exit(1)


if __name__ == "main__":  # pragma: no cover
    main()

if __name__ == "__main__":
    main()
