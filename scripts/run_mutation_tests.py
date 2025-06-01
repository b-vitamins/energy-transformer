#!/usr/bin/env python3
# ruff: noqa
"""Run mutation testing with intelligent test selection for organized test structure."""

import argparse
import subprocess
import sys
from pathlib import Path


def get_changed_files(base_branch: str = "main") -> list[str]:
    """Get list of changed Python files compared to base branch."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_branch}..HEAD"],
            capture_output=True,
            text=True,
            check=True
        )

        changed_files = []
        for file in result.stdout.strip().split('\n'):
            if file.startswith('energy_transformer/') and file.endswith('.py'):
                if not file.endswith('__init__.py'):
                    changed_files.append(file)

        return changed_files
    except subprocess.CalledProcessError:
        print("Error: Failed to get changed files")
        return []


def find_test_files_for_module(module_path: str) -> list[str]:
    """Find test files that likely test the given module in our organized structure."""
    path = Path(module_path)
    module_name = path.stem

    parts = path.parts

    test_locations: list[str] = []

    if len(parts) >= 2:
        category = parts[1]

        if category == 'layers':
            test_locations.extend([
                f"tests/unit/layers/test_{module_name}.py",
                f"tests/integration/**/test_*{module_name}*.py"
            ])
        elif category == 'models':
            if len(parts) >= 3:
                subcategory = parts[2]
                test_locations.extend([
                    f"tests/unit/models/{subcategory}/test_{module_name}.py",
                    f"tests/unit/models/test_{module_name}.py",
                    "tests/integration/test_model_building.py"
                ])
            else:
                test_locations.extend([
                    f"tests/unit/models/test_{module_name}.py",
                    "tests/integration/test_model_building.py"
                ])
        elif category == 'spec':
            test_locations.extend([
                f"tests/unit/spec/test_{module_name}.py",
                "tests/integration/test_spec_to_model.py",
                f"tests/functional/test_*{module_name}*.py"
            ])

        if any(keyword in module_name.lower() for keyword in ['eval', 'validate', 'parse']):
            test_locations.append("tests/security/")

        test_locations.append(f"tests/regression/**/test_*{module_name}*.py")

    existing_tests: list[str] = []
    for pattern in test_locations:
        if '*' in pattern:
            for test_file in Path().glob(pattern):
                if test_file.exists() and test_file.is_file():
                    existing_tests.append(str(test_file))
        elif Path(pattern).exists():
            existing_tests.append(pattern)

    seen: set[str] = set()
    unique_tests: list[str] = []
    for test in existing_tests:
        if test not in seen:
            seen.add(test)
            unique_tests.append(test)

    if not unique_tests and len(parts) >= 2:
        category = parts[1]
        category_path = f"tests/unit/{category}/"
        if Path(category_path).exists():
            unique_tests = [category_path]

    if not unique_tests:
        unique_tests = ["tests/unit/"]

    return unique_tests


def get_test_category_for_module(module_path: str) -> str:
    """Determine primary test category for a module."""
    path = Path(module_path)

    if 'security' in str(path) or 'validate' in module_path or 'parse' in module_path:
        return 'security'

    if len(path.parts) > 1 and path.parts[1] == 'spec':
        return 'unit,integration'

    if len(path.parts) > 1 and path.parts[1] in ['layers', 'models']:
        return 'unit'

    return 'unit,integration'


def run_mutation_testing_on_file(
    file_path: str,
    test_paths: list[str],
    timeout: int = 300,
    test_categories: str | None = None
) -> tuple[bool, str, dict[str, int]]:
    """Run mutation testing on a single file."""
    print(f"\n{'='*60}")
    print(f"Mutation testing: {file_path}")
    print(f"Using tests: {', '.join(test_paths)}")
    if test_categories:
        print(f"Test categories: {test_categories}")
    print(f"{'='*60}")

    if test_categories:
        marker_expr = ' or '.join(test_categories.split(','))
        test_cmd = f"python -m pytest -x -q --tb=no -m '{marker_expr}' {' '.join(test_paths)}"
    else:
        test_cmd = f"python -m pytest -x -q --tb=no {' '.join(test_paths)}"

    cmd = [
        "poetry", "run", "mutmut", "run",
        "--paths-to-mutate", file_path,
        "--runner", test_cmd,
        "--simple-number-mutations",
        "--no-backup"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        output = result.stdout + result.stderr
        print(output)

        stats = {
            'killed': 0,
            'survived': 0,
            'timeout': 0,
            'suspicious': 0
        }

        for line in output.split('\n'):
            for stat in stats:
                if stat in line:
                    try:
                        parts = line.split(stat)
                        if parts:
                            num_str = parts[0].strip().split()[-1]
                            stats[stat] = int(num_str)
                    except Exception:
                        pass

        total_mutants = sum(stats.values())
        if total_mutants == 0:
            return True, "No mutants generated", stats

        if stats['survived'] == 0:
            return True, "All mutants killed! ✅", stats
        survival_rate = (stats['survived'] / total_mutants) * 100
        return False, f"{stats['survived']} mutants survived ({survival_rate:.1f}%) ❌", stats

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout} seconds", {'timeout': 1}
    except Exception as e:  # noqa: BLE001
        return False, f"Error: {e!s}", {}


def generate_survival_report():
    """Generate report of surviving mutants with details."""
    try:
        result = subprocess.run(
            ["poetry", "run", "mutmut", "results"],
            capture_output=True,
            text=True,
            check=True
        )

        print("\n" + "="*60)
        print("SURVIVING MUTANTS REPORT")
        print("="*60)
        print(result.stdout)

        subprocess.run([
            "poetry", "run", "mutmut", "html"],
            capture_output=True,
            check=False
        )

        if Path("html/index.html").exists():
            print("\nDetailed HTML report generated at: html/index.html")

    except Exception as e:  # noqa: BLE001
        print(f"Could not generate report: {e}")


def suggest_test_improvements(file_path: str, stats: dict[str, int]):
    """Suggest specific test improvements based on mutation results."""
    if stats.get('survived', 0) == 0:
        return

    print(f"\n{'='*40}")
    print("TEST IMPROVEMENT SUGGESTIONS")
    print(f"{'='*40}")

    module_name = Path(file_path).stem

    suggestions = [
        "1. Check boundary conditions (>, >=, <, <=)",
        "2. Verify exact numeric constants and calculations",
        "3. Test all branches of conditional statements",
        "4. Ensure exception types and messages are verified",
        "5. Test with edge case inputs (0, negative, None, empty)"
    ]

    if 'attention' in module_name:
        suggestions.extend([
            "6. Test attention masking edge cases",
            "7. Verify scaling factors (1/sqrt(d))",
            "8. Test with single token inputs"
        ])
    elif 'layer_norm' in module_name:
        suggestions.extend([
            "6. Test epsilon boundary conditions",
            "7. Verify normalization with zero variance",
            "8. Test gradient computation"
        ])
    elif 'hopfield' in module_name:
        suggestions.extend([
            "6. Test energy function calculations",
            "7. Verify convergence conditions",
            "8. Test with degenerate patterns"
        ])

    print("\n".join(suggestions))
    print("\nRun 'mutmut show <id>' to see specific surviving mutants")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run mutation testing with intelligent test selection'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Specific files to test (default: changed files)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run mutation testing on all files (warning: very slow)'
    )
    parser.add_argument(
        '--category',
        choices=['unit', 'integration', 'functional', 'security', 'regression'],
        help='Run only tests from specific category'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per file in seconds (default: 300)'
    )
    parser.add_argument(
        '--base-branch',
        default='main',
        help='Base branch for comparison (default: main)'
    )
    parser.add_argument(
        '--suggest',
        action='store_true',
        help='Show test improvement suggestions for surviving mutants'
    )

    args = parser.parse_args()

    if args.all:
        files_to_test = [
            str(f) for f in Path("energy_transformer").rglob("*.py")
            if not f.name.startswith('__') and not f.name.endswith('__init__.py')
        ]
    elif args.files:
        files_to_test = args.files
    else:
        files_to_test = get_changed_files(args.base_branch)

    if not files_to_test:
        print("No files to test!")
        return 0

    print(f"Testing {len(files_to_test)} files:")
    for f in files_to_test:
        print(f"  - {f}")

    failed_files: list[tuple[str, str]] = []
    all_stats: dict[str, dict[str, int]] = {}

    for file_path in files_to_test:
        test_paths = find_test_files_for_module(file_path)

        test_categories = args.category
        if not test_categories:
            test_categories = get_test_category_for_module(file_path)

        success, message, stats = run_mutation_testing_on_file(
            file_path, test_paths, args.timeout, test_categories
        )

        all_stats[file_path] = stats

        if not success:
            failed_files.append((file_path, message))

            if args.suggest and stats:
                suggest_test_improvements(file_path, stats)

    print("\n" + "="*60)
    print("MUTATION TESTING SUMMARY")
    print("="*60)

    total_stats = {
        'killed': sum(s.get('killed', 0) for s in all_stats.values()),
        'survived': sum(s.get('survived', 0) for s in all_stats.values()),
        'timeout': sum(s.get('timeout', 0) for s in all_stats.values()),
        'suspicious': sum(s.get('suspicious', 0) for s in all_stats.values())
    }

    total = sum(total_stats.values())
    if total > 0:
        print("\nOverall Statistics:")
        print(f"  Total mutants: {total}")
        print(f"  Killed: {total_stats['killed']} ({total_stats['killed']/total*100:.1f}%)")
        print(f"  Survived: {total_stats['survived']} ({total_stats['survived']/total*100:.1f}%)")
        if total_stats['timeout']:
            print(f"  Timeout: {total_stats['timeout']}")
        if total_stats['suspicious']:
            print(f"  Suspicious: {total_stats['suspicious']}")

    if failed_files:
        print(f"\n❌ {len(failed_files)} files have surviving mutants:\n")
        for file_path, message in failed_files:
            print(f"  - {file_path}: {message}")

        generate_survival_report()
        return 1

    print("\n✅ All mutants killed! Your tests are effective.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
