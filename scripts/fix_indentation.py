#!/usr/bin/env python3
"""
Fix indentation errors in Python files that use 1-space instead of 4-space indentation.

This script fixes the specific indentation pattern where class bodies are indented
with 1 space per level instead of the standard 4 spaces.

Usage:
    python3 scripts/fix_indentation.py
"""

import os
import re
from pathlib import Path


def fix_file_indentation(filepath):
    """
    Fix indentation by using autopep8 with aggressive indent fixing.

    Args:
        filepath: Path to Python file to fix

    Returns:
        True if successful, False otherwise
    """
    print(f"Fixing {filepath}...")

    try:
        # Try using autopep8 with aggressive mode
        import subprocess
        result = subprocess.run(
            ['autopep8', '--in-place', '--aggressive', '--aggressive', filepath],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Verify it compiles
            compile_result = subprocess.run(
                ['python3', '-m', 'py_compile', filepath],
                capture_output=True,
                text=True
            )

            if compile_result.returncode == 0:
                print(f"  ✓ {filepath} fixed and verified")
                return True
            else:
                print(f"  ✗ {filepath} still has errors after autopep8:")
                print(f"    {compile_result.stderr}")
                return False
        else:
            print(f"  ✗ autopep8 failed: {result.stderr}")
            return False

    except ImportError:
        print(f"  ! autopep8 not installed, trying manual fix...")
        return manual_fix(filepath)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def manual_fix(filepath):
    """
    Manually fix indentation using regex patterns.

    Note: This is a fallback and may not work for all cases.
    For best results, install autopep8:
        pip install autopep8
    """
    print(f"  Attempting manual fix for {filepath}...")

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Backup original
        backup_path = f"{filepath}.backup"
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"  Created backup: {backup_path}")

        # This is a simplified fix - replace each occurrence of
        # newline + N spaces with newline + 4*N spaces
        lines = content.split('\n')
        fixed_lines = []

        in_class = False
        for line in lines:
            if re.match(r'^class \w+', line):
                in_class = True
                fixed_lines.append(line)
            elif in_class and line and not line[0].isspace():
                in_class = False
                fixed_lines.append(line)
            elif in_class and line.startswith(' ') and not line.startswith('    '):
                # Line is in class and has wrong indentation
                spaces = len(line) - len(line.lstrip(' '))
                if 0 < spaces < 4:
                    # Fix it (this is simplistic and may not handle all cases)
                    fixed_lines.append('    ' * spaces + line.lstrip(' '))
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        with open(filepath, 'w') as f:
            f.write('\n'.join(fixed_lines))

        print(f"  Manual fix applied to {filepath}")
        print(f"  Please verify the file manually!")
        return True

    except Exception as e:
        print(f"  ✗ Manual fix failed: {e}")
        return False


def main():
    """Main function."""
    print("=" * 70)
    print("Python Indentation Fixer")
    print("=" * 70)
    print()

    # Files that need fixing (from error analysis)
    files_to_fix = [
        'src/observability/metrics_dashboard.py',
        'src/deployment/setup.py',
        'src/rag/query_profiler.py',
        'src/rag/ast_chunker.py',
    ]

    # Check if autopep8 is available
    try:
        import subprocess
        result = subprocess.run(['autopep8', '--version'], capture_output=True)
        if result.returncode == 0:
            print("✓ autopep8 is installed")
        else:
            print("! autopep8 not found, will use manual fixing")
            print("  For best results, install: pip install autopep8")
    except:
        print("! autopep8 not found, will use manual fixing")
        print("  For best results, install: pip install autopep8")

    print()

    # Fix each file
    success_count = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file_indentation(filepath):
                success_count += 1
        else:
            print(f"  ! File not found: {filepath}")

    print()
    print("=" * 70)
    print(f"Fixed {success_count}/{len(files_to_fix)} files")
    print("=" * 70)

    if success_count < len(files_to_fix):
        print()
        print("Some files could not be fixed automatically.")
        print("You may need to fix them manually in your IDE.")
        print()
        print("Alternative: Use a Python IDE with auto-formatting:")
        print("  - PyCharm: Code → Reformat Code")
        print("  - VSCode: Shift+Alt+F (with Python extension)")
        print("  - vim: gg=G (with proper Python indent settings)")


if __name__ == '__main__':
    main()
