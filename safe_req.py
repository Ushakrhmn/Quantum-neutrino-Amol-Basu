import os
import ast
import json
import traceback
from pathlib import Path
import pkg_resources

def extract_top_level_modules(tree):
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split('.')[0])
    return modules

def extract_imports_from_py(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(path))
            return extract_top_level_modules(tree)
    except Exception:
        print(f"❌ Skipping invalid Python file: {path}")
        return set()

def extract_imports_from_ipynb(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
        code = '\n'.join([''.join(cell) for cell in code_cells])
        tree = ast.parse(code, filename=str(path))
        return extract_top_level_modules(tree)
    except Exception:
        print(f"❌ Skipping invalid notebook: {path}")
        return set()

def collect_imports(base_path):
    all_imports = set()
    for root, _, files in os.walk(base_path):
        for file in files:
            path = Path(root) / file
            if file.endswith('.py'):
                all_imports |= extract_imports_from_py(path)
            elif file.endswith('.ipynb'):
                all_imports |= extract_imports_from_ipynb(path)
    return sorted(all_imports)

def main(project_path='./', output='requirements.txt'):
    print(f"🔍 Scanning project: {project_path}")
    imports = collect_imports(project_path)

    # Get actual installable packages (names from pip)
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    # Match imports against installed packages (case-insensitive)
    cleaned = sorted({imp for imp in imports if imp.lower() in installed_packages})

    if not cleaned:
        print("⚠️ No valid packages found.")
        return

    with open(output, 'w') as f:
        for imp in cleaned:
            f.write(f"{imp}\n")

    print(f"✅ Written {len(cleaned)} clean imports to {output}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('project_path', nargs='?', default='.', help='Path to scan (default: current dir)')
    parser.add_argument('--output', '-o', default='requirements.txt', help='Output file (default: requirements.txt)')
    args = parser.parse_args()
    main(args.project_path, args.output)
