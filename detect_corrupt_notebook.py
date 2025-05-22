import os, json

for root, _, files in os.walk("."):
    for f in files:
        if f.endswith(".ipynb"):
            path = os.path.join(root, f)
            try:
                with open(path, 'r') as infile:
                    json.load(infile)
            except Exception as e:
                print(f"❌ {path} is invalid: {e}")
