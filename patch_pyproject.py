import tomli

with open("pyproject.toml", "rb") as f:
    data = tomli.load(f)

if "exclude" not in data["tool"]["ruff"]:
    data["tool"]["ruff"]["exclude"] = []
if "training" not in data["tool"]["ruff"]["exclude"]:
    data["tool"]["ruff"]["exclude"].append("training")
if "exclude" not in data["tool"]["ruff"]["lint"]:
    data["tool"]["ruff"]["lint"]["exclude"] = []
if "training" not in data["tool"]["ruff"]["lint"]["exclude"]:
    data["tool"]["ruff"]["lint"]["exclude"].append("training")

import tomli_w

with open("pyproject.toml", "wb") as f:
    tomli_w.dump(data, f)
