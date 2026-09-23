"""Smoke test the public notebook without requiring a registered Jupyter kernel."""
import json
from pathlib import Path


def test_quickstart_example(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/quickstart_library_ranking.ipynb").read_text(encoding="utf-8"))
    code = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    code = code.replace("OUTPUT_DIR = 'notebook_outputs'", f"OUTPUT_DIR = {str(tmp_path)!r}")
    monkeypatch.chdir(root)
    exec(compile(code, "quickstart_library_ranking.ipynb", "exec"), {})
    import pandas as pd

    validation = pd.read_csv(tmp_path / "validation_predictions.csv")
    ranked = pd.read_csv(tmp_path / "full_ranked_library.csv")
    top = pd.read_csv(tmp_path / "top_candidates.csv")
    assert len(validation) == len(ranked) > 0
    assert len(top) == 20
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["predicted_activity"].is_monotonic_decreasing


def test_quickstart_user_library(monkeypatch, tmp_path):
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/quickstart_library_ranking.ipynb").read_text(encoding="utf-8"))
    code = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    library_path = tmp_path / "library.csv"
    pd.DataFrame({"SMILES": [pd.read_csv(root / "data/TYK2.csv").SMILES.iloc[0], "CCO", "CCN"]}).to_csv(library_path, index=False)
    code = code.replace("LIBRARY_CSV = None", f"LIBRARY_CSV = {str(library_path)!r}")
    code = code.replace("OUTPUT_DIR = 'notebook_outputs'", f"OUTPUT_DIR = {str(tmp_path)!r}")
    monkeypatch.chdir(root)
    exec(compile(code, "quickstart_library_ranking.ipynb", "exec"), {})
    ranked = pd.read_csv(tmp_path / "full_ranked_library.csv")
    assert set(ranked.SMILES) == {"CCO", "CCN"}
    assert len(pd.read_csv(tmp_path / "top_candidates.csv")) == 2
