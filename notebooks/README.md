# Start here: rank your own compound library

Open [`quickstart_library_ranking.ipynb`](quickstart_library_ranking.ipynb) after installing the dependencies in the repository's `requirements-notebook.txt`.

The notebook accepts two CSVs:

- A measured table with one SMILES column and one numeric activity column.
- An optional virtual library with a SMILES column. Without one, the notebook uses a held-out part of the measured example data as a demonstration pool.

Edit the **Configuration** cell, then run all cells. It writes `notebook_outputs/validation_predictions.csv`, `notebook_outputs/full_ranked_library.csv`, and `notebook_outputs/top_candidates.csv`. The output directory is ignored by Git.

The notebook uses a Morgan fingerprint and random forest so a fresh clone can run on a laptop. The validation split keeps molecules with the same non-stereo connectivity together. The forest's tree-to-tree spread is an exploratory uncertainty proxy, not a calibrated error estimate. The final ranking is by predicted activity alone; check model performance and chemistry before ordering compounds.

The 14-model TYK2 research comparison depends on large experimental data and model artifacts that are not part of a fresh GitHub clone. Its saved results are separate from this reusable notebook.

For a supervisor's audit of those results, use [the TYK2 review notebook](review_tyk2_results.ipynb) with its [artifact manifest and guide](REVIEW_TYK2.md).
