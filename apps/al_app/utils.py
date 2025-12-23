
"""Compatibility wrapper re-exporting helpers from the package.

Apps should import these via the package; this file keeps legacy imports working
while forwarding to `explainable_al.utils`.
"""

from explainable_al import utils as pkg_utils

get_ecfp_fingerprints = pkg_utils.get_ecfp_fingerprints
get_maccs_keys = pkg_utils.get_maccs_keys
get_chemberta_embeddings = pkg_utils.get_chemberta_embeddings
calculate_metrics = pkg_utils.calculate_metrics

__all__ = [
    'get_ecfp_fingerprints',
    'get_maccs_keys',
    'get_chemberta_embeddings',
    'calculate_metrics',
]

