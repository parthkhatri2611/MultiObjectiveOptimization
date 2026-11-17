# Recorded run and launcher requirements

The native source is preserved as executed, including the need for an external import-order launcher. The completed Kaggle campaign used RECOVER_OPTIMIZER_IMPORT_ORDER.py, which creates a controller wrapper and pre-imports syftr.hierarchical_tpe before replacing Optuna's sampler factory. Without that preload an evolutionary arm may fail with `TypeError: function() argument code must be code, not str`.

The SAFE batch notebook restores the environment and normally starts its controller. If the historical import-order error occurs, first confirm that the controller has exited. Then run the included recovery helper as a notebook cell. It preserves completed trials, quarantines only empty failed initialization folders and launches the corrected controller. Never run two controllers or overwrite a nonempty study. The helper contains Kaggle-specific paths and is intended for the declared seed-11 campaign.

Source identity checks and captured manifests are retained. Do not disable them to make a mismatched environment run. The archive provided in runtime_assets is the original restoration bundle, not a newly validated replacement. The saved study was completed using the documented recovery; we have CPU-audited the final outputs, not rerun six hours of GPU work for this release.
