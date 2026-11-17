# Setup collision correction

Use RUN_BATCH_SAFE.py (replace the entire old batch cell), or the notebook ending _BATCH_SAFE.ipynb. Keep the existing matching dataset and T4x2/Internet settings. No ZIP re-upload or CUDA rebuild is needed.

The setup launcher waits for the existing setup owner without launching concurrent installation. It exclusively owns the real setup lock throughout the unchanged installer, and reuses a validated live server when all readiness evidence is present. The outer batch guard makes repeated starts a no-op. Do not delete lock files or repeatedly restart cells. Every software test still runs with the isolated status fixture.

In the existing interactive session, running the corrected full batch cell reuses already installed dependencies and can reuse completed readiness. For a separate committed version, select Save Version / Save & Run All; that separate session may need to restore its environment again. Source algorithms and experiment protocols are unchanged.

Restricted snapshot correction: latest SAFE code preserves executed_source_gpu1 and archives the subset as executed_source_restricted_v2. The partially restored session can be reused; no reset/re-upload is required.
