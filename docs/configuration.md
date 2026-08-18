# Configuration

Root catalogs contain only method-neutral dataset, detector, and runtime
metadata. A concrete method owns its defaults and experiment presets below its
own package. The selected method manifest is discovered from
`methods/*/manifest.py`; no second list of method names is maintained.

Resolution order is method defaults, dataset catalog, detector catalog,
runtime catalog, then explicit command-line overrides. The resolved mapping is
serialized deterministically and assigned a SHA256 fingerprint.

All runtime paths are repository-relative. Local absolute dataset locations
appear only as workstation junction targets documented in `data/README.md`.
