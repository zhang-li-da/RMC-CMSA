# Original controller source

The `code/controller_src/` directory contains the four author-controlled controller modules used by RMC-CMSA:

- `rmc_cmsa_final_four_component.py`
- `modular_rmc_cmsa_v12.py`
- `modular_rmc_cmsa_v3.py`
- `base_algorithm.py`

These files are provided for source inspection and extension. They are not a standalone executable checkout: the import graph also requires the excluded `external_baselines.py`, CMMOP problem package, and the CMSA/RSC dependency. Those dependencies have different or unresolved redistribution terms and are therefore documented in `DEPENDENCIES.md` rather than copied here. This release makes no blanket license grant for third-party material; the author-controlled source remains subject to the authors' rights until a reuse license is selected.
