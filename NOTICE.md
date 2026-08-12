# Third-Party and Upstream Attribution

## Gigax

MATE-NPC-Security builds on and modifies components of the Gigax
LLM-powered NPC runtime:

https://github.com/GigaxGames/gigax

The MATE development branch was based on upstream commit:

c3c209de6290f9f3f2f9395625bb2e8094122531

The upstream package metadata identifies Gigax as MIT-licensed.

The original Gigax runtime provides the foundation for NPC prompting,
scene representation, structured action generation, and related runtime
components.

MATE-NPC-Security adds and/or substantially modifies security-related
components including:

- Multi-RAAC runtime integration
- Action-Tiered Policy (ATP)
- schema-constrained security-aware action control
- Threat-Evidence Gate (TEG)
- extensible critical-action authorization
- Original Multi-RAAC detector construction
- Low-FPR Sparse Multi-RAAC
- shared-forward sparse inference
- detector calibration and evaluation utilities
- public security-focused runtime configuration and tests

The retained `gigax` Python namespace is used for compatibility with the
underlying NPC runtime.

## Models and Datasets

Model weights and third-party datasets are not redistributed as part of
the source repository. Users are responsible for complying with the
licenses and terms of the corresponding model and dataset providers.
