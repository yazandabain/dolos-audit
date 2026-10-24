# Third-party sources

The MIT license covers this project's original code. It does not change the licenses of
models, libraries, the sprint's report template, or upstream starter materials.

- The Dolos challenge and loading conventions come from Poseidon Research's
  [collusion_hackathon/redteaming](https://github.com/poseidon-research/collusion_hackathon/tree/d585a7b3be50366cc2bdf5789bf8add5890b2abd/redteaming),
  pinned at d585a7b3be50366cc2bdf5789bf8add5890b2abd. Its dependency configuration and lock
  are retained under runtime/ as reproducibility metadata. No license was listed at the
  starter repository root during inspection. Unmodified starter source used for private
  inspection is excluded from Git; no redistribution permission is inferred.
- Poseidon-Research/Dolos-1.5B and Qwen/Qwen2.5-1.5B-Instruct each publish an Apache 2.0
  LICENSE at the pinned revisions in configs/models.json. No model weights are included.
  Preserve their licenses and notices when downloading or redistributing models.
- The report template is provided by Apart Research for sprint submissions. Its provenance
  is recorded in references/official-sources.json. Its text is provided here for that purpose.
- PyTorch, Transformers, Hugging Face Hub, safetensors, Modal, pytest, ruff, uv, GitHub
  Actions, and the Actions runner retain their respective upstream licenses. Dependency
  pins and artifact hashes are recorded in the lockfiles. No third-party package code is
  relicensed by this project.
