# Retained License Texts

These files are byte-for-byte copies of the license texts at the pinned
upstream snapshots recorded in `../PROVENANCE.md`.

| File | Upstream | Snapshot | SHA-256 |
| --- | --- | --- | --- |
| `FAISS-LICENSE` | `facebookresearch/faiss` | `0ca9df4792b173d573044ee14ca0704780176e82` | `52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85` |
| `FAISS-MLX-LICENSE` | `MLXPorts/faiss-mlx` | `d092af559375144fc719cd88a10e414f92c625fa` | `b8d7376c7b21f8f4895af89772e9de0558bb99515820ceb21ba4be4e96efffcc` |
| `CUVS-LICENSE` | `rapidsai/cuvs` | `ad9e2d2a617c8d51e3eebc920e5a60ad8dc59bcd` | `756005f963846334943e8bfc08ef98cd254257d8467ac7a7ffd42a1be262f442` |

`tools/check_license_headers.sh` verifies these digests before release. Do not
normalize whitespace or replace a retained text without updating its pinned
snapshot, provenance entry, and expected digest together.
