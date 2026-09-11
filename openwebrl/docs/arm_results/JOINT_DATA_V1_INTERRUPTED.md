# Interrupted first joint-data preparation

The initial builder under runtime `arm-reproduction/joint-data-v1-20260910`
hit its 240 CPU-second safety cap while processing Piotr candidate draws.
It is incomplete and must not be used for training. Its verified context
tokenizations can be reused by exact context/image identity. The next build
rechecks image hashes and saved C2 prefix hashes, and performs a separate
end-to-end verification of retained response tokens and action masks.

No model weights or GPU work were involved. The corrected builder avoids
tokenizing every rejected alternative and skips higher-hash draws once a
usable lower-hash pair is available. Detailed exclusion counts consequently
describe processed draws, while all draw identities remain inventoried.
