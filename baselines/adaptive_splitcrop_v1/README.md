# Adaptive split-hand diffusion baseline v1

This directory freezes the adaptive crop run trained from commit `3b29933`.
Large checkpoints and generated datasets remain under ignored local `outputs/`
and `data/`; their paths, sizes, and SHA-256 checksums are recorded here.

The baseline is selected as a visual/system baseline. Its 183-record seen-eval
set is derived from training sequences, and its changed crop population prevents
aggregate loss from being treated as a controlled comparison with
`faithful_mano_v2`.

Verify the local checkpoint directly with:

```bash
sha256sum outputs/hand_restoration/hugg_aria_glove2hand_faithful_mano_adaptive_splitcrop_v1/controlnet_final.pt
```

Expected SHA-256:
`76a9a0243700388b20cffed5d0b8cf3f4bf5e32ebaa3a7a8a257149fd509e1e3`.
