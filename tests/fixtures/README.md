# Portable training reference

`train_before_tensor_cleanup.py` is the exact corrected trainer before the
lifetime-only logits cleanup, archived from commit
`4cc7bff729537e17b471bef36e3a40cd2e11e5d3`. It is a tiny-model test reference,
not a training entry point for project runs.

SHA-256: `208aeccd7d54ebc0af0b94843c2ad2d92add078f3aff03a56b34f483d1461774`.

Keeping the source in the test tree makes paired loss/weight-equivalence tests
work in shallow clones, squash merges and exported source snapshots. No
historical Git object is required by the reference test.

`train_main_lifetime_cleanup.py` is `scripts/train.py` as on main at commit
`2d0a121` (the frozen reference plus only the lifetime cleanup). The reuse-gate
test runs the gate against it instead of the working-tree trainer, so opt-in
trainer features do not break that historical test; a deliberately changed
copy must still be rejected.

SHA-256: `f744097c9406970e9a095f83f13026105e172f9608dbf5ef1778628722e83037`.
