# Vision Test Layout

The test tree is organized by gate purpose rather than implementation
language:

- `pr/`: the three stable PR gates declared in `../test.yaml`;
- `golden/`: hardware-backed numerical golden tests, run when OpenCL or image
  preprocessing changes and before release;
- `unit/`: CTest unit and contract coverage grouped by subsystem;
- `benchmarks/`: benchmark binaries and their metric unit test;
- `scheduled/`: scheduled performance collection entry points;
- `cmake/`: build and embedded-source checks used by CTest;
- `support/`: helpers shared by multiple test categories;
- `data/`: small checked-in fixtures used by gate tests.

`tests/output/` is created by test runners only when an artifact is produced.
It must not be committed or kept as a source-tree placeholder.

The default PR gate remains:

```bash
# Run from the SDK root.
scripts/test/robot-test run components/model_zoo/vision --scope pr
```

For OpenCL or preprocessing changes, run the manual golden gate in addition
to the PR gate. The same gate is mandatory in the release scope:

```bash
# Run from the SDK root.
scripts/test/robot-test run components/model_zoo/vision \
  --scope manual
```

From the standalone vision repository, the equivalent golden entry point is
`bash tests/golden/run_opencl_preprocess_golden.sh`.
