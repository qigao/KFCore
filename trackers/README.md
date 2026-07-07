# trackers C

This directory contains a native C implementation of the core tracking
algorithms:

- `SORT`
- `ByteTrack`
- `C-BIoU`
- `OC-SORT`

The public API is in `include/trackers/tracker.h`. Tracker instances are
created with `sort_create`, `bytetrack_create`, `cbiou_create`, or
`ocsort_create`, updated with caller-owned detection/output arrays, and released
with the matching destroy function.

It uses:

- installed `KFCore::kfcore` from `KFCORE_ROOT` for Kalman predict/update
- local `vendor/miniblas` for tracker-side linear algebra helpers

Typical workflow:

```bash
cmake -S . -B build/c-trackers -G Ninja ^
  -DKFCORE_ROOT=C:/projects/cpp/external/pkgs/kfcore ^
  -DTINYTEST_DIR=C:/projects/cpp/external/pkgs/turbonet/include
cmake --build build/c-trackers
ctest --test-dir build/c-trackers --output-on-failure
```

To build the example executable:

```bash
cmake -S . -B build/c-trackers -G Ninja -DBUILD_EXAMPLES=ON ^
  -DKFCORE_ROOT=C:/projects/cpp/external/pkgs/kfcore ^
  -DTINYTEST_DIR=C:/projects/cpp/external/pkgs/turbonet/include
cmake --build build/c-trackers
build\c-trackers\bin\example.exe
```
