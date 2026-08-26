# PopSift upstream record

## Source

- Repository: https://github.com/alicevision/popsift.git
- Branch used for the snapshot: `develop`
- Commit: `36d704d39b4cc065839d84f3706b3fa88eff2518`
- Upstream project version: `0.10.1`
- License: Mozilla Public License 2.0; see `COPYING.md`

## Imported scope

KFCore imports the 63 files under upstream `src/popsift/` and
`cmake/sift_config.h.in`. The application, sample data, documentation,
standalone package export files, and upstream command-line test harness are
not part of this fork.

The baseline import was byte-for-byte copied from the commit above. Line-ending
differences may be introduced by the checkout's Git attributes; comparisons
therefore ignore a trailing carriage return.

## Verification

From the KFCore source root, with a checkout of the exact upstream commit at
`build/research/popsift-upstream`:

```powershell
git -C build/research/popsift-upstream rev-parse HEAD
git diff --no-index --ignore-cr-at-eol -- `
  build/research/popsift-upstream/src/popsift `
  sift/vendor/popsift/src/popsift
```

The first command must print the commit recorded above. Before KFCore patches
are applied, the second command must produce no content diff.

## KFCore modifications

- `src/popsift/common/sync_queue.h`: replaces the allocation-growing
  `std::queue` with a constructor-sized ring, explicit close/drain semantics,
  producer backpressure, and waiter wake-up. Covered by
  `test_popsift_internal`.
- `src/popsift/popsift.h` and `src/popsift/popsift.cu`: add configurable pending
  capacity, checked CUDA device selection, RAII job image storage, promise-based
  worker exceptions, per-job cleanup, and ordered close/drain/join. Covered by
  `test_popsift_internal` and the real-CUDA `test_popsift_adapter`.
- `CMakeLists.txt` in this directory is KFCore-owned build integration. It
  compiles only the imported library sources into a private static target; it
  is not an upstream file.

The algorithm kernels, configuration types, descriptor layout, and
`cmake/sift_config.h.in` remain unchanged from the recorded upstream commit.
