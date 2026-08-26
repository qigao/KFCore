# PopSift upstream record

## Source

- Repository: https://github.com/alicevision/popsift.git
- Branch used for the snapshot: `develop`
- Commit: `36d704d39b4cc065839d84f3706b3fa88eff2518`
- Upstream project version: `0.10.1`
- License: Mozilla Public License 2.0; see `COPYING.md`
- KFCore maintained source: https://github.com/qigao/KFCore under
  `sift/vendor/popsift/`.
- KFCore source revision: `@KFCORE_SOURCE_REVISION_RESOLVED@`
- Corresponding modified source:
  @KFCORE_POPSIFT_SOURCE_URL@

The configured revision and URL above are installed with the executable. A
source-archive build without Git metadata must set `KFCORE_SOURCE_REVISION` to
its immutable release tag or commit; configuration fails when neither source is
available.

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
  worker exceptions, immutable configuration after first admission, per-job
  cleanup, and ordered close/drain/join.
- `src/popsift/scale_geometry.h`: extracts image scaling and automatic octave
  calculation into a pure operation so concurrent producers do not mutate
  worker configuration.
- `src/popsift/common/cuda_cleanup.h`, `src/popsift/s_image.cu`,
  `src/popsift/sift_octave.h`, `src/popsift/sift_octave.cu`, and
  `src/popsift/sift_pyramid.cu`: make destructor cleanup exhaustive and
  non-throwing, retain the first CUDA cleanup error for diagnostics, and add a
  test-only cleanup fault injector. These lifecycle changes are covered by
  `test_popsift_internal`; real concurrent CUDA extraction is covered by
  `test_popsift_adapter`.
- `src/popsift/common/test_hooks.h`: provides build-test-only worker-stage fault
  injection used to verify startup failure, accepted-job completion, image-pool
  lease return, continued service after per-job failures, and ordered join.
- `CMakeLists.txt` in this directory is KFCore-owned build integration. It
  compiles only the imported library sources into a private static target and
  generates installed provenance with the exact KFCore revision; it is not an
  upstream file.

The extraction kernels, configuration types, descriptor layout, and
`cmake/sift_config.h.in` remain unchanged from the recorded upstream commit.
Executable-form redistributors must keep the MPL notice and make these
MPL-covered source files, including the KFCore modifications above, available
as required by MPL-2.0.
