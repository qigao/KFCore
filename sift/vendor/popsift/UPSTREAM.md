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

The import commit contains no source modifications. Later commits update this
section with every locally modified MPL-covered file, the behavioral reason,
and the corresponding KFCore tests.
