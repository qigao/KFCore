# Predicate tokenizer provider

KFCore keeps the exact CLIP-BPE tokenizer outside `KFCore::relation_core`.
The optional provider is built only when explicitly enabled:

```text
KFCORE_ENABLE_HF_TOKENIZER_PROVIDER=ON
KFCORE_TOKENIZERS_CPP_SOURCE_DIR=<pinned tokenizers-cpp checkout>
```

The provider target is:

```text
KFCore::relation_tokenizer_hf
```

It implements the abstract `PredicateTokenizer` interface used by
`PredicateTextEncoder`. When the provider is disabled, relation_core has no
Rust/tokenizers dependency.

## Reproducible dependency boundary

KFCore's exact CLIP parity workflow pins:

```text
mlc-ai/tokenizers-cpp
c586c52f93f7b060753bd2388eb96a105cb7374d

Rust
1.90.0
```

The provider consumes a local HuggingFace `tokenizer.json`; runtime does not
download tokenizer assets.

## Linux

Install the pinned Rust toolchain, clone tokenizers-cpp with submodules, then
configure KFCore with the normal Salts/SaltsUtils package discovery plus:

```sh
rustup toolchain install 1.90.0 --profile minimal
rustup default 1.90.0

cmake -S . -B build/tokenizer-linux \
  -DKFCORE_ENABLE_HF_TOKENIZER_PROVIDER=ON \
  -DKFCORE_TOKENIZERS_CPP_SOURCE_DIR=/src/tokenizers-cpp

cmake --build build/tokenizer-linux --target kfcore_relation_tokenizer_hf
```

The dedicated Linux parity CI additionally compares the C++ provider byte for
byte against Python HuggingFace `CLIPTokenizer` goldens.

## Windows

The pinned tokenizers-cpp CMake maps:

```text
x86_64 / AMD64  -> x86_64-pc-windows-msvc
ARM64 / AARCH64 -> aarch64-pc-windows-msvc
```

Install the matching Rust target before configuring:

```powershell
rustup toolchain install 1.90.0 --profile minimal
rustup default 1.90.0
rustup target add x86_64-pc-windows-msvc

cmake -S . -B build/tokenizer-win -G Ninja `
  -DKFCORE_ENABLE_HF_TOKENIZER_PROVIDER=ON `
  -DKFCORE_TOKENIZERS_CPP_SOURCE_DIR=C:/src/tokenizers-cpp

cmake --build build/tokenizer-win --target kfcore_relation_tokenizer_hf
```

KFCore rejects other Windows architectures when this provider is enabled,
rather than letting tokenizers-cpp silently select x86_64.

## Android

Use the Android NDK CMake toolchain. The pinned tokenizers-cpp mapping is:

| Android ABI | Rust target |
|---|---|
| `arm64-v8a` | `aarch64-linux-android` |
| `armeabi-v7a` | `armv7-linux-androideabi` |
| `x86_64` | `x86_64-linux-android` |
| `x86` | `i686-linux-android` |

Example for arm64:

```sh
rustup toolchain install 1.90.0 --profile minimal
rustup default 1.90.0
rustup target add aarch64-linux-android

cmake -S . -B build/tokenizer-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DKFCORE_ENABLE_HF_TOKENIZER_PROVIDER=ON \
  -DKFCORE_TOKENIZERS_CPP_SOURCE_DIR=/src/tokenizers-cpp

cmake --build build/tokenizer-android --target kfcore_relation_tokenizer_hf
```

KFCore rejects unsupported Android ABIs before entering the Rust build.

## Packaging boundary

The tokenizer provider is currently a source-build integration target rather
than an exported installed-package target. Applications that require live
predicate strings should build KFCore together with the pinned tokenizers-cpp
source. Precomputed `PredicateVocabulary` use does not require this provider.

This is deliberate: it keeps the default installed relation runtime free of a
Rust/tokenizers dependency while preserving an exact local tokenizer path for
applications that opt in.
