# Model Package V1

KFCore Model Package V1 is the on-disk contract used by `KFCore::runtime_core`
to select and verify model artifacts independently of the execution backend.

## Package entry point

`ModelPackage::load(path)` accepts either:

- a package directory containing `model.json`; or
- a JSON manifest file, whose parent directory becomes the package root.

The manifest must be non-empty and no larger than 1 MiB. Its schema identifier
must be exactly:

```text
kfcore.model/1
```

The package requires non-empty `id`, `version`, `model_type`, and a
non-empty `artifacts` list. `variant` is optional.

## Semantic contract identity

Typed models may bind backend-invariant output semantics through the optional
paired fields:

```text
semantic_contract
semantic_version
```

The fields must either both be absent or both be non-empty. The runtime package
layer preserves the identity but does not interpret model-specific semantics.
Typed model loaders validate the contract before execution.

For example, RTMW WholeBody133 packages use:

```json
{
  "semantic_contract": "pose.coco-wholebody-133",
  "semantic_version": "1"
}
```

A typed model may additionally pin a package-relative semantic sidecar:

```json
{
  "semantic_config": "pose.json",
  "semantic_config_sha256": "<lowercase sha256>"
}
```

These fields must be declared together and require a semantic contract identity.
The runtime verifies path containment and exposes
`verify_model_semantic_config()` for content-integrity checking, but it does
not interpret the sidecar. Typed model code owns the sidecar schema and
semantics.

`variant` is not a substitute for semantic identity. Backends must not infer a
semantic contract from tensor dimensions, provider names, artifact format, or
other execution details.

For `model_type="relation.relate-anything"`, the manifest must also set
`predicate_order_sha256` to the digest emitted by the fixed-vocabulary relation
ONNX exporter. The relation loader rejects a missing digest or a different
ordered list of predicate names before loading the backend. Existing fixed
relation manifests need this field when upgraded; other model types do not.

The digest follows the repository's ordered-name convention: SHA-256 over each
UTF-8 predicate name followed by a NUL byte, in order. Names must be non-empty
and cannot contain NUL bytes. Package creation must use the digest from the
metadata generated with the selected ONNX artifact and verify that metadata's
`onnx_sha256` equals the artifact's `sha256`.

## Artifact contract

Every artifact requires:

```text
id
format
path
sha256
backend
device
```

Artifact IDs are unique. Paths are package-relative, are canonicalized, and
must remain inside the package root. SHA-256 values are lowercase hexadecimal.

At load time, provenance references are validated structurally. Before an
artifact is executed, `verify_model_artifact()` hashes the artifact bytes and
requires the digest to match the manifest.

If `source_artifact` is present, it must name another artifact in the same
package, must not reference itself, and `source_sha256` must equal the source
artifact's declared digest.

## YOLO semantic flavor

For `model_type="yolo-detection"`, every artifact must explicitly declare one
of:

- `raw-yolo`;
- `raw-yolox`;
- `compact-nms`;
- `efficient-nms`.

The decoder contract is therefore package metadata, not something inferred
from the selected backend.

## TensorRT engine provenance

An artifact with `format="tensorrt-engine"` must:

- declare `backend="tensorrt"`;
- use a CUDA device constraint;
- derive from an ONNX artifact in the same package;
- carry matching `source_artifact` / `source_sha256`;
- record exact TensorRT `runtime_version` as
  `major.minor.patch.build`;
- record a supported target `platform`;
- record `compute_capability` as `major.minor`;
- record non-empty precision metadata;
- declare `hardware_compatibility` as either `exact-device` or
  `same-compute-capability`.

`exact-device` also requires `device_name`.

The resolver requires the engine's declared TensorRT version to equal the
loaded backend runtime version exactly. Compute capability must also match the
selected CUDA device exactly. `same-compute-capability` is accepted only for
TensorRT 10.9 or newer; older engines must use `exact-device`.

## Execution resolution

Applications provide an ordered or exact `ExecutionPolicy`. For each policy
entry, KFCore considers the matching loaded backend, device, and package
artifacts. Backend/device constraints and TensorRT runtime compatibility are
checked before the artifact is verified and loaded.

There is no implicit provider fallback beyond the order explicitly supplied by
the application.

## Trust boundary

A Model Package is metadata plus artifacts, not serialized runtime state.
Backend handles, execution contexts, device pointers, and other live runtime
objects are never persisted in the package.
