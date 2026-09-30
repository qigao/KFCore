# Runtime backend plugin ABI

KFCore separates typed model APIs from execution providers with a C ABI defined
in `runtime/abi/include/kfcore/runtime/abi/backend_v1.h`.

## Loading boundary

Applications load an explicit plugin path or ask KFCore to enumerate one
explicitly controlled directory. Directory loading accepts only the platform's
unversioned KFCore backend library naming convention and processes candidates
in sorted order.

The runtime does not scan system library paths for providers.

## ABI negotiation

The exported entry point is:

```text
kfcore_backend_query_v1
```

The host requests ABI major 1 and negotiates downward from the newest supported
minor version. ABI structures use `struct_size` and append-only fields so a
newer host can validate the portion implemented by an older compatible plugin.

The current V1 table includes opaque backend/model/context handles, device
enumeration, artifact probing/loading, tensor metadata, context creation,
execution, diagnostics, and bounded dynamic output.

## Ownership

Handles are opaque. The plugin owns backend/model/context implementation state;
the host destroys them only through the function pointers returned in the ABI
table.

Tensor views are non-owning. Memory location is explicit:

- host;
- pinned host;
- device.

Device identity is carried separately. No raw runtime handle or device pointer
is serialized into a Model Package.

## Dynamic output

ABI v1.2 added bounded dynamic host output. The caller supplies storage
capacity; the backend reports the actual shape and byte count. The backend does
not return an unbounded allocation to the caller.

## Errors

Backend calls return `kf_status_v1`. KFCore converts backend failures into
runtime errors and asks the plugin to format its last diagnostic when
available. ABI mismatch is reported separately from backend execution failure.

## Provider-neutral rule

Model semantics are defined above this layer. Backend selection must not change
the meaning of a typed model output. Backend-specific derived artifacts belong
in Model Package metadata and are selected through explicit execution policy.
