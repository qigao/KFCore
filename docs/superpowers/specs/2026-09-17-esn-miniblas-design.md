# ESN miniblas design

## Goal

Add a small, production-oriented Echo State Network (ESN) algorithm module to KFCore using the existing `KFCore::miniblas_internal` numerical backend.

## Scope

The first slice provides:

- float32 leaky-reservoir state updates;
- linear readout prediction;
- batch ridge-regression fitting for readout weights;
- caller-owned model/state/workspace memory;
- deterministic unit tests for state evolution, readout, fitting, and invalid input handling.

The first slice deliberately does not add random reservoir generation, spectral-radius estimation/scaling, sparse reservoirs, online RLS, deep/grouped ESN, serialization, or task-specific gesture integration.

## Architecture

`miniblas` remains an internal numerical primitive layer. A new top-level `esn` module owns ESN semantics and links privately to `KFCore::miniblas_internal`.

The public model is a non-owning view over column-major weight matrices:

- `input_weights`: reservoir x input;
- `reservoir_weights`: reservoir x reservoir;
- `reservoir_bias`: reservoir;
- `output_weights`: output x reservoir;
- `output_bias`: output.

State and scratch buffers are provided by the caller. The runtime performs no hidden allocation and stores no global state.

## Runtime contract

The reservoir update is

`x(t) = (1 - leak) * x(t-1) + leak * tanh(Win*u(t) + W*x(t-1) + b)`.

Prediction is

`y(t) = Wout*x(t) + bout`.

`leak` must be finite and in `(0, 1]`. Dimensions and required pointers must be valid. Invalid inputs fail immediately with a non-zero status.

## Ridge fitting

Given a column-major state matrix `X` with shape `reservoir x samples` and target matrix `Y` with shape `output x samples`, fit

`Wout = Y X^T (X X^T + lambda I)^-1`.

The implementation must not form an explicit inverse. It builds the regularized Gram matrix with `matmul`, factors it with `cholesky`, and solves on the right with `trisolveright`. `lambda` must be finite and strictly positive.

## Errors and ownership

The API returns `KFCORE_ESN_OK`, `KFCORE_ESN_INVALID_ARGUMENT`, or `KFCORE_ESN_NUMERICAL_FAILURE`. Numerical failure is reserved for factorization failure. No fallback solver is attempted.

All weights, state, targets, samples, and workspaces remain caller-owned. Fitting writes only the output-weight buffer and supplied workspaces.

## Testing

Tests use TinyTest and cover:

1. a two-neuron leaky update with an analytically known result;
2. a linear readout with known weights and bias;
3. one-dimensional ridge fitting that recovers a slope of approximately 2;
4. invalid leak/lambda/null-pointer rejection;
5. a singular/non-positive-definite fitting path returning numerical failure when applicable.

The module is built as `KFCore::esn` and its test is registered only when `BUILD_TESTS` is enabled.
