# Face Swap CPU/GPU Performance Recording Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the CPU preprocessing, CUDA preprocessing, and complete TensorRT GPU pipeline costs reproducible and record the measured hardware/resource context.

**Architecture:** Retain the existing CPU/OpenCV versus CUDA ImageProcessor benchmark as the preprocessing comparison. Add an opt-in real-model pipeline benchmark that excludes engine loading, measures the compatibility path, and prints profiled stage P50/P95. Collect CPU/GPU device telemetry externally so the inference library does not acquire an NVML or platform-monitoring dependency.

**Tech Stack:** C++17, TensorRT 11.2, CUDA 12.8, OpenCV Lite, TurboUtils TinyTest, Windows process counters, `nvidia-smi`.

### Task 1: Add the real-model pipeline benchmark

- Add an opt-in `benchmark_face_swap_pipeline` target beside the existing integration test.
- Reuse the validated test assets and the public `swap_profiled()` report.
- Warm engines before measurement; report unprofiled total latency separately from stage P50/P95.
- Validate output shape/type and preserve optional GFPGAN separation.

### Task 2: Collect both performance routes

- Run `benchmark_face_preprocess` for CPU/OpenCV and CUDA ImageProcessor operations.
- Run `benchmark_face_swap_pipeline` for the complete TensorRT GPU route.
- Sample process CPU time/working set and GPU utilization/VRAM/power during a sustained benchmark run.

### Task 3: Record reproducible conclusions

- Record hardware, sample count, measured values, byte/pixel work units, and commands.
- Distinguish facts and calculations from inferences.
- State explicitly that GPU utilization is not FLOPS and that no complete CPU inference backend exists.
