#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esn_group.h"
#include "esn_group_deep.h"
#ifdef KFCORE_ESN_SEQUENCE_EMBEDDED
#define TINYTEST_NO_MAIN
#endif
#include "tinytest.h"

enum {
    TOTAL_NEURONS = 32, MAX_BLOCKS = 4, MAX_GROUPS = 2,
    WASHOUT = 100, TRAIN = 800, VALIDATION = 300, TEST = 400,
    SAMPLES = WASHOUT + TRAIN + VALIDATION + TEST,
    POWER_ITERATIONS = 200, MG_DELAY = 17, MG_BURN_IN = 1000, MEMORY_DELAY = 5
};
typedef enum { SHALLOW, GROUPED, DEEP, GROUPED_DEEP, ARCH_COUNT } Architecture;
typedef enum { OSCILLATORS, MACKEY_GLASS, DELAYED_INPUT, TASK_COUNT } Task;
static const char* const architecture_names[] = { "ESN", "gESN", "dESN", "gdESN" };
static const char* const task_names[] = { "oscillators", "mackey-glass", "delay-5" };
static const float ridge_candidates[] = { 0.01f, 0.1f, 1.0f };
static const float input_scale = 0.5f;
static const float spectral_radius = 0.8f;
static const float leak_rate = 0.8f;
static const float density = 0.3f;
#ifndef KFCORE_ESN_QUALITY
static const double regression_nrmse_limit = 0.15;
static const double regression_baseline_ratio = 0.8;
#endif

/* Test-only bounded storage: no model or dataset allocation in the step loop. */
typedef struct {
    kfcore_esn_model blocks[MAX_BLOCKS];
    kfcore_esn_deep_model groups[MAX_GROUPS];
    float recurrent[MAX_BLOCKS][TOTAL_NEURONS * TOTAL_NEURONS];
    float input_weights[MAX_BLOCKS][TOTAL_NEURONS * TOTAL_NEURONS];
    float bias[MAX_BLOCKS][TOTAL_NEURONS];
    float state[TOTAL_NEURONS];
    float workspace[TOTAL_NEURONS * 3];
    float features[SAMPLES * TOTAL_NEURONS];
    float inputs[SAMPLES + 1];
    float targets[SAMPLES];
    float weights[TOTAL_NEURONS];
    float best_weights[TOTAL_NEURONS];
    float gram[TOTAL_NEURONS * TOTAL_NEURONS];
} SequenceFixture;
static SequenceFixture fixture;

static void make_sequence(Task task)
{
    if (task == OSCILLATORS)
    {
        const double first_frequency = 0.17, second_frequency = 0.31;
        const double second_amplitude = 0.5;
        for (int t = 0; t <= SAMPLES; ++t)
            fixture.inputs[t] = (float)((sin(first_frequency * t) +
                second_amplitude * sin(second_frequency * t)) / (1.0 + second_amplitude));
    }
    else if (task == MACKEY_GLASS)
    {
        /* Euler h=1 discretization, constant history 1.2; not a paper dataset replica. */
        const double beta = 0.2, gamma = 0.1, initial = 1.2;
        const int exponent = 10;
        double history[MG_DELAY + 1];
        for (int i = 0; i <= MG_DELAY; ++i) history[i] = initial;
        double value = initial;
        for (int t = 0; t < MG_BURN_IN + SAMPLES + 1; ++t)
        {
            const double delayed = history[t % (MG_DELAY + 1)];
            history[(t + MG_DELAY) % (MG_DELAY + 1)] = value;
            value += beta * delayed / (1.0 + pow(delayed, exponent)) - gamma * value;
            if (t >= MG_BURN_IN) fixture.inputs[t - MG_BURN_IN] = (float)(value - 1.0);
        }
    }
    else
    {
        /* Reuse the production deterministic generator, independently of model seeds. */
        static float random_matrix[TOTAL_NEURONS * TOTAL_NEURONS];
        for (int start = 0; start <= SAMPLES; start += TOTAL_NEURONS * TOTAL_NEURONS)
        {
            const kfcore_esn_status status = kfcore_esn_init_reservoir_weights(
                random_matrix, TOTAL_NEURONS, UINT64_C(10000) + (uint64_t)start, 1.0f);
            check_equal(status, KFCORE_ESN_OK);
            for (int i = 0; i < TOTAL_NEURONS * TOTAL_NEURONS && start + i <= SAMPLES; ++i)
                fixture.inputs[start + i] = random_matrix[i];
        }
    }
    for (int t = 0; t < SAMPLES; ++t)
        fixture.targets[t] = task == DELAYED_INPUT
            ? (t >= MEMORY_DELAY ? fixture.inputs[t - MEMORY_DELAY] : 0.0f)
            : fixture.inputs[t + 1];
}

static kfcore_esn_status initialize(Architecture architecture, uint64_t seed)
{
    const int count = architecture == SHALLOW ? 1 : architecture == GROUPED_DEEP ? 4 : 2;
    const int size = TOTAL_NEURONS / count;
    const int depth = architecture == DEEP || architecture == GROUPED_DEEP ? 2 : 1;
    memset(fixture.state, 0, sizeof(fixture.state));
    for (int b = 0; b < count; ++b)
    {
        const int inputs = b % depth == 0 ? 1 : size;
        kfcore_esn_status status = kfcore_esn_init_reservoir_weights(
            fixture.recurrent[b], size, seed + (uint64_t)b, density);
        if (status != KFCORE_ESN_OK) return status;
        status = kfcore_esn_scale_spectral_radius(fixture.recurrent[b], size,
            spectral_radius, POWER_ITERATIONS, fixture.workspace);
        if (status != KFCORE_ESN_OK) return status;
        status = kfcore_esn_init_reservoir_weights(fixture.input_weights[b], size,
            seed + UINT64_C(100) + (uint64_t)b, 1.0f);
        if (status != KFCORE_ESN_OK) return status;
        for (int i = 0; i < size * inputs; ++i) fixture.input_weights[b][i] *= input_scale;
        memset(fixture.bias[b], 0, sizeof(fixture.bias[b]));
        fixture.blocks[b] = (kfcore_esn_model){ inputs, size, 0, leak_rate,
            fixture.input_weights[b], fixture.recurrent[b], fixture.bias[b], NULL, NULL };
    }
    for (int g = 0; g < count / depth; ++g)
        fixture.groups[g] = (kfcore_esn_deep_model){ depth, &fixture.blocks[g * depth] };
    return KFCORE_ESN_OK;
}

static kfcore_esn_status collect(Architecture architecture)
{
    const kfcore_esn_grouped_model grouped = { 2, fixture.blocks };
    const kfcore_esn_deep_model deep = { 2, fixture.blocks };
    const kfcore_esn_grouped_deep_model grouped_deep = { 2, fixture.groups };
    for (int t = 0; t < SAMPLES; ++t)
    {
        kfcore_esn_status status;
        switch (architecture)
        {
        case SHALLOW: status = kfcore_esn_step(&fixture.blocks[0], &fixture.inputs[t],
                          fixture.state, fixture.workspace); break;
        case GROUPED: status = kfcore_esn_grouped_step(&grouped, &fixture.inputs[t],
                          fixture.state, fixture.workspace); break;
        case DEEP: status = kfcore_esn_deep_step(&deep, &fixture.inputs[t],
                          fixture.state, fixture.workspace); break;
        default: status = kfcore_esn_grouped_deep_step(&grouped_deep, &fixture.inputs[t],
                          fixture.state, fixture.workspace); break;
        }
        if (status != KFCORE_ESN_OK) return status;
        memcpy(&fixture.features[t * TOTAL_NEURONS], fixture.state, sizeof(fixture.state));
    }
    return KFCORE_ESN_OK;
}

static double score(int begin, int count, int baseline)
{
    const float zero_bias = 0.0f;
    const kfcore_esn_model readout = { .reservoir_size = TOTAL_NEURONS, .output_size = 1,
        .output_weights = fixture.weights, .output_bias = &zero_bias };
    double sum = 0.0, squares = 0.0, error = 0.0;
    double training_mean = 0.0;
    if (baseline == 2)
    {
        for (int t = WASHOUT; t < WASHOUT + TRAIN; ++t)
            training_mean += fixture.targets[t];
        training_mean /= TRAIN;
    }
    for (int t = begin; t < begin + count; ++t)
    {
        float prediction = baseline == 2 ? (float)training_mean : fixture.inputs[t];
        if (!baseline && kfcore_esn_predict(&readout,
                &fixture.features[t * TOTAL_NEURONS], &prediction) != KFCORE_ESN_OK)
            return INFINITY;
        const double target = fixture.targets[t];
        const double residual = prediction - target;
        error += residual * residual;
        sum += target;
        squares += target * target;
    }
    const double centered_squares = squares - sum * sum / count;
    return centered_squares > 0.0 ? sqrt(error / centered_squares) : INFINITY;
}

static double evaluate(Architecture architecture, Task task, uint64_t seed)
{
    make_sequence(task);
    kfcore_esn_status status = initialize(architecture, seed);
    check_equal(status, KFCORE_ESN_OK);
    if (status != KFCORE_ESN_OK) return INFINITY;
    status = collect(architecture);
    check_equal(status, KFCORE_ESN_OK);
    if (status != KFCORE_ESN_OK) return INFINITY;
    double best = DBL_MAX;
    float selected = 0.0f;
    for (size_t i = 0; i < sizeof(ridge_candidates) / sizeof(ridge_candidates[0]); ++i)
    {
        status = kfcore_esn_fit_ridge(&fixture.features[WASHOUT * TOTAL_NEURONS],
            &fixture.targets[WASHOUT], TOTAL_NEURONS, 1, TRAIN, ridge_candidates[i],
            fixture.weights, fixture.gram);
        check_equal(status, KFCORE_ESN_OK);
        if (status != KFCORE_ESN_OK) return INFINITY;
        const double validation = score(WASHOUT + TRAIN, VALIDATION, 0);
        if (validation < best)
        {
            best = validation;
            selected = ridge_candidates[i];
            memcpy(fixture.best_weights, fixture.weights, sizeof(fixture.weights));
        }
    }
    if (selected == 0.0f) return INFINITY;
    memcpy(fixture.weights, fixture.best_weights, sizeof(fixture.weights));
    const double result = score(WASHOUT + TRAIN + VALIDATION, TEST, 0);
    const double baseline = score(WASHOUT + TRAIN + VALIDATION, TEST,
                                  task == DELAYED_INPUT ? 2 : 1);
    printf("%s,%s,%llu,%.6g,%.6g,%.6g,%.6g\n", task_names[task],
        architecture_names[architecture], (unsigned long long)seed,
        (double)selected, best, result, baseline);
    check(isfinite(result));
#ifndef KFCORE_ESN_QUALITY
    check(result < regression_nrmse_limit);
    check(result < regression_baseline_ratio * baseline);
#endif
    return result;
}

spec("esn sequence learning")
{
    it("learns held-out oscillators across four reservoir architectures")
    {
        for (int architecture = 0; architecture < ARCH_COUNT; ++architecture)
            (void)evaluate((Architecture)architecture, OSCILLATORS, UINT64_C(42));
    }
#ifdef KFCORE_ESN_QUALITY
    it("reports multi-seed temporal prediction and delayed-memory quality")
    {
        const int seed_count = 5;
        printf("task,architecture,seed,lambda,validation_nrmse,test_nrmse,baseline_nrmse\n");
        for (int task = 0; task < TASK_COUNT; ++task)
            for (int architecture = 0; architecture < ARCH_COUNT; ++architecture)
            {
                double sum = 0.0, squares = 0.0, minimum = DBL_MAX, maximum = 0.0;
                for (int seed = 0; seed < seed_count; ++seed)
                {
                    const double result = evaluate((Architecture)architecture, (Task)task,
                        UINT64_C(42) + (uint64_t)seed);
                    sum += result;
                    squares += result * result;
                    if (result < minimum) minimum = result;
                    if (result > maximum) maximum = result;
                }
                const double mean = sum / seed_count;
                const double variance = fmax(0.0, squares / seed_count - mean * mean);
                printf("summary,%s,%s,mean=%.6g,std=%.6g,min=%.6g,max=%.6g\n",
                    task_names[task], architecture_names[architecture], mean,
                    sqrt(variance), minimum, maximum);
            }
    }
#endif
}
