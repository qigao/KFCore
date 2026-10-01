#include "linalg.h"

#include <stdio.h>

static int failures = 0;

static void expect_zero(const char* name, int status)
{
    if (status != 0)
    {
        fprintf(stderr, "%s: expected success, got %d\n", name, status);
        ++failures;
    }
}

static void expect_failure(const char* name, int status)
{
    if (status == 0)
    {
        fprintf(stderr, "%s: expected failure, got success\n", name);
        ++failures;
    }
}

int main(void)
{
    float a[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    float b[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float x[2] = { 1.0f, 2.0f };
    float y[2] = { 0.0f, 0.0f };

    expect_zero("matmul valid",
                matmul("N", "N", 2, 2, 2, 1.0f, a, b, 0.0f, c));
    expect_failure("matmul invalid transpose",
                   matmul("X", "N", 1, 1, 1, 1.0f, a, b, 0.0f, c));

    expect_zero("matmulsym valid", matmulsym(a, b, 2, 2, c));
    expect_failure("matmulsym negative rows", matmulsym(a, b, -1, 1, c));

    expect_zero("matvec valid", matvec("N", 2, 2, 1.0f, a, x, 0.0f, y));
    expect_failure("matvec invalid transpose",
                   matvec("X", 2, 2, 1.0f, a, x, 0.0f, y));

    expect_zero("rank1update valid", rank1update(c, x, x, 2, 2, 1.0f));
    expect_failure("rank1update negative rows",
                   rank1update(c, x, x, -1, 1, 1.0f));

    expect_zero("trisolve valid", trisolve(a, b, 2, 2, "N"));
    expect_failure("trisolve invalid transpose", trisolve(a, b, 2, 2, "X"));

    expect_zero("trisolveright valid", trisolveright(a, b, 2, 2, "N"));
    expect_failure("trisolveright invalid transpose",
                   trisolveright(a, b, 2, 2, "X"));

    expect_zero("symmetricrankupdate valid", symmetricrankupdate(a, b, 2, 2));
    expect_failure("symmetricrankupdate negative rows",
                   symmetricrankupdate(a, b, -1, 1));

    if (failures != 0)
    {
        fprintf(stderr, "linalg status contract: %d failure(s)\n", failures);
        return 1;
    }

    puts("linalg status contract: PASS");
    return 0;
}
