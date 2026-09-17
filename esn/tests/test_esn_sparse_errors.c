#include "esn_sparse.h"
#define TINYTEST_NO_MAIN
#include "tinytest.h"

spec("kfcore esn sparse errors")
{
    it("rejects a null nonzero-count output")
    {
        const float dense[1] = { 1.0f };
        check_equal(kfcore_esn_sparse_count_nonzero(dense, 1, NULL),
                    KFCORE_ESN_INVALID_ARGUMENT);
    }
}
