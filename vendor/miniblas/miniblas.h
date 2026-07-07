/** @file miniblas.h
 * Minimal BLAS backend
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * @brief Minimal generic BLAS implementation
 *
 * Note: all matrices are stored in column-major order.
 *
 * @{ */

/******************************************************************************
 * SYSTEM INCLUDE FILES
 ******************************************************************************/

/******************************************************************************
 * PROJECT INCLUDE FILES
 ******************************************************************************/

/******************************************************************************
 * DEFINES
 ******************************************************************************/

/******************************************************************************
 * TYPEDEFS
 ******************************************************************************/

/******************************************************************************
 * LOCAL DATA DEFINITIONS
 ******************************************************************************/

/******************************************************************************
 * LOCAL FUNCTION PROTOTYPES
 ******************************************************************************/

/******************************************************************************
 * FUNCTION PROTOTYPES
 ******************************************************************************/

#ifdef __cplusplus
extern "C"
{
#endif

    int lsame_(const char* a, const char* b);

    int scopy_(int* n, const float* sx, int* incx, float* sy, int* incy);

    int sswap_(int* n, float* sx, int* incx, float* sy, int* incy);

    int sscal_(int* n, float* sa, float* sx, int* incx);

    int saxpy_(int* n, float* sa, const float* sx, int* incx, float* sy, int* incy);

    float sdot_(int* n, const float* sx, int* incx, const float* sy, int* incy);

    float snrm2_(int* n, const float* sx, int* incx);

    int sgemv_(const char* trans, int* m, int* n, float* alpha, const float* a, int* lda,
               const float* x, int* incx, float* beta, float* y, int* incy);

    int sger_(int* m, int* n, float* alpha, const float* x, int* incx, const float* y, int* incy,
              float* a, int* lda);

    int svec_mean_(int* n, const float* sx, int* incx, float* mean);

    int svec_variance_(int* n, const float* sx, int* incx, int* ddof, float* variance);

    int svec_rms_(int* n, const float* sx, int* incx, float* rms);

    int svec_normalize_(int* n, float* sx, int* incx, float* eps, float* norm);

    int svec_l1_distance_(int* n, const float* sx, int* incx, const float* sy, int* incy,
                          float* distance);

    int svec_linf_distance_(int* n, const float* sx, int* incx, const float* sy, int* incy,
                            float* distance);

    int svec_cosine_similarity_(int* n, const float* sx, int* incx, const float* sy, int* incy,
                                float* cosine);

    float smat2_det_(const float* a);

    int smat2_inv_(const float* a, float* inv_a, float* eps);

    float smat3_det_(const float* a);

    int smat3_inv_(const float* a, float* inv_a, float* eps);

    int strsm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m,
              int* n, float* alpha, const float* a, int* lda, float* b, int* ldb);

    int sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda,
              float* b, int* ldb, float* beta, float* c__, int* ldc);

    int ssyrk_(char* uplo, char* trans, int* n, int* k, float* alpha, float* a, int* lda,
              float* beta, float* c__, int* ldc);

    int ssymm_(char* side, char* uplo, int* m, int* n, float* alpha, float* a, int* lda, float* b,
              int* ldb, float* beta, float* c__, int* ldc);

    int strmm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m,
               int* n, float* alpha, float* a, int* lda, float* b, int* ldb);

#ifdef __cplusplus
}
#endif

/* @} */
