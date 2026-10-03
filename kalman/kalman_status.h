#ifndef KFCORE_KALMAN_STATUS_H
#define KFCORE_KALMAN_STATUS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kfcore_kalman_status
{
    KFCORE_KALMAN_OK = 0,
    KFCORE_KALMAN_INVALID_ARGUMENT = -1,
    KFCORE_KALMAN_REJECTED = -2,
    KFCORE_KALMAN_NUMERICAL_FAILURE = -3,
    KFCORE_KALMAN_WORKSPACE_TOO_SMALL = -4,
    KFCORE_KALMAN_SIZE_OVERFLOW = -5,
    KFCORE_KALMAN_CALLBACK_FAILURE = -6
} kfcore_kalman_status;

#ifdef __cplusplus
}
#endif

#endif
