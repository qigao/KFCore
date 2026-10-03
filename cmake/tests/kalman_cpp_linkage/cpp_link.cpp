#include "kalman_status.h"
#include "kalman_takasu.h"
#include "kalman_udu.h"
#include "kalman_ekf.h"
#include "kalman_ukf.h"
#include "signal_filters.h"
#include "frame_transform.h"
#include "navtoolbox.h"
#include "nav_fusion2d.h"
#include "nav_fusion3d.h"

#include <cstddef>

int main()
{
    std::size_t required = 0;
    float alpha = 0.0f;
    float rotation[9] = {};
    float translation[3] = {};

    if (kalman_predict_workspace_floats(1, 0, &required) != KFCORE_KALMAN_OK)
        return 1;
    if (kalman_ekf_takasu_predict_workspace_floats(1, 0, &required) != KFCORE_KALMAN_OK)
        return 2;
    if (kalman_ukf_predict_workspace_floats(1, &required) != KFCORE_KALMAN_OK)
        return 3;
    if (kf_signal_lowpass_alpha(1.0f, 0.01f, &alpha) != 0)
        return 4;

    frame_pose_identity(rotation, translation);
    (void)nav_wrap_pi(0.0f);

    return alpha > 0.0f ? 0 : 5;
}
