#include <kfcore/yolo/tracking.hpp>

int main()
{
    kfcore::yolo::ByteTrackSession session;
    const auto result = session.update({640, 480, {}});
    return result.detections.empty() ? 0 : 1;
}
