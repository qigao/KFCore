#pragma once
#include <kfcore/mediapipe/gesture_recognizer.hpp>

namespace preview {
inline constexpr char kNoHandStatus[] = "No hand detected";

// Display wording does not change the official category or its ESN input score.
inline const char* gesture_display_name(kfcore::mediapipe::CannedGesture gesture) {
    return gesture == kfcore::mediapipe::CannedGesture::None
        ? "Unrecognized (None)" : kfcore::mediapipe::gesture_name(gesture);
}
} // namespace preview
