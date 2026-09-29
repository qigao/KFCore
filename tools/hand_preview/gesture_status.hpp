#pragma once
#include <kfcore/mediapipe/gesture_recognizer.hpp>
#include <kfcore/gesture_interaction/composition_esn.hpp>
#include <kfcore/gesture_interaction/interaction.hpp>

namespace preview {
inline constexpr char kNoHandStatus[] = "未检测到手";

// Display wording does not change the official category or its ESN input score.
inline const char* gesture_display_name(kfcore::mediapipe::CannedGesture gesture) {
    using kfcore::mediapipe::CannedGesture;
    switch (gesture) {
    case CannedGesture::None: return "未识别";
    case CannedGesture::ClosedFist: return "握拳";
    case CannedGesture::OpenPalm: return "张掌";
    case CannedGesture::PointingUp: return "食指向上";
    case CannedGesture::ThumbDown: return "拇指向下";
    case CannedGesture::ThumbUp: return "拇指向上";
    case CannedGesture::Victory: return "胜利手势";
    case CannedGesture::ILoveYou: return "我爱你手势";
    }
    return "未知手势";
}

inline const char* composition_display_name(kfcore::gesture_interaction::Composition label) {
    using kfcore::gesture_interaction::Composition;
    switch (label) {
    case Composition::None: return "中性";
    case Composition::OpenCloseOpen: return "张掌-握拳-张掌";
    case Composition::CloseOpenClose: return "握拳-张掌-握拳";
    case Composition::Grasp: return "抓取";
    case Composition::Release: return "放开";
    case Composition::Wave: return "挥手";
    }
    return "未知动作";
}

inline const char* event_display_name(kfcore::gesture_interaction::EventKind kind) {
    using kfcore::gesture_interaction::EventKind;
    switch (kind) {
    case EventKind::GestureStarted: return "手势开始";
    case EventKind::GestureEnded: return "手势结束";
    case EventKind::GestureCancelled: return "手势取消";
    case EventKind::Grasp: return "抓取";
    case EventKind::Release: return "放开";
    case EventKind::Wave: return "挥手";
    case EventKind::GraspCancelled: return "抓取取消";
    }
    return "未知事件";
}

inline const char* event_reason_display_name(kfcore::gesture_interaction::EventReason reason) {
    using kfcore::gesture_interaction::EventReason;
    switch (reason) {
    case EventReason::Recognized: return "已识别";
    case EventReason::Unrecognized: return "未识别";
    case EventReason::HandLost: return "手部丢失";
    case EventReason::FrameGap: return "帧间隔过长";
    case EventReason::MultipleHands: return "检测到多只手";
    case EventReason::SourceChanged: return "视频源改变";
    case EventReason::Reset: return "已重置";
    case EventReason::ModelChanged: return "模型改变";
    case EventReason::Capacity: return "容量已满";
    }
    return "未知原因";
}
} // namespace preview
