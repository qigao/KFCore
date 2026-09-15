#include "kfcore/hand_gesture/runtime.hpp"
#include "kfcore/hand_models/types.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

namespace
{

kfcore::hand_models::HandResult sample_hand(float translate_x)
{
    using namespace kfcore::hand_models;

    HandResult hand;
    hand.track_id = 1;
    hand.handedness = Handedness::Right;
    hand.gesture = Gesture::Open;
    hand.landmark_confidence = 0.95F;
    hand.palm.confidence = 0.97F;
    hand.palm.box = {40.0F + translate_x, 40.0F, 160.0F, 160.0F};

    for (std::size_t index = 0U; index < hand.landmarks.size(); ++index)
    {
        hand.landmarks[index] = {
            100.0F + translate_x + static_cast<float>(index) * 2.0F,
            80.0F + static_cast<float>(index),
            static_cast<float>(index) * 0.25F,
        };
    }
    return hand;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: kfgesture-smoke <model-package> <onnxruntime-plugin>\n";
        return 2;
    }

    try
    {
        kfcore::runtime::Runtime runtime;
        (void)runtime.load_backend(argv[2]);
        const auto package = kfcore::runtime::ModelPackage::load(argv[1]);
        const auto policy = kfcore::runtime::ExecutionPolicy::exact("onnxruntime", "cpu");
        auto recognizer = kfcore::hand_gesture::TemporalGestureRecognizer::load(
            runtime, package, policy);

        for (std::uint64_t frame_index = 0U; frame_index < 3U; ++frame_index)
        {
            kfcore::hand_models::HandFrame frame;
            frame.hands.push_back(sample_hand(static_cast<float>(frame_index) * 8.0F));
            const kfcore::hand_gesture::GestureFrameMetadata metadata{
                1'000'000'000ULL + frame_index * 33'333'333ULL,
                640,
                480,
            };
            const auto events = recognizer->update(frame, metadata);
            std::cout << "frame=" << frame_index << " events=" << events.size() << '\n';
        }

        const auto& route = recognizer->execution_route();
        std::cout << "route=" << route.backend_id << '/' << route.device_id << '\n';
        std::cout << "temporal gesture runtime smoke ready\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "temporal gesture smoke failed: " << error.what() << '\n';
        return 1;
    }
}
