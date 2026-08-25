#include "kfcore/yolo/opencv.hpp"
#include <opencv2/core.hpp>

#include "tinytest.hpp"

#include <cstddef>
#include <cstdint>

using namespace kfcore::yolo;

spec("YOLO OpenCV adapter") {
    it("borrows a strided CV_8UC3 ROI without flattening it") {
        cv::Mat parent(20, 30, CV_8UC3, cv::Scalar::all(0));
        cv::Mat roi = parent(cv::Rect(3, 4, 10, 8));

        const ImageView view = image_view(roi, PixelFormat::Bgr8);

        check(view.width == 10);
        check(view.height == 8);
        check(view.row_stride == roi.step[0]);
        check(view.data == static_cast<const void*>(roi.data));
        check(view.memory_kind == MemoryKind::Host);
    }

    it("rejects unsupported Mat element types") {
        cv::Mat gray(10, 10, CV_8UC1);

        check_throws_as(image_view(gray, PixelFormat::Bgr8), YoloError);
    }

    it("rejects an empty Mat") {
        cv::Mat empty;

        check_throws_as(image_view(empty), YoloError);
    }

    it("draws only inside an ROI") {
        cv::Mat parent(20, 30, CV_8UC3, cv::Scalar::all(0));
        const cv::Mat before = parent.clone();
        cv::Mat roi = parent(cv::Rect(3, 4, 10, 8));
        TrackFrame tracks{
            10,
            8,
            {{{{1.0f, 1.0f, 8.0f, 6.0f}, 0.9f, 2}, std::uint64_t{42}}},
        };

        draw_tracks(roi, tracks);

        bool changed_inside = false;
        for (int y = 0; y < parent.rows; ++y) {
            for (int x = 0; x < parent.cols; ++x) {
                const cv::Vec3b actual = parent.at<cv::Vec3b>(y, x);
                const cv::Vec3b expected = before.at<cv::Vec3b>(y, x);
                const bool inside_roi = x >= 3 && x < 13 && y >= 4 && y < 12;
                if (inside_roi) {
                    changed_inside = changed_inside || actual != expected;
                } else {
                    check(actual[0] == expected[0]);
                    check(actual[1] == expected[1]);
                    check(actual[2] == expected[2]);
                }
            }
        }
        check(changed_inside);
    }

    it("skips unconfirmed tracks unless explicitly requested") {
        cv::Mat image(8, 10, CV_8UC3, cv::Scalar::all(0));
        const cv::Mat before = image.clone();
        TrackFrame tracks{
            10,
            8,
            {{{{1.0f, 1.0f, 8.0f, 6.0f}, 0.9f, 2}, std::nullopt}},
        };

        draw_tracks(image, tracks);
        check(cv::norm(image, before, cv::NORM_INF) == 0.0);

        DrawOptions options;
        options.draw_unconfirmed = true;
        draw_tracks(image, tracks, options);
        check(cv::norm(image, before, cv::NORM_INF) > 0.0);
    }
}
