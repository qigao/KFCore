#include "kfcore/yolo/opencv.hpp"
#include <opencv2/core.hpp>

#include "tinytest.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

using namespace kfcore::yolo;

namespace {

cv::Mat ink_mask_above_box(const cv::Mat& image, int box_top) {
    cv::Mat mask(box_top, image.cols, CV_8UC1, cv::Scalar::all(0));
    for (int y = 0; y < box_top; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            mask.at<unsigned char>(y, x) =
                pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0 ? 255 : 0;
        }
    }
    return mask;
}

}  // namespace

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

    it("borrows continuous BGR and RGB Mats") {
        cv::Mat image(4, 6, CV_8UC3, cv::Scalar::all(0));

        const ImageView bgr = image_view(image, PixelFormat::Bgr8);
        const ImageView rgb = image_view(image, PixelFormat::Rgb8);

        check(bgr.row_stride == std::size_t{18});
        check(bgr.pixel_format == PixelFormat::Bgr8);
        check(rgb.pixel_format == PixelFormat::Rgb8);
    }

    it("rejects unsupported Mat element types") {
        cv::Mat gray(10, 10, CV_8UC1);

        check_throws_as(image_view(gray, PixelFormat::Bgr8), YoloError);
    }

    it("rejects an empty Mat") {
        cv::Mat empty;

        check_throws_as(image_view(empty), YoloError);
    }

    it("rejects an invalid pixel format") {
        cv::Mat image(4, 6, CV_8UC3, cv::Scalar::all(0));

        check_throws_as(image_view(image, static_cast<PixelFormat>(99)), YoloError);
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

    it("renders deterministic class score and ID labels") {
        cv::Mat first(32, 256, CV_8UC3, cv::Scalar::all(0));
        cv::Mat repeat = first.clone();
        cv::Mat different_class = first.clone();
        cv::Mat different_score = first.clone();
        TrackFrame base{256, 32, {{{{4.0f, 12.0f, 30.0f, 28.0f}, 0.25f, 3}, std::uint64_t{9}}}};
        TrackFrame class_changed{256, 32,
                                 {{{{4.0f, 12.0f, 30.0f, 28.0f}, 0.25f, 4}, std::uint64_t{9}}}};
        TrackFrame score_changed{256, 32,
                                 {{{{4.0f, 12.0f, 30.0f, 28.0f}, 0.75f, 3}, std::uint64_t{9}}}};

        draw_tracks(first, base);
        draw_tracks(repeat, base);
        draw_tracks(different_class, class_changed);
        draw_tracks(different_score, score_changed);

        check(cv::norm(first, repeat, cv::NORM_INF) == 0.0);
        check(cv::norm(first, different_class, cv::NORM_INF) > 0.0);
        check(cv::norm(first, different_score, cv::NORM_INF) > 0.0);
    }

    it("uses a deterministic ID-derived color at the rectangle boundary") {
        cv::Mat first(128, 192, CV_8UC3, cv::Scalar::all(0));
        cv::Mat repeat = first.clone();
        cv::Mat different_id = first.clone();
        TrackFrame first_track{192, 128,
                               {{{{100.0f, 60.0f, 150.0f, 110.0f}, 0.5f, 3},
                                 std::uint64_t{101}}}};
        TrackFrame second_track{192, 128,
                                {{{{100.0f, 60.0f, 150.0f, 110.0f}, 0.5f, 3},
                                  std::uint64_t{202}}}};

        draw_tracks(first, first_track);
        draw_tracks(repeat, first_track);
        draw_tracks(different_id, second_track);

        const cv::Vec3b first_color = first.at<cv::Vec3b>(80, 100);
        const cv::Vec3b repeat_color = repeat.at<cv::Vec3b>(80, 100);
        const cv::Vec3b second_color = different_id.at<cv::Vec3b>(80, 100);
        check(first_color == repeat_color);
        check(first_color != second_color);
    }

    it("includes confirmed IDs in the label independently of their color") {
        constexpr int box_top = 55;
        cv::Mat first(96, 512, CV_8UC3, cv::Scalar::all(0));
        cv::Mat second = first.clone();
        TrackFrame first_track{512, 96,
                               {{{{4.0f, static_cast<float>(box_top), 400.0f, 85.0f},
                                  0.25f, 3},
                                 std::uint64_t{101}}}};
        TrackFrame second_track{512, 96,
                                {{{{4.0f, static_cast<float>(box_top), 400.0f, 85.0f},
                                   0.25f, 3},
                                  std::uint64_t{202}}}};

        draw_tracks(first, first_track);
        draw_tracks(second, second_track);

        check(cv::norm(ink_mask_above_box(first, box_top),
                       ink_mask_above_box(second, box_top), cv::NORM_INF) > 0.0);
    }

    it("labels explicitly drawn unconfirmed tracks") {
        cv::Mat low_score(32, 256, CV_8UC3, cv::Scalar::all(0));
        cv::Mat high_score = low_score.clone();
        TrackFrame low{256, 32, {{{{4.0f, 12.0f, 30.0f, 28.0f}, 0.25f, 3}, std::nullopt}}};
        TrackFrame high{256, 32, {{{{4.0f, 12.0f, 30.0f, 28.0f}, 0.75f, 3}, std::nullopt}}};
        DrawOptions options;
        options.draw_unconfirmed = true;

        draw_tracks(low_score, low, options);
        draw_tracks(high_score, high, options);

        check(cv::norm(low_score, high_score, cv::NORM_INF) > 0.0);
    }

    it("validates every track before changing pixels") {
        cv::Mat image(20, 30, CV_8UC3, cv::Scalar::all(0));
        const cv::Mat before = image.clone();
        TrackFrame tracks{
            30,
            20,
            {
                {{{1.0f, 1.0f, 10.0f, 10.0f}, 0.9f, 1}, std::uint64_t{1}},
                {{{2.0f, 2.0f, 12.0f, 12.0f},
                  (std::numeric_limits<float>::quiet_NaN)(), 1},
                 std::uint64_t{2}},
            },
        };

        check_throws_as(draw_tracks(image, tracks), YoloError);
        check(cv::norm(image, before, cv::NORM_INF) == 0.0);
    }

    it("rejects invalid draw frames and clips finite boxes") {
        cv::Mat image(20, 30, CV_8UC3, cv::Scalar::all(0));
        TrackFrame mismatched{29, 20, {}};
        DrawOptions bad_options;
        bad_options.line_thickness = 0;
        TrackFrame nonfinite{30, 20,
                             {{{{(std::numeric_limits<float>::infinity)(), 1.0f, 2.0f, 3.0f},
                                0.9f, 1},
                               std::uint64_t{1}}}};
        TrackFrame reversed{30, 20,
                            {{{{4.0f, 4.0f, 2.0f, 8.0f}, 0.9f, 1}, std::uint64_t{1}}}};
        TrackFrame clipped{30, 20,
                           {{{{-1000.0f, -1000.0f, 1000.0f, 1000.0f}, 0.9f, 1},
                             std::uint64_t{1}}}};

        check_throws_as(draw_tracks(image, mismatched), YoloError);
        check_throws_as(draw_tracks(image, TrackFrame{30, 20, {}}, bad_options), YoloError);
        check_throws_as(draw_tracks(image, nonfinite), YoloError);
        check_throws_as(draw_tracks(image, reversed), YoloError);
        draw_tracks(image, clipped);
        check(cv::norm(image, cv::NORM_INF) > 0.0);
    }

    it("rejects invalid scores and font scales without partial rendering") {
        cv::Mat image(48, 96, CV_8UC3, cv::Scalar::all(0));
        const cv::Mat before = image.clone();
        TrackFrame below_range{96, 48,
                               {
                                   {{{4.0f, 20.0f, 40.0f, 42.0f}, 0.5f, 3},
                                    std::uint64_t{1}},
                                   {{{44.0f, 20.0f, 80.0f, 42.0f}, -0.01f, 3},
                                    std::uint64_t{2}},
                               }};
        TrackFrame above_range{96, 48,
                               {
                                   {{{4.0f, 20.0f, 40.0f, 42.0f}, 0.5f, 3},
                                    std::uint64_t{1}},
                                   {{{44.0f, 20.0f, 80.0f, 42.0f}, 1.01f, 3},
                                    std::uint64_t{2}},
                               }};
        TrackFrame valid{96, 48,
                         {{{{4.0f, 20.0f, 40.0f, 42.0f}, 0.5f, 3}, std::uint64_t{1}}}};
        DrawOptions zero_font;
        zero_font.font_scale = 0.0;
        DrawOptions nan_font;
        nan_font.font_scale = (std::numeric_limits<double>::quiet_NaN)();

        check_throws_as(draw_tracks(image, below_range), YoloError);
        check(cv::norm(image, before, cv::NORM_INF) == 0.0);
        check_throws_as(draw_tracks(image, above_range), YoloError);
        check(cv::norm(image, before, cv::NORM_INF) == 0.0);
        check_throws_as(draw_tracks(image, valid, zero_font), YoloError);
        check(cv::norm(image, before, cv::NORM_INF) == 0.0);
        check_throws_as(draw_tracks(image, valid, nan_font), YoloError);
        check(cv::norm(image, before, cv::NORM_INF) == 0.0);
    }
}
