#include "kfcore/yolo/opencv.hpp"
#include "kfcore/yolo/tensorrt.hpp"
#include "kfcore/yolo/tracking.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Arguments {
    fs::path engine;
    fs::path images;
    fs::path output;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

Arguments parse_arguments(int argc, char** argv) {
    if (argc != 7) {
        fail("usage: track_image_sequence --engine <engine> --images <directory> --output <directory>");
    }
    Arguments arguments;
    bool seen_engine = false;
    bool seen_images = false;
    bool seen_output = false;
    for (int index = 1; index < argc; index += 2) {
        const std::string option(argv[index]);
        if (index + 1 >= argc || argv[index + 1][0] == '\0') {
            fail("missing value for " + option);
        }
        if (option == "--engine") {
            if (seen_engine) fail("duplicate --engine option");
            arguments.engine = argv[index + 1];
            seen_engine = true;
        } else if (option == "--images") {
            if (seen_images) fail("duplicate --images option");
            arguments.images = argv[index + 1];
            seen_images = true;
        } else if (option == "--output") {
            if (seen_output) fail("duplicate --output option");
            arguments.output = argv[index + 1];
            seen_output = true;
        } else {
            fail("unknown option: " + option);
        }
    }
    if (!seen_engine || !seen_images || !seen_output) {
        fail("--engine, --images, and --output are all required");
    }
    return arguments;
}

std::string lowercase_extension(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension;
}

bool is_supported_image(const fs::path& path) {
    const std::string extension = lowercase_extension(path);
    static const std::vector<std::string> extensions = {
        ".avif", ".bmp", ".dib", ".exr", ".hdr", ".jpeg", ".jpg", ".jpe", ".jp2",
        ".pbm", ".pgm", ".pic", ".png", ".pnm", ".ppm", ".pxm", ".ras", ".sr",
        ".tif", ".tiff", ".webp",
    };
    return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
}

void require_directory(const fs::path& path, const char* option) {
    std::error_code error;
    if (!fs::is_directory(path, error) || error) {
        fail(std::string(option) + " is not a readable directory: " + path.string());
    }
}

std::vector<fs::path> image_paths(const fs::path& directory) {
    std::vector<fs::path> paths;
    std::error_code error;
    fs::directory_iterator iterator(directory, error);
    if (error) {
        fail("cannot enumerate --images directory: " + directory.string());
    }
    for (const fs::directory_entry& entry : iterator) {
        if (entry.is_regular_file(error) && !error && is_supported_image(entry.path())) {
            paths.push_back(entry.path());
        }
        if (error) {
            fail("cannot inspect image sequence entry: " + entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end(), [](const fs::path& left, const fs::path& right) {
        return left.generic_u8string() < right.generic_u8string();
    });
    if (paths.empty()) {
        fail("--images directory contains no supported images: " + directory.string());
    }
    return paths;
}

void prepare_output(const fs::path& images, const fs::path& output) {
    std::error_code error;
    if (fs::exists(output, error)) {
        if (error || !fs::is_directory(output, error) || error) {
            fail("--output must be a directory or a creatable directory: " + output.string());
        }
    } else if (!fs::create_directories(output, error) || error) {
        fail("cannot create --output directory: " + output.string());
    }
    if (fs::equivalent(images, output, error) && !error) {
        fail("--output must not be the same directory as --images");
    }
    if (error) {
        fail("cannot compare --images and --output directories");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        require_directory(arguments.images, "--images");
        prepare_output(arguments.images, arguments.output);
        const std::vector<fs::path> inputs = image_paths(arguments.images);

        const std::shared_ptr<const kfcore::yolo::Engine> engine =
            kfcore::yolo::Engine::load(arguments.engine);
        std::unique_ptr<kfcore::yolo::TensorRtDetector> detector = engine->create_detector();
        kfcore::yolo::ByteTrackSession session;
        session.reset();

        for (const fs::path& input : inputs) {
            cv::Mat image = cv::imread(input.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                fail("failed to decode image: " + input.string());
            }
            const kfcore::yolo::DetectionFrame detections =
                detector->detect(kfcore::yolo::image_view(image));
            const kfcore::yolo::TrackFrame tracks = session.update(detections);
            kfcore::yolo::draw_tracks(image, tracks);

            const fs::path destination = arguments.output / input.filename();
            if (!cv::imwrite(destination.string(), image)) {
                fail("failed to write annotated image: " + destination.string());
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "track_image_sequence: " << error.what() << '\n';
        return 1;
    }
}
