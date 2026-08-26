#include "kfcore/sift/error.hpp"
#include "kfcore/sift/popsift_options.hpp"
#include "tinytest.hpp"

#include <limits>
#include <string>

using namespace kfcore::sift;

namespace
{

template <typename Callable>
void expect_sift_error(Callable&& callable, SiftErrorCode code, const char* message_fragment)
{
    try
    {
        callable();
        check(false);
    }
    catch (const SiftError& error)
    {
        check(error.code() == code);
        check(std::string(error.what()).find(message_fragment) != std::string::npos);
    }
}

} // namespace

spec("PopSift options")
{
    it("provides finite defaults and accepts explicit valid limits")
    {
        PopSiftOptions defaults;
        defaults.validate();
        check(defaults.device == 0);
        check(defaults.max_image_bytes == PopSiftOptions::kDefaultMaxImageBytes);
        check(defaults.max_features == PopSiftOptions::kDefaultMaxFeatures);
        check(defaults.max_pending_jobs == PopSiftOptions::kDefaultMaxPendingJobs);

        PopSiftOptions options;
        options.device          = 2;
        options.max_image_bytes = 4096;
        options.max_features    = 512;
        options.max_pending_jobs = 4;
        options.normalization   = PopSiftDescriptorNormalization::Classic;
        options.validate();
    }

    it("rejects invalid devices and unrepresentable or zero limits")
    {
        PopSiftOptions options;
        options.device = -1;
        expect_sift_error([&] { options.validate(); }, SiftErrorCode::InvalidArgument, "device");

        options = {};
        options.max_image_bytes = 0;
        expect_sift_error([&] { options.validate(); },
                          SiftErrorCode::ResourceLimitExceeded, "image byte limit");

        options = {};
        options.max_features = 0;
        expect_sift_error([&] { options.validate(); },
                          SiftErrorCode::ResourceLimitExceeded, "feature limit");

        options.max_features = PopSiftOptions::kMaximumMaxFeatures + 1U;
        expect_sift_error([&] { options.validate(); },
                          SiftErrorCode::ResourceLimitExceeded, "safe PopSift range");

        options.max_features = PopSiftOptions::kMaximumMaxFeatures;
        options.validate();

        options = {};
        options.max_pending_jobs = 0;
        expect_sift_error([&] { options.validate(); },
                          SiftErrorCode::ResourceLimitExceeded, "pending job limit");

        options.max_pending_jobs = PopSiftOptions::kMaximumMaxPendingJobs + 1U;
        expect_sift_error([&] { options.validate(); },
                          SiftErrorCode::ResourceLimitExceeded, "pending job limit");

        options.max_pending_jobs = PopSiftOptions::kMaximumMaxPendingJobs;
        options.validate();

        options = {};
        options.normalization = static_cast<PopSiftDescriptorNormalization>(99);
        expect_sift_error([&] { options.validate(); }, SiftErrorCode::InvalidArgument,
                          "normalization");
    }
}
