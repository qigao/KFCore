/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cmath>
#include <cstring>
#include "popsift.h"

#include "gauss_filter.h"
#include "scale_geometry.h"
#include "sift_config.h"
#include "sift_pyramid.h"
#include "common/debug_macros.h"
#if defined(KFCORE_POPSIFT_TESTING)
#include "common/test_hooks.h"
#endif

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace
{

void selectCudaDevice(int device)
{
    const cudaError_t error = cudaSetDevice(device);
    std::ostringstream message;
    message << "Cannot set CUDA device " << device;
    POP_CUDA_FATAL_TEST(error, message.str());
}

std::size_t checkedImageBytes(int width, int height, std::size_t element_size,
                              const void* image_data)
{
    if (width <= 0 || height <= 0 || image_data == nullptr)
    {
        throw std::invalid_argument("PopSift job requires positive dimensions and image data");
    }
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
    if (w > (std::numeric_limits<std::size_t>::max)() / h)
    {
        throw std::overflow_error("PopSift job dimensions overflow the host byte count");
    }
    const std::size_t pixels = w * h;
    if (pixels > (std::numeric_limits<std::size_t>::max)() / element_size)
    {
        throw std::overflow_error("PopSift job element size overflows the host byte count");
    }
    return pixels * element_size;
}

template <typename ImageType>
void populateImagePool(popsift::SyncQueue<popsift::ImageBase*>& pool,
                       std::size_t image_count)
{
    for (std::size_t index = 0; index < image_count; ++index)
    {
        std::unique_ptr<popsift::ImageBase> image(new ImageType);
        if (!pool.push(image.get()))
        {
            throw std::runtime_error("PopSift image pool closed during initialization");
        }
        image.release();
    }
}

void returnImage(popsift::SyncQueue<popsift::ImageBase*>& pool,
                 popsift::ImageBase*& image) noexcept
{
    if (image == nullptr)
    {
        return;
    }
    if (!pool.push(image))
    {
        delete image;
    }
    image = nullptr;
}

} // namespace

PopSift::PopSift( const popsift::Config& config, popsift::Config::ProcessingMode mode,
                  ImageMode imode, int device, std::size_t max_pending_jobs )
    : _pipe(max_pending_jobs)
    , _image_mode( imode )
    , _device(device)
{
    try
    {
        selectCudaDevice(_device);
        configure(config);

        if( imode == ByteImages )
        {
            populateImagePool<popsift::Image>(_pipe._unused, kImagePoolCapacity);
        }
        else
        {
            populateImagePool<popsift::ImageFloat>(_pipe._unused, kImagePoolCapacity);
        }

        _pipe._thread_stage1.reset( new std::thread( &PopSift::uploadImages, this ));
        if( mode == popsift::Config::ExtractingMode )
            _pipe._thread_stage2.reset( new std::thread( &PopSift::extractDownloadLoop, this ));
        else
            _pipe._thread_stage2.reset( new std::thread( &PopSift::matchPrepareLoop, this ));
    }
    catch (...)
    {
        _pipe.uninit();
        _isInit = false;
        throw;
    }
}

PopSift::PopSift( ImageMode imode, int device, std::size_t max_pending_jobs )
    : _pipe(max_pending_jobs)
    , _image_mode( imode )
    , _device(device)
{
    try
    {
        selectCudaDevice(_device);
        if( imode == ByteImages )
        {
            populateImagePool<popsift::Image>(_pipe._unused, kImagePoolCapacity);
        }
        else
        {
            populateImagePool<popsift::ImageFloat>(_pipe._unused, kImagePoolCapacity);
        }

        _pipe._thread_stage1.reset( new std::thread( &PopSift::uploadImages, this ));
        _pipe._thread_stage2.reset( new std::thread( &PopSift::extractDownloadLoop, this ));
    }
    catch (...)
    {
        _pipe.uninit();
        _isInit = false;
        throw;
    }
}

PopSift::~PopSift()
{
    if(_isInit)
    {
        uninit();
    }
}

bool PopSift::configure( const popsift::Config& config, bool /*force*/ )
{
    std::lock_guard<std::mutex> lock(_config_mutex);
    if( _configuration_locked ) {
        return false;
    }

    _config = config;
    _config.levels = max( 2, config.levels );

    return true;
}

bool PopSift::applyConfiguration(bool force)
{
    if( force || ( _config  != _shadow_config ) )
    {
        popsift::init_filter( _config,
                              _config.sigma,
                              _config.levels );
        popsift::init_constants(  _config.sigma,
                                  _config.levels,
                                  _config.getPeakThreshold(),
                                  _config._edge_limit,
                                  _config.getMaxExtrema(),
                                  _config.getNormalizationMultiplier() );
    }
    _shadow_config = _config;
    return true;
}

void PopSift::private_apply_scale_factor( int& w, int& h )
{
    const popsift::ScaledImageGeometry geometry =
        popsift::scaleImageGeometry(_config, w, h);
    w = geometry.width;
    h = geometry.height;
}

void PopSift::lockConfigurationForImage(int w, int h)
{
    std::lock_guard<std::mutex> lock(_config_mutex);
    if (!_configuration_locked)
    {
        const popsift::ScaledImageGeometry geometry =
            popsift::scaleImageGeometry(_config, w, h);
        _config.octaves = geometry.octaves;
        _configuration_locked = true;
    }
}

bool PopSift::private_init( int w, int h )
{
    Pipe& p = _pipe;

    private_apply_scale_factor( w, h );

    if( p._pyramid != nullptr ) {
        p._pyramid->resetDimensions( _config, w, h );
        return true;
    }

    p._pyramid = new popsift::Pyramid( _config, w, h );

    const cudaError_t sync_error = cudaDeviceSynchronize();
    POP_CUDA_FATAL_TEST(sync_error, "Cannot synchronize CUDA SIFT pyramid initialization");

    return true;
}

bool PopSift::private_uninit()
{
    Pipe& p = _pipe;

    delete p._pyramid;
    p._pyramid = nullptr;

    return true;
}

void PopSift::uninit( )
{
    if(!_isInit)
    {
        std::cerr << "[warning] Attempt to release resources from an uninitialized instance" << std::endl;
        return;
    }
    _pipe.uninit();

    _isInit = false;
}

PopSift::AllocTest PopSift::testTextureFit( int width, int height )
{
    popsift::Config config;
    {
        std::lock_guard<std::mutex> lock(_config_mutex);
        config = _config;
    }
    const bool warn = popsift::cuda::device_prop_t::dont_warn;
    bool retval = _device_properties.checkLimit_2DtexLinear( width,
                                                        height,
                                                        warn );
    if( !retval )
    {
        return AllocTest::ImageExceedsLinearTextureLimit;
    }


    /* Scale the width and height - we need that size for the largest
     * octave. */
    const popsift::ScaledImageGeometry geometry =
        popsift::scaleImageGeometry(config, width, height);
    width = geometry.width;
    height = geometry.height;

    /* _config.level does not contain the 3 blur levels beyond the first
     * that is required for downscaling to the following octave.
     * We need all layers to check if we can support enough layers.
     */
    int depth = config.levels + 3;

    retval = _device_properties.checkLimit_2DsurfLayered( width,
                                                          height,
                                                          depth,
                                                          warn );

    return (retval ? AllocTest::Ok : AllocTest::ImageExceedsLayeredSurfaceLimit);
}

std::string PopSift::testTextureFitErrorString( AllocTest err, int width, int height )
{
    ostringstream ostr;

    switch( err )
    {
        case AllocTest::Ok :
            ostr << "?    No error." << endl;
            break;
        case AllocTest::ImageExceedsLinearTextureLimit :
            _device_properties.checkLimit_2DtexLinear( width, height, false );
            ostr << "E    Cannot load unscaled image. " << endl
                 << "E    It exceeds the max CUDA linear texture size. " << endl
                 << "E    Max is (" << width << "," << height << ")" << endl;
            break;
        case AllocTest::ImageExceedsLayeredSurfaceLimit :
            {
                const float upscaleFactor = _config.getUpscaleFactor();
                const float scaleFactor = 1.0f / powf( 2.0f, -upscaleFactor );
                int w = ceilf( width  * scaleFactor );
                int h = ceilf( height * scaleFactor );
                int d = _config.levels + 3;

                _device_properties.checkLimit_2DsurfLayered( w, h, d, false );

                w = w / scaleFactor;
                h = h / scaleFactor;
                ostr << "E    Cannot use"
                     << (upscaleFactor==1 ? " default " : " ")
                     << "downscaling factor " << -upscaleFactor
                     << " (i.e. upscaling by " << pow(2,upscaleFactor) << "). "
                     << endl
                     << "E    It exceeds the max CUDA layered surface size. " << endl
                     << "E    Change downscaling to fit into (" << w << "," << h
                     << ") with " << (d-3) << " levels per octave." << endl;
            }
            break;
        default:
            ostr << "E    Programming error, please report." << endl;
            break;
    }
    return ostr.str();
}


SiftJob* PopSift::enqueue( int                  w,
                           int                  h,
                           const unsigned char* imageData )
{
    if( _image_mode != ByteImages )
    {
        stringstream ss;
        ss << "Image mode error" << endl
           << "E    Cannot load byte images into a PopSift pipeline configured for float images";
        POP_FATAL(ss.str());
    }

    AllocTest a = testTextureFit( w, h );
    if( a != AllocTest::Ok )
    {
        cerr << __FILE__ << ":" << __LINE__ << " Image too large" << endl
             << testTextureFitErrorString( a,w,h );
        return nullptr;
    }

    auto reservation = _pipe._queue_stage1.reserve();
    if (!reservation)
    {
        throw std::runtime_error("PopSift pipeline is closed and cannot accept a byte image");
    }
    std::unique_ptr<SiftJob> job(new SiftJob( w, h, imageData ));
    lockConfigurationForImage(w, h);
    if (!reservation.commit(job.get()))
    {
        throw std::runtime_error("PopSift pipeline closed while accepting a byte image");
    }
    return job.release();
}

SiftJob* PopSift::enqueue( int          w,
                           int          h,
                           const float* imageData )
{
    if( _image_mode != FloatImages )
    {
        stringstream ss;
        ss << "Image mode error" << endl
           << "E    Cannot load float images into a PopSift pipeline configured for byte images";
        POP_FATAL(ss.str());
    }

    AllocTest a = testTextureFit( w, h );
    if( a != AllocTest::Ok )
    {
        cerr << __FILE__ << ":" << __LINE__ << " Image too large" << endl
             << testTextureFitErrorString( a,w,h );
        return nullptr;
    }

    auto reservation = _pipe._queue_stage1.reserve();
    if (!reservation)
    {
        throw std::runtime_error("PopSift pipeline is closed and cannot accept a float image");
    }
    std::unique_ptr<SiftJob> job(new SiftJob( w, h, imageData ));
    lockConfigurationForImage(w, h);
    if (!reservation.commit(job.get()))
    {
        throw std::runtime_error("PopSift pipeline closed while accepting a float image");
    }
    return job.release();
}

void PopSift::uploadImages( )
{
    std::exception_ptr startup_error;
    try
    {
        selectCudaDevice(_device);
#if defined(KFCORE_POPSIFT_TESTING)
        popsift::testing::throwIfWorkerFailureInjected(
            popsift::testing::WorkerFailureStage::UploadStartup);
#endif
    }
    catch (...)
    {
        startup_error = std::current_exception();
        _pipe._queue_stage1.close();
    }

    SiftJob* job = nullptr;
    while (_pipe._queue_stage1.pull(job))
    {
        if (startup_error != nullptr)
        {
            job->setError(startup_error);
            continue;
        }

        popsift::ImageBase* image = nullptr;
        try
        {
            if (!_pipe._unused.pull(image))
            {
                throw std::runtime_error("PopSift staging image pool closed during upload");
            }
#if defined(KFCORE_POPSIFT_TESTING)
            popsift::testing::throwIfWorkerFailureInjected(
                popsift::testing::WorkerFailureStage::UploadJob);
#endif
            job->setImg(image);
            if (!_pipe._queue_stage2.push(job))
            {
                throw std::runtime_error("PopSift extraction stage closed during upload");
            }
            image = nullptr;
        }
        catch (...)
        {
            returnImage(_pipe._unused, image);
            job->setError(std::current_exception());
        }
    }
    _pipe._queue_stage2.close();
}

void PopSift::extractDownloadLoop( )
{
    Pipe& p = _pipe;
    std::exception_ptr startup_error;
    try
    {
        selectCudaDevice(_device);
#if defined(KFCORE_POPSIFT_TESTING)
        popsift::testing::throwIfWorkerFailureInjected(
            popsift::testing::WorkerFailureStage::ExtractStartup);
#endif
    }
    catch (...)
    {
        startup_error = std::current_exception();
        p._queue_stage1.close();
        p._queue_stage2.close();
    }

    SiftJob* job = nullptr;
    while (p._queue_stage2.pull(job))
    {
        popsift::ImageBase* image = job->getImg();
        if (startup_error != nullptr)
        {
            returnImage(p._unused, image);
            job->setError(startup_error);
            continue;
        }

        std::unique_ptr<popsift::FeaturesHost> features;
        try
        {
#if defined(KFCORE_POPSIFT_TESTING)
            popsift::testing::throwIfWorkerFailureInjected(
                popsift::testing::WorkerFailureStage::ExtractJob);
#endif
            applyConfiguration();
            private_init(image->getWidth(), image->getHeight());

            p._pyramid->step1(_config, image);
            returnImage(p._unused, image);
            p._pyramid->step2(_config);

            features.reset(p._pyramid->get_descriptors(_config));
            const cudaError_t sync_error = cudaDeviceSynchronize();
            POP_CUDA_FATAL_TEST(sync_error, "Cannot synchronize CUDA SIFT extraction");

            if (_config.getLogMode() == popsift::Config::All)
            {
                p._pyramid->download_and_save_array("pyramid");
                p._pyramid->save_descriptors(_config, features.get(), "pyramid");
            }

            job->setFeatures(features.release());
        }
        catch (...)
        {
            returnImage(p._unused, image);
            job->setError(std::current_exception());
        }
    }

    private_uninit();
}

void PopSift::matchPrepareLoop( )
{
    Pipe& p = _pipe;
    std::exception_ptr startup_error;
    try
    {
        selectCudaDevice(_device);
#if defined(KFCORE_POPSIFT_TESTING)
        popsift::testing::throwIfWorkerFailureInjected(
            popsift::testing::WorkerFailureStage::ExtractStartup);
#endif
    }
    catch (...)
    {
        startup_error = std::current_exception();
        p._queue_stage1.close();
        p._queue_stage2.close();
    }

    SiftJob* job = nullptr;
    while (p._queue_stage2.pull(job))
    {
        popsift::ImageBase* image = job->getImg();
        if (startup_error != nullptr)
        {
            returnImage(p._unused, image);
            job->setError(startup_error);
            continue;
        }

        std::unique_ptr<popsift::FeaturesDev> features;
        try
        {
#if defined(KFCORE_POPSIFT_TESTING)
            popsift::testing::throwIfWorkerFailureInjected(
                popsift::testing::WorkerFailureStage::ExtractJob);
#endif
            applyConfiguration();
            private_init(image->getWidth(), image->getHeight());
            p._pyramid->step1(_config, image);
            returnImage(p._unused, image);
            p._pyramid->step2(_config);

            features.reset(p._pyramid->clone_device_descriptors(_config));
            const cudaError_t sync_error = cudaDeviceSynchronize();
            POP_CUDA_FATAL_TEST(sync_error, "Cannot synchronize CUDA SIFT descriptor cloning");

            job->setFeatures(features.release());
        }
        catch (...)
        {
            returnImage(p._unused, image);
            job->setError(std::current_exception());
        }
    }

    private_uninit();
}

SiftJob::SiftJob( int w, int h, const unsigned char* imageData )
    : _w(w)
    , _h(h)
    , _imageData(checkedImageBytes(w, h, sizeof(unsigned char), imageData))
    , _img(nullptr)
{
    _f = _p.get_future();
    memcpy(_imageData.data(), imageData, _imageData.size());
}

SiftJob::SiftJob( int w, int h, const float* imageData )
    : _w(w)
    , _h(h)
    , _imageData(checkedImageBytes(w, h, sizeof(float), imageData))
    , _img(nullptr)
{
    _f = _p.get_future();
    memcpy(_imageData.data(), imageData, _imageData.size());
}

SiftJob::~SiftJob( ) = default;

void SiftJob::setImg( popsift::ImageBase* img )
{
    img->resetDimensions( _w, _h );
    img->load( _imageData.data() );
    _img = img;
}

popsift::ImageBase* SiftJob::getImg()
{
    return _img;
}

void SiftJob::setFeatures( popsift::FeaturesBase* f )
{
    _p.set_value( f );
}

popsift::FeaturesHost* SiftJob::get()
{
    return getHost();
}

popsift::FeaturesBase* SiftJob::getBase()
{
    return _f.get();
}

popsift::FeaturesHost* SiftJob::getHost()
{
    return dynamic_cast<popsift::FeaturesHost*>( _f.get() );
}

popsift::FeaturesDev* SiftJob::getDev()
{
    return dynamic_cast<popsift::FeaturesDev*>(_f.get());
}

void SiftJob::setError(std::exception_ptr ptr)
{
    if (ptr == nullptr)
    {
        ptr = std::make_exception_ptr(
            std::runtime_error("PopSift worker reported an unspecified failure"));
    }
    _p.set_exception(ptr);
}

void PopSift::Pipe::uninit() noexcept
{
    _queue_stage1.close();
    if (_thread_stage1 != nullptr)
    {
        if (_thread_stage1->joinable())
        {
            _thread_stage1->join();
        }
        _thread_stage1.reset(nullptr);
    }
    else
    {
        _queue_stage2.close();
    }

    if (_thread_stage2 != nullptr)
    {
        if (_thread_stage2->joinable())
        {
            _thread_stage2->join();
        }
        _thread_stage2.reset(nullptr);
    }

    _queue_stage2.close();
    _unused.close();
    popsift::ImageBase* image = nullptr;
    while (_unused.pull(image))
    {
        delete image;
    }
}
