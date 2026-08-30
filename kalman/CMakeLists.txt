# CMakeLists.txt for the KFCore Kalman filter library sources.
#
# This subdirectory builds the `kfcore` target that contains every Kalman
# filter implementation (Takasu, UDU, EKF, UKF) together with the supporting
# linear-algebra and mini-BLAS helpers.  The target is consumed by the
# top-level project and by the test executables through the KFCore::kfcore
# alias.

set(KFCORE_SOURCES
    kalman_takasu.c
    kalman_udu.c
    kalman_ekf.c
    kalman_ukf.c
    signal_filters.c)

set(KFCORE_PUBLIC_HEADERS
    kalman_takasu.h
    kalman_udu.h
    kalman_ekf.h
    kalman_ukf.h
    signal_filters.h)

add_library(kfcore SHARED)
add_library(KFCore::kfcore ALIAS kfcore)

target_sources(kfcore
    PRIVATE ${KFCORE_SOURCES}
            ${KFCORE_PUBLIC_HEADERS})

# Public include directory so consumers can do `#include <kalman_takasu.h>`.
target_include_directories(kfcore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kfcore>)

target_compile_features(kfcore PUBLIC c_std_11)

# Link the vendored mini-BLAS / linear-algebra library privately into kfcore.
# kfcore is a shared library, so installed consumers do not need the private
# miniblas archive or target.
target_link_libraries(kfcore PRIVATE miniblas)

# Navigation helper functions are part of the standard KFCore package.
list(APPEND KFCORE_PUBLIC_HEADERS
        navigation_tools/frame_transform.h
        navigation_tools/navtoolbox.h
        navigation_tools/nav_fusion2d.h
        navigation_tools/nav_fusion3d.h)
target_sources(kfcore
        PRIVATE navigation_tools/frame_transform.c
                navigation_tools/frame_transform.h
                navigation_tools/navtoolbox.c
                navigation_tools/navtoolbox.h
                navigation_tools/nav_fusion2d.c
                navigation_tools/nav_fusion2d.h
                navigation_tools/nav_fusion3d.c
                navigation_tools/nav_fusion3d.h)
target_include_directories(kfcore
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/navigation_tools>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kfcore/navigation_tools>)

target_compile_options(kfcore
    PRIVATE
        $<$<C_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wdouble-promotion>)

# Link the math library on platforms that expose it as a separate library.
if(NOT WIN32)
    target_link_libraries(kfcore PUBLIC m)
endif()

set_target_properties(kfcore PROPERTIES
    FOLDER "kalman"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Expose the list of public headers to the parent scope so the top-level
# project can reuse them for install rules and package config generation.
set(KFCORE_PUBLIC_HEADERS ${KFCORE_PUBLIC_HEADERS} PARENT_SCOPE)
add_subdirectory(tests)
