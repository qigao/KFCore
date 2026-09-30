# CMakeLists.txt for the KFCore Kalman filter library sources.
#
# This subdirectory builds the `kfcore_kalman` target that contains every Kalman
# filter implementation (Takasu, UDU, EKF, UKF) together with the supporting
# linear-algebra and mini-BLAS helpers. The target is consumed by the
# top-level project and by test executables through the KFCore::kalman alias.

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

add_library(kfcore_kalman STATIC)

target_sources(kfcore_kalman
    PRIVATE ${KFCORE_SOURCES}
            ${KFCORE_PUBLIC_HEADERS})

target_include_directories(kfcore_kalman
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kfcore>)

target_compile_features(kfcore_kalman PUBLIC c_std_11)

target_link_libraries(kfcore_kalman PRIVATE KFCore::miniblas_internal)

list(APPEND KFCORE_PUBLIC_HEADERS
        navigation_tools/frame_transform.h
        navigation_tools/navtoolbox.h
        navigation_tools/nav_fusion2d.h
        navigation_tools/nav_fusion3d.h)
target_sources(kfcore_kalman
        PRIVATE navigation_tools/frame_transform.c
                navigation_tools/frame_transform.h
                navigation_tools/navtoolbox.c
                navigation_tools/navtoolbox.h
                navigation_tools/nav_fusion2d.c
                navigation_tools/nav_fusion2d.h
                navigation_tools/nav_fusion3d.c
                navigation_tools/nav_fusion3d.h)
target_include_directories(kfcore_kalman
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/navigation_tools>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kfcore/navigation_tools>)

target_compile_options(kfcore_kalman
    PRIVATE
        $<$<C_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wdouble-promotion>)

if(NOT WIN32)
    target_link_libraries(kfcore_kalman PUBLIC m)
endif()

cmake_config_target(kfcore_kalman
    ALIAS KFCore::kalman
    EXPORT_NAME kalman
    EXPORT_SET KFCoreTargets
    FOLDER "kalman")

set(KFCORE_PUBLIC_HEADERS ${KFCORE_PUBLIC_HEADERS} PARENT_SCOPE)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
