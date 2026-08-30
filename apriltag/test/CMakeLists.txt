set(_apriltag_test_folder "apriltag/tests")

add_library(getline OBJECT getline.c)
cmake_config_target(getline NO_INSTALL FOLDER "${_apriltag_test_folder}")

cmake_add_test(
    NAME test_detection
    NO_TEST
    SOURCES test_detection.c
    LIBS KFCore::apriltag getline
    FOLDER "${_apriltag_test_folder}")

cmake_add_test(
    NAME test_tag_pose_estimation
    NO_TEST
    SOURCES test_tag_pose_estimation.c
    LIBS KFCore::apriltag getline
    FOLDER "${_apriltag_test_folder}")

# test images with true detection
set(TEST_IMAGE_NAMES
    "33369213973_9d9bb4cc96_c"
    "34085369442_304b6bafd9_c"
    "34139872896_defdb2f8d9_c"
)

foreach(IMG IN LISTS TEST_IMAGE_NAMES)
    add_test(
        NAME test_detection_${IMG}
        COMMAND $<TARGET_FILE:test_detection> data/${IMG}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# Add pose estimation tests for each image
foreach(IMG IN LISTS TEST_IMAGE_NAMES)
    add_test(
        NAME test_tag_pose_estimation_${IMG}
        COMMAND $<TARGET_FILE:test_tag_pose_estimation>
                data/${IMG}.jpg data/${IMG}.txt
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# Quick decode test
file(GLOB COMMON_SRC "${PROJECT_SOURCE_DIR}/common/*.c")
file(GLOB TAG_FILES "${PROJECT_SOURCE_DIR}/tag*.c")

set(_test_quick_decode_libs TurboUtils::Core)
if(UNIX)
    list(APPEND _test_quick_decode_libs m)
endif()

cmake_add_test(
    NAME test_quick_decode
    SOURCES
        test_quick_decode.c
        "${PROJECT_SOURCE_DIR}/apriltag_quad_thresh.c"
        ${COMMON_SRC}
        ${TAG_FILES}
    LIBS ${_test_quick_decode_libs}
    INCLUDES "${PROJECT_SOURCE_DIR}"
    FOLDER "${_apriltag_test_folder}")

