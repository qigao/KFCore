# CMakeLists.txt for the KFCore Kalman filter unit tests.
#
# The test executable links against the `kfcore` library (via the KFCore::kfcore
# alias) and the TurboNet::TinyTest framework that provides the tinytest.h
# header and test runner.

if(BUILD_TESTING)
    add_executable(kfcore_kalman_tests test.c)

    target_link_libraries(kfcore_kalman_tests
        PRIVATE
            KFCore::kfcore
            TurboNet::TinyTest)

    # The test source includes linalg.h / navtoolbox.h directly, so make the
    # vendored miniblas headers reachable as well.
    target_link_libraries(kfcore_kalman_tests PRIVATE miniblas)

    if(TARGET apriltag)
        target_link_libraries(kfcore_kalman_tests PRIVATE apriltag)
    endif()

    add_test(NAME kfcore_kalman_tests COMMAND kfcore_kalman_tests)

    set_target_properties(kfcore_kalman_tests PROPERTIES
        FOLDER "kalman/tests")
endif()
