cmake_minimum_required(VERSION 3.16)
project(apriltag VERSION 3.4.5 LANGUAGES C)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

if(NOT TARGET Salts::Core)
    find_package(Salts CONFIG REQUIRED
                 PATHS "$ENV{SALTS_ROOT}" NO_DEFAULT_PATH)
endif()

# Set a default build type if none was specified
set(default_build_type "Release")

if(WIN32)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE  STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS  "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra)
    add_compile_options(-Wpedantic)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -Wno-gnu-zero-variadic-macro-arguments
            -Wno-strict-prototypes
            -Wno-static-in-inline
        )
    endif()
    add_compile_options(-Wno-shift-negative-value)
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
      set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
    else()
      add_compile_options(-Werror)
    endif()
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang" AND CMAKE_C_SIMULATE_ID MATCHES "MSVC")
    # error: 'strdup' is deprecated: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name: _strdup.
    # "strdup" is standard since C23
    add_definitions(-D _CRT_NONSTDC_NO_DEPRECATE)
    # ignore "'fopen' is deprecated" and "'strncpy' is deprecated" warnings
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
endif()

aux_source_directory(common COMMON_SRC)
set(APRILTAG_SRCS apriltag.c apriltag_pose.c apriltag_quad_thresh.c)

if(TARGET miniblas)
    list(APPEND APRILTAG_SRCS apriltag_pose_miniblas.c)
    set(APRILTAG_WITH_MINIBLAS_HELPERS ON)
else()
    set(APRILTAG_WITH_MINIBLAS_HELPERS OFF)
endif()

# Library
file(GLOB TAG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tag*.c)
add_library(${PROJECT_NAME} SHARED ${APRILTAG_SRCS} ${COMMON_SRC} ${TAG_FILES})

if(CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT APPLE AND NOT CMAKE_C_SIMULATE_ID MATCHES "MSVC")
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-z,relro,-z,now,-z,defs")
endif()

if (MSVC)
    add_compile_definitions("_CRT_SECURE_NO_WARNINGS")
else()
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads)
endif()

if (UNIX)
    target_link_libraries(${PROJECT_NAME} PUBLIC m)
endif()

if(APRILTAG_WITH_MINIBLAS_HELPERS)
    target_link_libraries(${PROJECT_NAME} PUBLIC miniblas)
    target_compile_definitions(${PROJECT_NAME} PUBLIC APRILTAG_HAVE_MINIBLAS=1)
endif()

find_package(libjpeg-turbo CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC
    $<IF:$<TARGET_EXISTS:libjpeg-turbo::turbojpeg>,libjpeg-turbo::turbojpeg,libjpeg-turbo::turbojpeg-static>)
target_compile_definitions(${PROJECT_NAME} PUBLIC APRILTAG_USE_LIBJPEG_TURBO=1)

target_link_libraries(${PROJECT_NAME} PUBLIC Salts::Core)

set_target_properties(${PROJECT_NAME} PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    DEBUG_POSTFIX "d"
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(${PROJECT_NAME}
    ALIAS KFCore::apriltag
    NO_INSTALL
    FOLDER "apriltag"
    VERSION ${PROJECT_VERSION}
    SOVERSION 3)


include(GNUInstallDirs)
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/apriltag")

# Examples
if(BUILD_EXAMPLES)
    # apriltag_demo
    add_executable(apriltag_demo example/apriltag_demo.c)
    target_link_libraries(apriltag_demo PRIVATE KFCore::apriltag)
    cmake_config_target(apriltag_demo
        NO_INSTALL FOLDER "apriltag/examples")

    # apriltag_print_tag
    find_path(STB_IMAGE_WRITE_INCLUDE_DIR NAMES stb_image_write.h REQUIRED)
    add_executable(apriltag_print_tag example/apriltag_print_tag.c)
    target_include_directories(apriltag_print_tag PRIVATE ${STB_IMAGE_WRITE_INCLUDE_DIR})
    target_link_libraries(apriltag_print_tag PRIVATE KFCore::apriltag)
    cmake_config_target(apriltag_print_tag
        NO_INSTALL FOLDER "apriltag/examples")

    # opencv_demo
    set(_OpenCV_REQUIRED_COMPONENTS core imgproc videoio highgui)
    find_package(OpenCV COMPONENTS ${_OpenCV_REQUIRED_COMPONENTS} QUIET CONFIG)
    if(OpenCV_FOUND)
        enable_language(CXX)
        # NB: contrib required for TickMeter in OpenCV 2.4. This is only required for 16.04 backwards compatibility and can be removed in the future.
        #     If we add it to the find_package initially, the demo won't build for newer OpenCV versions
        if(OpenCV_VERSION VERSION_LESS "3.0.0")
            list(APPEND _OpenCV_REQUIRED_COMPONENTS contrib)
            find_package(OpenCV COMPONENTS ${_OpenCV_REQUIRED_COMPONENTS} CONFIG)
        endif()

        add_executable(opencv_demo example/opencv_demo.cc)
        target_link_libraries(opencv_demo PRIVATE KFCore::apriltag ${OpenCV_LIBRARIES})
        set_target_properties(opencv_demo PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF)
        cmake_config_target(opencv_demo
            NO_INSTALL FOLDER "apriltag/examples")
    else()
        message(STATUS "OpenCV not found: Not building demo")
    endif(OpenCV_FOUND)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
