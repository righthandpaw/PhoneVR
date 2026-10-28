cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -g")

set(ANDROID_STL c++_static)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

# AprilTag (BSD-2, vendored in src/main/cpp/third_party/apriltag): fiducial marker detection for
# the experimental marker tracking spike. Plain C, built as a static library.
set(apriltag_dir src/main/cpp/third_party/apriltag)
add_library(apriltag STATIC
    ${apriltag_dir}/apriltag.c
    ${apriltag_dir}/apriltag_quad_thresh.c
    ${apriltag_dir}/apriltag_pose.c
    ${apriltag_dir}/tag36h11.c
    ${apriltag_dir}/common/g2d.c
    ${apriltag_dir}/common/homography.c
    ${apriltag_dir}/common/image_u8.c
    ${apriltag_dir}/common/image_u8_parallel.c
    ${apriltag_dir}/common/image_u8x3.c
    ${apriltag_dir}/common/image_u8x4.c
    ${apriltag_dir}/common/matd.c
    ${apriltag_dir}/common/pam.c
    ${apriltag_dir}/common/pnm.c
    ${apriltag_dir}/common/pthreads_cross.c
    ${apriltag_dir}/common/string_util.c
    ${apriltag_dir}/common/svd22.c
    ${apriltag_dir}/common/time_util.c
    ${apriltag_dir}/common/unionfind.c
    ${apriltag_dir}/common/workerpool.c
    ${apriltag_dir}/common/zarray.c
    ${apriltag_dir}/common/zhash.c
    ${apriltag_dir}/common/zmaxheap.c
)
target_include_directories(apriltag PUBLIC ${apriltag_dir})
target_compile_options(apriltag PRIVATE -O3 -w)
target_compile_definitions(apriltag PRIVATE NDEBUG)
set_target_properties(apriltag PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(native-lib-alvr SHARED
    src/main/cpp/alvr_main.cpp
)

# Import the ARCore (Google Play Services for AR) library.
add_library(arcore SHARED IMPORTED)
set_target_properties(arcore PROPERTIES IMPORTED_LOCATION
        ${ARCORE_LIBPATH}/${ANDROID_ABI}/libarcore_sdk_c.so
        INTERFACE_INCLUDE_DIRECTORIES ${ARCORE_INCLUDE}
)

set(libs_dir ${CMAKE_CURRENT_SOURCE_DIR}/libraries)
set(alvr_build_dir ${CMAKE_CURRENT_SOURCE_DIR}/../ALVR/build/alvr_client_core)

target_include_directories(native-lib-alvr
    # ALVR Headers
    PUBLIC ${alvr_build_dir}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../cardboard
    PUBLIC ${ARCORE_INCLUDE}
)

target_link_libraries(native-lib-alvr
    log android EGL GLESv3 mediandk camera2ndk arcore apriltag
    ${alvr_build_dir}/${ANDROID_ABI}/libalvr_client_core.so
    ${libs_dir}/jni/${ANDROID_ABI}/libGfxPluginCardboard.so
)
