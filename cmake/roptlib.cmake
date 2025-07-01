cmake_minimum_required(VERSION 3.5)

project(roptlib-download NONE)

include(ExternalProject)
ExternalProject_Add(roptlib
        GIT_REPOSITORY    https://github.com/kluophysics/roptlite/archive/refs/tags/v1.0.0.tar.gz
        GIT_TAG           master
        SOURCE_DIR        "${CMAKE_BINARY_DIR}/roptlib-src"
        BINARY_DIR        "${CMAKE_BINARY_DIR}/roptlib-build"
        # BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/../"

        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      "")