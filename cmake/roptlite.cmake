cmake_minimum_required(VERSION 3.5)

project(roptlite-download NONE)

include(ExternalProject)
ExternalProject_Add(roptlite
        GIT_REPOSITORY https://github.com/kluophysics/roptlite.git
        GIT_TAG "master"
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/roptlite-src"
        BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/roptlite-build"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      "")