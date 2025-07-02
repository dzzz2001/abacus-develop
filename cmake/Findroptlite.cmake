###############################################################################
# - Find or fetch roptlite and add it as a library target
###############################################################################

include(FetchContent)

# Try to find an existing ROPTLITE_DIR (with Problems/Problem.h as a marker)
find_path(ROPTLITE_DIR
    Problems/Problem.h
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/roptlite 
            ${CMAKE_CURRENT_SOURCE_DIR}/../_deps/
    HINTS ${ROPTLITE_DIR}
)

if(NOT ROPTLITE_DIR)
    # Download roptlite if not found
    FetchContent_Declare(
        roptlite
        GIT_REPOSITORY https://github.com/kluophysics/roptlite.git
        GIT_TAG "work"
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        # URL https://github.com/kluophysics/roptlite/archive/refs/tags/v1.0.1.tar.gz
    )
    FetchContent_MakeAvailable(roptlite)
    set(ROPTLITE_DIR ${roptlite_SOURCE_DIR})
endif()

# Set include directories for roptlite
set(roptlite_INCLUDE_DIRS
    ${ROPTLITE_DIR}
    ${ROPTLITE_DIR}/src
    ${ROPTLITE_DIR}/src/Manifolds
    ${ROPTLITE_DIR}/src/Others
    ${ROPTLITE_DIR}/src/Others/SparseBLAS
    ${ROPTLITE_DIR}/src/Others/fftw
    ${ROPTLITE_DIR}/src/Others/wavelet
    ${ROPTLITE_DIR}/src/Problems
    ${ROPTLITE_DIR}/src/Solvers
    ${ROPTLITE_DIR}/src/cwrapper
    ${ROPTLITE_DIR}/src/cwrapper/blas
    ${ROPTLITE_DIR}/src/cwrapper/lapack
    # ${ROPTLITE_DIR}/src/test
)

# Mark as advanced for cache
mark_as_advanced(ROPTLITE_DIR)

# Add the roptlite library target if not already present
# if(NOT TARGET roptlite)
#     add_subdirectory(${ROPTLITE_DIR} ${CMAKE_BINARY_DIR}/roptlite-build)
# endif()

# Handle the QUIET and REQUIRED arguments and set ROPTLITE_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(roptlite DEFAULT_MSG ROPTLITE_DIR)

# # Export variables for parent scope
# set(roptlite_INCLUDE_DIRS ${roptlite_INCLUDE_DIRS} PARENT_SCOPE)
# set(ROPTLITE_DIR ${ROPTLITE_DIR} PARENT_SCOPE)

message(STATUS "Found roptlite: ${ROPTLITE_DIR}")
message(STATUS "roptlite_INCLUDE_DIRS: ${roptlite_INCLUDE_DIRS}")

