###############################################################################
# - Find or fetch ROPTLITE and add it as a library target
###############################################################################


# Try to find an existing ROPTLITE_INCLUDE_DIR (with Problems/Problem.h as a marker)
find_path(ROPTLITE_INCLUDE_DIR
    Problems/Problem.h
    HINTS ${ROPTLITE_INCLUDE_DIR}
)

if(NOT ROPTLITE_INCLUDE_DIR)
include(FetchContent)
    # Download ROPTLITE if not found
    FetchContent_Declare(
        ROPTLITE
        GIT_REPOSITORY https://github.com/kluophysics/ROPTLITE.git
        GIT_TAG "work"
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        # URL https://github.com/kluophysics/ROPTLITE/archive/refs/tags/v1.0.1.tar.gz
    )
    FetchContent_MakeAvailable(ROPTLITE)
    set(ROPTLITE_INCLUDE_DIR 
    ${ROPTLITE_SOURCE_DIR}
    # ${ROPTLITE_SOURCE_DIR}/src
    # ${ROPTLITE_SOURCE_DIR}/src/Manifolds
    # ${ROPTLITE_SOURCE_DIR}/src/Others
    # ${ROPTLITE_SOURCE_DIR}/src/Others/SparseBLAS
    # ${ROPTLITE_SOURCE_DIR}/src/Others/fftw
    # ${ROPTLITE_SOURCE_DIR}/src/Others/wavelet
    # ${ROPTLITE_SOURCE_DIR}/src/Problems
    # ${ROPTLITE_SOURCE_DIR}/src/Solvers
    # ${ROPTLITE_SOURCE_DIR}/src/cwrapper
    # ${ROPTLITE_SOURCE_DIR}/src/cwrapper/blas
    # ${ROPTLITE_SOURCE_DIR}/src/cwrapper/lapack
    )
endif()

# Handle the QUIET and REQUIRED arguments and set ROPTLITE_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROPTLITE DEFAULT_MSG ROPTLITE_INCLUDE_DIR)


message(STATUS "Found ROPTLITE: ${ROPTLITE_INCLUDE_DIR}")

# Mark as advanced for cache
mark_as_advanced(ROPTLITE_INCLUDE_DIR)

