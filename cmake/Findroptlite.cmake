###############################################################################
# - Find or fetch roptlite and add it as a library target
###############################################################################


# Try to find an existing ROPTLITE_INCLUDE_DIR (with Problems/Problem.h as a marker)
find_path(ROPTLITE_INCLUDE_DIR
    src/Problems/Problem.h
    HINTS ${ROPTLITE_INCLUDE_DIR}
    HINTS ${roptlite_INCLUDE_DIR}

)

if(NOT ROPTLITE_INCLUDE_DIR)
include(FetchContent)
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
    set(ROPTLITE_INCLUDE_DIR 
    ${roptlite_SOURCE_DIR}
    ${roptlite_SOURCE_DIR}/src
    ${roptlite_SOURCE_DIR}/src/Manifolds
    ${roptlite_SOURCE_DIR}/src/Others
    ${roptlite_SOURCE_DIR}/src/Others/SparseBLAS
    ${roptlite_SOURCE_DIR}/src/Others/fftw
    ${roptlite_SOURCE_DIR}/src/Others/wavelet
    ${roptlite_SOURCE_DIR}/src/Problems
    ${roptlite_SOURCE_DIR}/src/Solvers
    ${roptlite_SOURCE_DIR}/src/cwrapper
    ${roptlite_SOURCE_DIR}/src/cwrapper/blas
    ${roptlite_SOURCE_DIR}/src/cwrapper/lapack
    )
endif()

# Handle the QUIET and REQUIRED arguments and set ROPTLITE_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(roptlite DEFAULT_MSG ROPTLITE_INCLUDE_DIR)


message(STATUS "Found roptlite: ${ROPTLITE_INCLUDE_DIR}")

# Mark as advanced for cache
mark_as_advanced(ROPTLITE_INCLUDE_DIR)

