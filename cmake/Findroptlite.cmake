###############################################################################
# - Find or fetch roptlite and add it as a library target
###############################################################################

include(FetchContent)

# Try to find an existing ROPTLITE_DIR (with include/roptlite/version.h)
find_path(ROPTLITE_DIR
    include/roptlite/version.h
    HINTS ${ROPTLITE_DIR}
)

if(NOT ROPTLITE_DIR)
    # Download roptlite if not found
    FetchContent_Declare(
        roptlite
        URL https://github.com/kluophysics/roptlite/archive/refs/tags/v1.0.0.tar.gz
    )
    FetchContent_MakeAvailable(roptlite)
    set(ROPTLITE_DIR ${roptlite_SOURCE_DIR})
endif()

# Handle the QUIET and REQUIRED arguments and set ROPTLITE_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(roptlite DEFAULT_MSG ROPTLITE_DIR)


# Mark as advanced for cache
mark_as_advanced(ROPTLITE_DIR)

# # Add the roptlite library target if not already present
# if(NOT TARGET roptlite)
#     add_subdirectory(${ROPTLITE_DIR} ${CMAKE_BINARY_DIR}/roptlib-build)
# endif()

