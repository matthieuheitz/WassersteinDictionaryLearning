# FindHalide.cmake
# ... shamelessly based on FindJeMalloc.cmake

find_path(HALIDE_ROOT_DIR
    NAMES include/Halide.h include/HalideRuntime.h
)

find_library(HALIDE_LIBRARIES
    NAMES Halide
    HINTS ${HALIDE_ROOT_DIR}/lib
)

find_path(HALIDE_INCLUDES
    NAMES Halide.h HalideRuntime.h
    HINTS ${HALIDE_ROOT_DIR}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Halide DEFAULT_MSG
    HALIDE_LIBRARIES
    HALIDE_INCLUDES
)

set(HALIDE_LIBRARY_DIR ${HALIDE_LIBRARIES})
set(HALIDE_INCLUDE_DIR ${HALIDE_INCLUDES})

mark_as_advanced(
    HALIDE_ROOT_DIR
    HALIDE_LIBRARIES
    HALIDE_LIBRARY_DIR
    HALIDE_INCLUDES
    HALIDE_INCLUDE_DIR
)
