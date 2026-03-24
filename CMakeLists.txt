cmake_minimum_required(VERSION 4.1.2)

project(visual-simulation-of-smoke VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Toolchain
# ============================================================================
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
endif ()

set(VISUAL_SMOKE_WITH_CUDA OFF)
if (CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    set(VISUAL_SMOKE_WITH_CUDA ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif ()

# ============================================================================
# Targets
# ============================================================================
set(visual_smoke_sources
        visual-simulation-of-smoke.h
        validate.cpp
        backend-parallel.cpp
)

if (VISUAL_SMOKE_WITH_CUDA)
    list(APPEND visual_smoke_sources backend-cuda.cu)
endif ()

add_library(
        visual-simulation-of-smoke
        SHARED
        ${visual_smoke_sources}
)

if (VISUAL_SMOKE_WITH_CUDA)
    add_executable(
            visual-simulation-of-smoke-app
            main.cu
    )
endif ()

target_compile_definitions(visual-simulation-of-smoke PRIVATE VISUAL_SIMULATION_OF_SMOKE_BUILD_SHARED)
if (NOT VISUAL_SMOKE_WITH_CUDA)
    target_compile_definitions(visual-simulation-of-smoke PRIVATE VISUAL_SIMULATION_OF_SMOKE_NO_CUDA)
endif ()
target_include_directories(visual-simulation-of-smoke PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (VISUAL_SMOKE_WITH_CUDA)
    target_link_libraries(visual-simulation-of-smoke PRIVATE CUDA::cudart)
    target_link_libraries(visual-simulation-of-smoke-app PRIVATE visual-simulation-of-smoke CUDA::cudart)
endif ()

# ============================================================================
# Compile Options
# ============================================================================
if (MSVC)
    target_compile_options(visual-simulation-of-smoke PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
    if (TARGET visual-simulation-of-smoke-app)
        target_compile_options(visual-simulation-of-smoke-app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
    endif ()
else ()
    target_compile_options(visual-simulation-of-smoke PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    if (TARGET visual-simulation-of-smoke-app)
        target_compile_options(visual-simulation-of-smoke-app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    endif ()
endif ()

if (VISUAL_SMOKE_WITH_CUDA)
    target_compile_options(
            visual-simulation-of-smoke
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
    )
    target_compile_options(
            visual-simulation-of-smoke-app
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
    )

    set_target_properties(
            visual-simulation-of-smoke
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif ()

# ============================================================================
# Output Layout
# ============================================================================
set(visual_smoke_targets visual-simulation-of-smoke)
if (TARGET visual-simulation-of-smoke-app)
    list(APPEND visual_smoke_targets visual-simulation-of-smoke-app)
endif ()

foreach (target IN LISTS visual_smoke_targets)
    set_target_properties(
            ${target}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach ()

# ============================================================================
# Runtime Lookup
# ============================================================================
if (UNIX)
    foreach (target IN LISTS visual_smoke_targets)
        set_target_properties(
                ${target}
                PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
        )
    endforeach ()
endif ()
