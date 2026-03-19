#ifndef VISUAL_SIMULATION_OF_SMOKE_H
#define VISUAL_SIMULATION_OF_SMOKE_H

#include <stdint.h>

#ifdef _WIN32
#ifdef VISUAL_SIMULATION_OF_SMOKE_BUILD_SHARED
#define VISUAL_SIMULATION_OF_SMOKE_API __declspec(dllexport)
#else
#define VISUAL_SIMULATION_OF_SMOKE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define VISUAL_SIMULATION_OF_SMOKE_API __attribute__((visibility("default")))
#else
#define VISUAL_SIMULATION_OF_SMOKE_API
#endif

#define VISUAL_SIMULATION_OF_SMOKE_SUCCESS                 0
#define VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT  -1
#define VISUAL_SIMULATION_OF_SMOKE_ERROR_RUNTIME           -2
#define VISUAL_SIMULATION_OF_SMOKE_ERROR_ALLOCATION_FAILED -3

#ifdef __cplusplus
extern "C" {
#endif

VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_scalar_field_bytes(int32_t nx, int32_t ny, int32_t nz);
VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_velocity_x_bytes(int32_t nx, int32_t ny, int32_t nz);
VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_velocity_y_bytes(int32_t nx, int32_t ny, int32_t nz);
VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_velocity_z_bytes(int32_t nx, int32_t ny, int32_t nz);
VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_workspace_bytes(int32_t nx, int32_t ny, int32_t nz);

VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_clear_async(
    void* density,
    uint64_t density_bytes,
    void* temperature,
    uint64_t temperature_bytes,
    void* velocity_x,
    uint64_t velocity_x_bytes,
    void* velocity_y,
    uint64_t velocity_y_bytes,
    void* velocity_z,
    uint64_t velocity_z_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    void* cuda_stream);

VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_add_source_async(
    void* density,
    uint64_t density_bytes,
    void* temperature,
    uint64_t temperature_bytes,
    void* velocity_x,
    uint64_t velocity_x_bytes,
    void* velocity_y,
    uint64_t velocity_y_bytes,
    void* velocity_z,
    uint64_t velocity_z_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    float center_x,
    float center_y,
    float center_z,
    float radius,
    float density_amount,
    float temperature_amount,
    float velocity_source_x,
    float velocity_source_y,
    float velocity_source_z,
    int32_t block_x,
    int32_t block_y,
    int32_t block_z,
    void* cuda_stream);

VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_step_async(
    void* density,
    uint64_t density_bytes,
    void* temperature,
    uint64_t temperature_bytes,
    void* velocity_x,
    uint64_t velocity_x_bytes,
    void* velocity_y,
    uint64_t velocity_y_bytes,
    void* velocity_z,
    uint64_t velocity_z_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    void* workspace,
    uint64_t workspace_bytes,
    float dt,
    float ambient_temperature,
    float density_buoyancy,
    float temperature_buoyancy,
    float vorticity_epsilon,
    int32_t pressure_iterations,
    int32_t block_x,
    int32_t block_y,
    int32_t block_z,
    uint32_t use_monotonic_cubic,
    void* cuda_stream);

VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_compute_velocity_magnitude_async(
    void* velocity_x,
    uint64_t velocity_x_bytes,
    void* velocity_y,
    uint64_t velocity_y_bytes,
    void* velocity_z,
    uint64_t velocity_z_bytes,
    void* destination,
    uint64_t destination_bytes,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float cell_size,
    int32_t block_x,
    int32_t block_y,
    int32_t block_z,
    void* cuda_stream);

VISUAL_SIMULATION_OF_SMOKE_API const char* visual_simulation_of_smoke_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // VISUAL_SIMULATION_OF_SMOKE_H
