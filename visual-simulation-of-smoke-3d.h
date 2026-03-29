#ifndef VISUAL_SIMULATION_OF_SMOKE_3D_H
#define VISUAL_SIMULATION_OF_SMOKE_3D_H

#include <stdint.h>

#ifdef _WIN32
#ifdef VISUAL_SIMULATION_OF_SMOKE_BUILD_SHARED
#define SMOKE_SIMULATION_API __declspec(dllexport)
#else
#define SMOKE_SIMULATION_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define SMOKE_SIMULATION_API __attribute__((visibility("default")))
#else
#define SMOKE_SIMULATION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SmokeSimulationResult {
    SMOKE_SIMULATION_RESULT_OK              = 0,
    SMOKE_SIMULATION_RESULT_OUT_OF_MEMORY   = 1,
    SMOKE_SIMULATION_RESULT_BACKEND_FAILURE = 2,
} SmokeSimulationResult;

typedef enum SmokeSimulationBoundaryMode {
    SMOKE_SIMULATION_BOUNDARY_FIXED    = 0,
    SMOKE_SIMULATION_BOUNDARY_PERIODIC = 1,
} SmokeSimulationBoundaryMode;

typedef struct SmokeSimulationBoundaryConfig {
    SmokeSimulationBoundaryMode x;
    SmokeSimulationBoundaryMode y;
    SmokeSimulationBoundaryMode z;
} SmokeSimulationBoundaryConfig;

typedef enum SmokeSimulationScalarAdvectionMode {
    SMOKE_SIMULATION_SCALAR_ADVECTION_LINEAR           = 0,
    SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC = 1,
} SmokeSimulationScalarAdvectionMode;

typedef struct SmokeSimulationConfig {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    int32_t pressure_iterations;
    float pressure_tolerance;
    float ambient_temperature;
    float buoyancy_density_factor;
    float buoyancy_temperature_factor;
    float vorticity_confinement;
    SmokeSimulationScalarAdvectionMode scalar_advection_mode;
    SmokeSimulationBoundaryConfig boundary;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
} SmokeSimulationConfig;

typedef struct SmokeSimulationContext_t* SmokeSimulationContext;

typedef struct SmokeSimulationContextCreateDesc {
    SmokeSimulationConfig config;
    void* stream;
    float initial_density;
    float initial_temperature;
} SmokeSimulationContextCreateDesc;

typedef struct SmokeSimulationStepDesc {
    const float* density_source;
    const float* temperature_source;
    const float* force_x;
    const float* force_y;
    const float* force_z;
    const uint8_t* occupancy;
    const float* solid_velocity_x;
    const float* solid_velocity_y;
    const float* solid_velocity_z;
    const float* solid_temperature;
} SmokeSimulationStepDesc;

typedef enum SmokeSimulationExportKind {
    SMOKE_SIMULATION_EXPORT_DENSITY            = 0,
    SMOKE_SIMULATION_EXPORT_TEMPERATURE        = 1,
    SMOKE_SIMULATION_EXPORT_PRESSURE           = 2,
    SMOKE_SIMULATION_EXPORT_DIVERGENCE         = 3,
    SMOKE_SIMULATION_EXPORT_VELOCITY           = 4,
    SMOKE_SIMULATION_EXPORT_VELOCITY_MAGNITUDE = 5,
    SMOKE_SIMULATION_EXPORT_VORTICITY_MAGNITUDE = 6,
    SMOKE_SIMULATION_EXPORT_OCCUPANCY         = 7,
} SmokeSimulationExportKind;

typedef struct SmokeSimulationExportDesc {
    SmokeSimulationExportKind kind;
} SmokeSimulationExportDesc;

SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_create_context_cuda(const SmokeSimulationContextCreateDesc* desc, SmokeSimulationContext* out_context);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_destroy_context_cuda(SmokeSimulationContext context);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_step_cuda(SmokeSimulationContext context, const SmokeSimulationStepDesc* desc);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_export_cuda(SmokeSimulationContext context, const SmokeSimulationExportDesc* desc, void* destination);

#ifdef __cplusplus
}
#endif

#endif // VISUAL_SIMULATION_OF_SMOKE_3D_H
