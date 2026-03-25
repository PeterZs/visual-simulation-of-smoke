#include "visual-simulation-of-smoke.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

    dim3 make_grid(const int nx, const int ny, const int nz, const dim3& block) {
        return dim3(static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    }

    __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    }

    __global__ void source_cells_kernel(float* density, float* temperature, const int nx, const int ny, const int nz, const float center_x, const float center_y, const float center_z, const float radius, const float density_amount, const float temperature_amount) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const float dx      = (static_cast<float>(x) + 0.5f) - center_x;
        const float dy      = (static_cast<float>(y) + 0.5f) - center_y;
        const float dz      = (static_cast<float>(z) + 0.5f) - center_z;
        const float radius2 = radius * radius;
        const float dist2   = dx * dx + dy * dy + dz * dz;
        if (dist2 > radius2) return;

        const auto index   = index_3d(x, y, z, nx, ny);
        const float weight = fmaxf(0.0f, 1.0f - dist2 / radius2);
        density[index] += density_amount * weight;
        temperature[index] += temperature_amount * weight;
    }

    __global__ void source_u_kernel(float* velocity_x, const int nx, const int ny, const int nz, const float center_x, const float center_y, const float center_z, const float radius, const float amount) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x > nx || y >= ny || z >= nz) return;

        const float dx      = static_cast<float>(x) - center_x;
        const float dy      = (static_cast<float>(y) + 0.5f) - center_y;
        const float dz      = (static_cast<float>(z) + 0.5f) - center_z;
        const float radius2 = radius * radius;
        const float dist2   = dx * dx + dy * dy + dz * dz;
        if (dist2 > radius2) return;
        velocity_x[index_3d(x, y, z, nx + 1, ny)] += amount * fmaxf(0.0f, 1.0f - dist2 / radius2);
    }

    __global__ void source_v_kernel(float* velocity_y, const int nx, const int ny, const int nz, const float center_x, const float center_y, const float center_z, const float radius, const float amount) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y > ny || z >= nz) return;

        const float dx      = (static_cast<float>(x) + 0.5f) - center_x;
        const float dy      = static_cast<float>(y) - center_y;
        const float dz      = (static_cast<float>(z) + 0.5f) - center_z;
        const float radius2 = radius * radius;
        const float dist2   = dx * dx + dy * dy + dz * dz;
        if (dist2 > radius2) return;
        velocity_y[index_3d(x, y, z, nx, ny + 1)] += amount * fmaxf(0.0f, 1.0f - dist2 / radius2);
    }

    __global__ void source_w_kernel(float* velocity_z, const int nx, const int ny, const int nz, const float center_x, const float center_y, const float center_z, const float radius, const float amount) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z > nz) return;

        const float dx      = (static_cast<float>(x) + 0.5f) - center_x;
        const float dy      = (static_cast<float>(y) + 0.5f) - center_y;
        const float dz      = static_cast<float>(z) - center_z;
        const float radius2 = radius * radius;
        const float dist2   = dx * dx + dy * dy + dz * dz;
        if (dist2 > radius2) return;
        velocity_z[index_3d(x, y, z, nx, ny)] += amount * fmaxf(0.0f, 1.0f - dist2 / radius2);
    }

} // namespace

int main() {
    auto cuda_ok = [](const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return true;
        std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
        return false;
    };
    auto smoke_ok = [](const int32_t code, const char* what) {
        if (code == 0) return true;
        std::cerr << what << " failed (" << code << ")\n";
        return false;
    };

    constexpr int32_t nx                   = 48;
    constexpr int32_t ny                   = 72;
    constexpr int32_t nz                   = 48;
    constexpr float cell_size              = 1.0f;
    constexpr float dt                     = 1.0f / 90.0f;
    constexpr float ambient_temperature    = 0.0f;
    constexpr float density_buoyancy       = 0.045f;
    constexpr float temperature_buoyancy   = 0.12f;
    constexpr float vorticity_epsilon      = 2.0f;
    constexpr int32_t pressure_iterations  = 80;
    constexpr int32_t block_x              = 8;
    constexpr int32_t block_y              = 8;
    constexpr int32_t block_z              = 4;
    constexpr uint32_t use_monotonic_cubic = 1u;
    constexpr int32_t frames               = 16;

    const uint64_t scalar_bytes     = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_x_bytes = static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_y_bytes = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny + 1) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_z_bytes = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz + 1) * sizeof(float);
    const std::size_t scalar_count  = static_cast<std::size_t>(scalar_bytes / sizeof(float));

    float* density                        = nullptr;
    float* temperature                    = nullptr;
    float* velocity_x                     = nullptr;
    float* velocity_y                     = nullptr;
    float* velocity_z                     = nullptr;
    float* temporary_previous_density     = nullptr;
    float* temporary_previous_temperature = nullptr;
    float* temporary_previous_velocity_x  = nullptr;
    float* temporary_previous_velocity_y  = nullptr;
    float* temporary_previous_velocity_z  = nullptr;
    float* temporary_pressure             = nullptr;
    float* temporary_divergence           = nullptr;
    float* temporary_omega_x              = nullptr;
    float* temporary_omega_y              = nullptr;
    float* temporary_omega_z              = nullptr;
    float* temporary_omega_magnitude      = nullptr;
    float* temporary_force_x              = nullptr;
    float* temporary_force_y              = nullptr;
    float* temporary_force_z              = nullptr;
    cudaStream_t stream                   = nullptr;
    int exit_code                         = EXIT_SUCCESS;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density), scalar_bytes), "cudaMalloc density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temperature), scalar_bytes), "cudaMalloc temperature")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_x), velocity_x_bytes), "cudaMalloc velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_y), velocity_y_bytes), "cudaMalloc velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_z), velocity_z_bytes), "cudaMalloc velocity_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_density), scalar_bytes), "cudaMalloc temporary_previous_density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_temperature), scalar_bytes), "cudaMalloc temporary_previous_temperature")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_x), velocity_x_bytes), "cudaMalloc temporary_previous_velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_y), velocity_y_bytes), "cudaMalloc temporary_previous_velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_z), velocity_z_bytes), "cudaMalloc temporary_previous_velocity_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_pressure), scalar_bytes), "cudaMalloc temporary_pressure")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_divergence), scalar_bytes), "cudaMalloc temporary_divergence")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_x), scalar_bytes), "cudaMalloc temporary_omega_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_y), scalar_bytes), "cudaMalloc temporary_omega_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_z), scalar_bytes), "cudaMalloc temporary_omega_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_magnitude), scalar_bytes), "cudaMalloc temporary_omega_magnitude")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_x), scalar_bytes), "cudaMalloc temporary_force_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_y), scalar_bytes), "cudaMalloc temporary_force_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_z), scalar_bytes), "cudaMalloc temporary_force_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) exit_code = EXIT_FAILURE;

    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(density, 0, scalar_bytes, stream), "cudaMemsetAsync density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(temperature, 0, scalar_bytes, stream), "cudaMemsetAsync temperature")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_x, 0, velocity_x_bytes, stream), "cudaMemsetAsync velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_y, 0, velocity_y_bytes, stream), "cudaMemsetAsync velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_z, 0, velocity_z_bytes, stream), "cudaMemsetAsync velocity_z")) exit_code = EXIT_FAILURE;

    const auto cuda_begin = std::chrono::steady_clock::now();
    for (int frame = 0; exit_code == EXIT_SUCCESS && frame < frames; ++frame) {
        const dim3 block{static_cast<unsigned>(block_x), static_cast<unsigned>(block_y), static_cast<unsigned>(block_z)};
        source_cells_kernel<<<make_grid(nx, ny, nz, block), block, 0, stream>>>(density, temperature, nx, ny, nz, static_cast<float>(nx) * 0.5f, static_cast<float>(ny) * 0.18f, static_cast<float>(nz) * 0.5f, 4.5f, 0.85f, 1.35f);
        source_u_kernel<<<make_grid(nx + 1, ny, nz, block), block, 0, stream>>>(velocity_x, nx, ny, nz, static_cast<float>(nx) * 0.5f, static_cast<float>(ny) * 0.18f, static_cast<float>(nz) * 0.5f, 4.5f, 0.0f);
        source_v_kernel<<<make_grid(nx, ny + 1, nz, block), block, 0, stream>>>(velocity_y, nx, ny, nz, static_cast<float>(nx) * 0.5f, static_cast<float>(ny) * 0.18f, static_cast<float>(nz) * 0.5f, 4.5f, 1.2f);
        source_w_kernel<<<make_grid(nx, ny, nz + 1, block), block, 0, stream>>>(velocity_z, nx, ny, nz, static_cast<float>(nx) * 0.5f, static_cast<float>(ny) * 0.18f, static_cast<float>(nz) * 0.5f, 4.5f, 0.0f);
        if (!cuda_ok(cudaGetLastError(), "source kernels")) exit_code = EXIT_FAILURE;

        VisualSimulationOfSmokeStepDesc cuda_desc{};
        cuda_desc.struct_size                    = sizeof(VisualSimulationOfSmokeStepDesc);
        cuda_desc.api_version                    = VISUAL_SIMULATION_OF_SMOKE_API_VERSION;
        cuda_desc.nx                             = nx;
        cuda_desc.ny                             = ny;
        cuda_desc.nz                             = nz;
        cuda_desc.cell_size                      = cell_size;
        cuda_desc.dt                             = dt;
        cuda_desc.ambient_temperature            = ambient_temperature;
        cuda_desc.density_buoyancy               = density_buoyancy;
        cuda_desc.temperature_buoyancy           = temperature_buoyancy;
        cuda_desc.vorticity_epsilon              = vorticity_epsilon;
        cuda_desc.pressure_iterations            = pressure_iterations;
        cuda_desc.use_monotonic_cubic            = use_monotonic_cubic;
        cuda_desc.density                        = density;
        cuda_desc.temperature                    = temperature;
        cuda_desc.velocity_x                     = velocity_x;
        cuda_desc.velocity_y                     = velocity_y;
        cuda_desc.velocity_z                     = velocity_z;
        cuda_desc.temporary_previous_density     = temporary_previous_density;
        cuda_desc.temporary_previous_temperature = temporary_previous_temperature;
        cuda_desc.temporary_previous_velocity_x  = temporary_previous_velocity_x;
        cuda_desc.temporary_previous_velocity_y  = temporary_previous_velocity_y;
        cuda_desc.temporary_previous_velocity_z  = temporary_previous_velocity_z;
        cuda_desc.temporary_pressure             = temporary_pressure;
        cuda_desc.temporary_divergence           = temporary_divergence;
        cuda_desc.temporary_omega_x              = temporary_omega_x;
        cuda_desc.temporary_omega_y              = temporary_omega_y;
        cuda_desc.temporary_omega_z              = temporary_omega_z;
        cuda_desc.temporary_omega_magnitude      = temporary_omega_magnitude;
        cuda_desc.temporary_force_x              = temporary_force_x;
        cuda_desc.temporary_force_y              = temporary_force_y;
        cuda_desc.temporary_force_z              = temporary_force_z;
        cuda_desc.block_x                        = block_x;
        cuda_desc.block_y                        = block_y;
        cuda_desc.block_z                        = block_z;
        cuda_desc.stream                         = stream;
        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_validate_desc(&cuda_desc), "visual_simulation_of_smoke_validate_desc")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_step_cuda(&cuda_desc), "visual_simulation_of_smoke_step_cuda")) exit_code = EXIT_FAILURE;
    }
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) exit_code = EXIT_FAILURE;
    const auto cuda_end = std::chrono::steady_clock::now();

    std::vector<float> host_density(scalar_count, 0.0f);
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemcpy(host_density.data(), density, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy density")) exit_code = EXIT_FAILURE;

    const float cuda_total_density = exit_code == EXIT_SUCCESS ? std::accumulate(host_density.begin(), host_density.end(), 0.0f) : 0.0f;
    const float cuda_peak_density  = exit_code == EXIT_SUCCESS && !host_density.empty() ? *std::max_element(host_density.begin(), host_density.end()) : 0.0f;

    cudaStreamDestroy(stream);
    cudaFree(density);
    cudaFree(temperature);
    cudaFree(velocity_x);
    cudaFree(velocity_y);
    cudaFree(velocity_z);
    cudaFree(temporary_previous_density);
    cudaFree(temporary_previous_temperature);
    cudaFree(temporary_previous_velocity_x);
    cudaFree(temporary_previous_velocity_y);
    cudaFree(temporary_previous_velocity_z);
    cudaFree(temporary_pressure);
    cudaFree(temporary_divergence);
    cudaFree(temporary_omega_x);
    cudaFree(temporary_omega_y);
    cudaFree(temporary_omega_z);
    cudaFree(temporary_omega_magnitude);
    cudaFree(temporary_force_x);
    cudaFree(temporary_force_y);
    cudaFree(temporary_force_z);
    if (exit_code != EXIT_SUCCESS) return exit_code;

    const double cuda_ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_begin).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "visual-simulation-of-smoke benchmark\n";
    std::cout << "grid: " << nx << " x " << ny << " x " << nz << '\n';
    std::cout << "frames: " << frames << '\n';
    std::cout << "| metric | cuda |\n";
    std::cout << "|---|---:|\n";
    std::cout << "| total_ms | " << cuda_ms << " |\n";
    std::cout << "| step_ms | " << cuda_ms / static_cast<double>(frames) << " |\n";
    std::cout << "| total_density | " << cuda_total_density << " |\n";
    std::cout << "| peak_density | " << cuda_peak_density << " |\n";
    return EXIT_SUCCESS;
}
