#include "visual-simulation-of-smoke.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <execution>
#include <numeric>
#include <vector>

extern "C" {

int32_t visual_simulation_of_smoke_step_parallel(const VisualSimulationOfSmokeStepDesc* desc) {
    if (desc == nullptr) return 1000;
    if (desc->struct_size < sizeof(VisualSimulationOfSmokeStepDesc)) return 1000;
    if (desc->stream != nullptr) return 3002;
    const int32_t nx = desc->nx;
    const int32_t ny = desc->ny;
    const int32_t nz = desc->nz;
    const float cell_size = desc->cell_size;
    const float dt = desc->dt;
    const float ambient_temperature = desc->ambient_temperature;
    const float density_buoyancy = desc->density_buoyancy;
    const float temperature_buoyancy = desc->temperature_buoyancy;
    const float vorticity_epsilon = desc->vorticity_epsilon;
    const int32_t pressure_iterations = desc->pressure_iterations;
    const bool cubic = desc->use_monotonic_cubic != 0u;
    if (nx <= 0 || ny <= 0 || nz <= 0) return 1001;
    if (cell_size <= 0.0f) return 1002;
    if (dt <= 0.0f) return 1003;
    if (pressure_iterations <= 0) return 1004;
    if (desc->density == nullptr) return 2001;
    if (desc->temperature == nullptr) return 2002;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_previous_density == nullptr) return 2007;
    if (desc->temporary_previous_temperature == nullptr) return 2008;
    if (desc->temporary_previous_velocity_x == nullptr) return 2009;
    if (desc->temporary_previous_velocity_y == nullptr) return 2010;
    if (desc->temporary_previous_velocity_z == nullptr) return 2011;
    if (desc->temporary_pressure == nullptr) return 2012;
    if (desc->temporary_divergence == nullptr) return 2013;
    if (desc->temporary_omega_x == nullptr) return 2014;
    if (desc->temporary_omega_y == nullptr) return 2015;
    if (desc->temporary_omega_z == nullptr) return 2016;
    if (desc->temporary_omega_magnitude == nullptr) return 2017;
    if (desc->temporary_force_x == nullptr) return 2018;
    if (desc->temporary_force_y == nullptr) return 2019;
    if (desc->temporary_force_z == nullptr) return 2020;

    auto* density = reinterpret_cast<float*>(desc->density);
    auto* temperature = reinterpret_cast<float*>(desc->temperature);
    auto* velocity_x = reinterpret_cast<float*>(desc->velocity_x);
    auto* velocity_y = reinterpret_cast<float*>(desc->velocity_y);
    auto* velocity_z = reinterpret_cast<float*>(desc->velocity_z);
    auto* previous_density = reinterpret_cast<float*>(desc->temporary_previous_density);
    auto* previous_temperature = reinterpret_cast<float*>(desc->temporary_previous_temperature);
    auto* previous_velocity_x = reinterpret_cast<float*>(desc->temporary_previous_velocity_x);
    auto* previous_velocity_y = reinterpret_cast<float*>(desc->temporary_previous_velocity_y);
    auto* previous_velocity_z = reinterpret_cast<float*>(desc->temporary_previous_velocity_z);
    auto* pressure = reinterpret_cast<float*>(desc->temporary_pressure);
    auto* divergence = reinterpret_cast<float*>(desc->temporary_divergence);
    auto* omega_x = reinterpret_cast<float*>(desc->temporary_omega_x);
    auto* omega_y = reinterpret_cast<float*>(desc->temporary_omega_y);
    auto* omega_z = reinterpret_cast<float*>(desc->temporary_omega_z);
    auto* omega_magnitude = reinterpret_cast<float*>(desc->temporary_omega_magnitude);
    auto* force_x = reinterpret_cast<float*>(desc->temporary_force_x);
    auto* force_y = reinterpret_cast<float*>(desc->temporary_force_y);
    auto* force_z = reinterpret_cast<float*>(desc->temporary_force_z);

    const std::uint64_t cell_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_x_field_bytes = static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_y_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_z_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz + 1) * sizeof(float);
    std::vector<int> cell_slices(static_cast<std::size_t>(nz));
    std::vector<int> w_slices(static_cast<std::size_t>(nz + 1));
    std::vector<int> y_slices(static_cast<std::size_t>(ny));
    std::iota(cell_slices.begin(), cell_slices.end(), 0);
    std::iota(w_slices.begin(), w_slices.end(), 0);
    std::iota(y_slices.begin(), y_slices.end(), 0);

    auto clampi = [](const int value, const int lo, const int hi) { return value < lo ? lo : (value > hi ? hi : value); };
    auto clampf = [](const float value, const float lo, const float hi) { return value < lo ? lo : (value > hi ? hi : value); };
    auto index_3d = [](const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    };
    auto fetch_clamped = [&](const float* field, const int x, const int y, const int z, const int sx, const int sy, const int sz) {
        return field[index_3d(clampi(x, 0, sx - 1), clampi(y, 0, sy - 1), clampi(z, 0, sz - 1), sx, sy)];
    };
    auto monotonic_cubic = [&](const float p0, const float p1, const float p2, const float p3, const float t) {
        const float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
        const float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
        const float a2 = -0.5f * p0 + 0.5f * p2;
        const float a3 = p1;
        return clampf(((a0 * t + a1) * t + a2) * t + a3, std::fmin(p1, p2), std::fmax(p1, p2));
    };
    auto sample_grid = [&](const float* field, float gx, float gy, float gz, const int sx, const int sy, const int sz) {
        gx = clampf(gx, 0.0f, static_cast<float>(sx - 1));
        gy = clampf(gy, 0.0f, static_cast<float>(sy - 1));
        gz = clampf(gz, 0.0f, static_cast<float>(sz - 1));
        if (!cubic) {
            const int x0 = clampi(static_cast<int>(std::floor(gx)), 0, sx - 1);
            const int y0 = clampi(static_cast<int>(std::floor(gy)), 0, sy - 1);
            const int z0 = clampi(static_cast<int>(std::floor(gz)), 0, sz - 1);
            const int x1 = std::min(x0 + 1, sx - 1);
            const int y1 = std::min(y0 + 1, sy - 1);
            const int z1 = std::min(z0 + 1, sz - 1);
            const float tx = gx - static_cast<float>(x0);
            const float ty = gy - static_cast<float>(y0);
            const float tz = gz - static_cast<float>(z0);
            const float c000 = field[index_3d(x0, y0, z0, sx, sy)];
            const float c100 = field[index_3d(x1, y0, z0, sx, sy)];
            const float c010 = field[index_3d(x0, y1, z0, sx, sy)];
            const float c110 = field[index_3d(x1, y1, z0, sx, sy)];
            const float c001 = field[index_3d(x0, y0, z1, sx, sy)];
            const float c101 = field[index_3d(x1, y0, z1, sx, sy)];
            const float c011 = field[index_3d(x0, y1, z1, sx, sy)];
            const float c111 = field[index_3d(x1, y1, z1, sx, sy)];
            const float c00 = c000 + (c100 - c000) * tx;
            const float c10 = c010 + (c110 - c010) * tx;
            const float c01 = c001 + (c101 - c001) * tx;
            const float c11 = c011 + (c111 - c011) * tx;
            const float c0 = c00 + (c10 - c00) * ty;
            const float c1 = c01 + (c11 - c01) * ty;
            return c0 + (c1 - c0) * tz;
        }
        const int ix = static_cast<int>(std::floor(gx));
        const int iy = static_cast<int>(std::floor(gy));
        const int iz = static_cast<int>(std::floor(gz));
        const float tx = gx - static_cast<float>(ix);
        const float ty = gy - static_cast<float>(iy);
        const float tz = gz - static_cast<float>(iz);
        float yz[4][4];
        for (int zz = 0; zz < 4; ++zz)
            for (int yy = 0; yy < 4; ++yy)
                yz[zz][yy] = monotonic_cubic(fetch_clamped(field, ix - 1, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 0, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 1, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 2, iy + yy - 1, iz + zz - 1, sx, sy, sz), tx);
        float zline[4];
        for (int zz = 0; zz < 4; ++zz) zline[zz] = monotonic_cubic(yz[zz][0], yz[zz][1], yz[zz][2], yz[zz][3], ty);
        return monotonic_cubic(zline[0], zline[1], zline[2], zline[3], tz);
    };
    auto sample_scalar = [&](const float* field, const float x, const float y, const float z) { return sample_grid(field, x / cell_size - 0.5f, y / cell_size - 0.5f, z / cell_size - 0.5f, nx, ny, nz); };
    auto sample_u = [&](const float* field, const float x, const float y, const float z) { return sample_grid(field, x / cell_size, y / cell_size - 0.5f, z / cell_size - 0.5f, nx + 1, ny, nz); };
    auto sample_v = [&](const float* field, const float x, const float y, const float z) { return sample_grid(field, x / cell_size - 0.5f, y / cell_size, z / cell_size - 0.5f, nx, ny + 1, nz); };
    auto sample_w = [&](const float* field, const float x, const float y, const float z) { return sample_grid(field, x / cell_size - 0.5f, y / cell_size - 0.5f, z / cell_size, nx, ny, nz + 1); };
    auto clamp_domain = [&](float& x, float& y, float& z) {
        x = clampf(x, 0.0f, static_cast<float>(nx) * cell_size);
        y = clampf(y, 0.0f, static_cast<float>(ny) * cell_size);
        z = clampf(z, 0.0f, static_cast<float>(nz) * cell_size);
    };
    auto sample_velocity = [&](const float* u, const float* v, const float* w, float x, float y, float z, float& out_x, float& out_y, float& out_z) {
        clamp_domain(x, y, z);
        out_x = sample_u(u, x, y, z);
        out_y = sample_v(v, x, y, z);
        out_z = sample_w(w, x, y, z);
    };
    auto center_u = [&](const float* u, const int i, const int j, const int k) {
        const int ci = clampi(i, 0, nx - 1);
        const int cj = clampi(j, 0, ny - 1);
        const int ck = clampi(k, 0, nz - 1);
        return 0.5f * (fetch_clamped(u, ci, cj, ck, nx + 1, ny, nz) + fetch_clamped(u, ci + 1, cj, ck, nx + 1, ny, nz));
    };
    auto center_v = [&](const float* v, const int i, const int j, const int k) {
        const int ci = clampi(i, 0, nx - 1);
        const int cj = clampi(j, 0, ny - 1);
        const int ck = clampi(k, 0, nz - 1);
        return 0.5f * (fetch_clamped(v, ci, cj, ck, nx, ny + 1, nz) + fetch_clamped(v, ci, cj + 1, ck, nx, ny + 1, nz));
    };
    auto center_w = [&](const float* w, const int i, const int j, const int k) {
        const int ci = clampi(i, 0, nx - 1);
        const int cj = clampi(j, 0, ny - 1);
        const int ck = clampi(k, 0, nz - 1);
        return 0.5f * (fetch_clamped(w, ci, cj, ck, nx, ny, nz + 1) + fetch_clamped(w, ci, cj, ck + 1, nx, ny, nz + 1));
    };
    auto set_boundaries = [&](float* u, float* v, float* w) {
        std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
            for (int j = 0; j < ny; ++j) {
                u[index_3d(0, j, k, nx + 1, ny)] = 0.0f;
                u[index_3d(nx, j, k, nx + 1, ny)] = 0.0f;
            }
        });
        std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
            for (int i = 0; i < nx; ++i) {
                v[index_3d(i, 0, k, nx, ny + 1)] = 0.0f;
                v[index_3d(i, ny, k, nx, ny + 1)] = 0.0f;
            }
        });
        std::for_each(std::execution::par_unseq, y_slices.begin(), y_slices.end(), [&](const int j) {
            for (int i = 0; i < nx; ++i) {
                w[index_3d(i, j, 0, nx, ny)] = 0.0f;
                w[index_3d(i, j, nz, nx, ny)] = 0.0f;
            }
        });
    };

    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const float dw_dy = (center_w(velocity_z, i, j + 1, k) - center_w(velocity_z, i, j - 1, k)) / (2.0f * cell_size);
                const float dv_dz = (center_v(velocity_y, i, j, k + 1) - center_v(velocity_y, i, j, k - 1)) / (2.0f * cell_size);
                const float du_dz = (center_u(velocity_x, i, j, k + 1) - center_u(velocity_x, i, j, k - 1)) / (2.0f * cell_size);
                const float dw_dx = (center_w(velocity_z, i + 1, j, k) - center_w(velocity_z, i - 1, j, k)) / (2.0f * cell_size);
                const float dv_dx = (center_v(velocity_y, i + 1, j, k) - center_v(velocity_y, i - 1, j, k)) / (2.0f * cell_size);
                const float du_dy = (center_u(velocity_x, i, j + 1, k) - center_u(velocity_x, i, j - 1, k)) / (2.0f * cell_size);
                const float wx = dw_dy - dv_dz;
                const float wy = du_dz - dw_dx;
                const float wz = dv_dx - du_dy;
                const auto index = index_3d(i, j, k, nx, ny);
                omega_x[index] = wx;
                omega_y[index] = wy;
                omega_z[index] = wz;
                omega_magnitude[index] = std::sqrt(wx * wx + wy * wy + wz * wz);
            }
    });

    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                auto mag = [&](const int x, const int y, const int z) { return fetch_clamped(omega_magnitude, x, y, z, nx, ny, nz); };
                const float gx = (mag(i + 1, j, k) - mag(i - 1, j, k)) / (2.0f * cell_size);
                const float gy = (mag(i, j + 1, k) - mag(i, j - 1, k)) / (2.0f * cell_size);
                const float gz = (mag(i, j, k + 1) - mag(i, j, k - 1)) / (2.0f * cell_size);
                const float inv_len = 1.0f / std::sqrt(std::max(gx * gx + gy * gy + gz * gz, 1.0e-12f));
                const float nxv = gx * inv_len;
                const float nyv = gy * inv_len;
                const float nzv = gz * inv_len;
                const auto index = index_3d(i, j, k, nx, ny);
                force_x[index] = vorticity_epsilon * cell_size * (nyv * omega_z[index] - nzv * omega_y[index]);
                force_y[index] = vorticity_epsilon * cell_size * (nzv * omega_x[index] - nxv * omega_z[index]);
                force_z[index] = vorticity_epsilon * cell_size * (nxv * omega_y[index] - nyv * omega_x[index]);
            }
    });

    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 1; i < nx; ++i)
                velocity_x[index_3d(i, j, k, nx + 1, ny)] += 0.5f * dt * (force_x[index_3d(i - 1, j, k, nx, ny)] + force_x[index_3d(i, j, k, nx, ny)]);
    });
    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 1; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const auto below = index_3d(i, j - 1, k, nx, ny);
                const auto above = index_3d(i, j, k, nx, ny);
                const float density_avg = 0.5f * (density[below] + density[above]);
                const float temperature_avg = 0.5f * (temperature[below] + temperature[above]);
                const float confinement_avg = 0.5f * (force_y[below] + force_y[above]);
                const float buoyancy = temperature_buoyancy * (temperature_avg - ambient_temperature) - density_buoyancy * density_avg;
                velocity_y[index_3d(i, j, k, nx, ny + 1)] += dt * (buoyancy + confinement_avg);
            }
    });
    std::for_each(std::execution::par_unseq, w_slices.begin() + 1, w_slices.end() - 1, [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                velocity_z[index_3d(i, j, k, nx, ny)] += 0.5f * dt * (force_z[index_3d(i, j, k - 1, nx, ny)] + force_z[index_3d(i, j, k, nx, ny)]);
    });

    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                float px = static_cast<float>(i) * cell_size;
                float py = (static_cast<float>(j) + 0.5f) * cell_size;
                float pz = (static_cast<float>(k) + 0.5f) * cell_size;
                float vx, vy, vz;
                sample_velocity(velocity_x, velocity_y, velocity_z, px, py, pz, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz);
                previous_velocity_x[index_3d(i, j, k, nx + 1, ny)] = sample_u(velocity_x, px, py, pz);
            }
    });
    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float px = (static_cast<float>(i) + 0.5f) * cell_size;
                float py = static_cast<float>(j) * cell_size;
                float pz = (static_cast<float>(k) + 0.5f) * cell_size;
                float vx, vy, vz;
                sample_velocity(velocity_x, velocity_y, velocity_z, px, py, pz, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz);
                previous_velocity_y[index_3d(i, j, k, nx, ny + 1)] = sample_v(velocity_y, px, py, pz);
            }
    });
    std::for_each(std::execution::par_unseq, w_slices.begin(), w_slices.end(), [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float px = (static_cast<float>(i) + 0.5f) * cell_size;
                float py = (static_cast<float>(j) + 0.5f) * cell_size;
                float pz = static_cast<float>(k) * cell_size;
                float vx, vy, vz;
                sample_velocity(velocity_x, velocity_y, velocity_z, px, py, pz, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz);
                previous_velocity_z[index_3d(i, j, k, nx, ny)] = sample_w(velocity_z, px, py, pz);
            }
    });
    set_boundaries(previous_velocity_x, previous_velocity_y, previous_velocity_z);

    std::memset(pressure, 0, static_cast<std::size_t>(cell_bytes));
    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                divergence[index_3d(i, j, k, nx, ny)] = (fetch_clamped(previous_velocity_x, i + 1, j, k, nx + 1, ny, nz) - fetch_clamped(previous_velocity_x, i, j, k, nx + 1, ny, nz)
                    + fetch_clamped(previous_velocity_y, i, j + 1, k, nx, ny + 1, nz) - fetch_clamped(previous_velocity_y, i, j, k, nx, ny + 1, nz) + fetch_clamped(previous_velocity_z, i, j, k + 1, nx, ny, nz + 1) - fetch_clamped(previous_velocity_z, i, j, k, nx, ny, nz + 1)) / cell_size;
    });
    for (int iteration = 0; iteration < pressure_iterations; ++iteration)
        for (int parity = 0; parity < 2; ++parity)
            std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        if (((i + j + k) & 1) != parity) continue;
                        float sum = 0.0f;
                        int count = 0;
                        if (i > 0) {
                            sum += pressure[index_3d(i - 1, j, k, nx, ny)];
                            ++count;
                        }
                        if (i + 1 < nx) {
                            sum += pressure[index_3d(i + 1, j, k, nx, ny)];
                            ++count;
                        }
                        if (j > 0) {
                            sum += pressure[index_3d(i, j - 1, k, nx, ny)];
                            ++count;
                        }
                        if (j + 1 < ny) {
                            sum += pressure[index_3d(i, j + 1, k, nx, ny)];
                            ++count;
                        }
                        if (k > 0) {
                            sum += pressure[index_3d(i, j, k - 1, nx, ny)];
                            ++count;
                        }
                        if (k + 1 < nz) {
                            sum += pressure[index_3d(i, j, k + 1, nx, ny)];
                            ++count;
                        }
                        pressure[index_3d(i, j, k, nx, ny)] = count > 0 ? (sum - divergence[index_3d(i, j, k, nx, ny)] * cell_size * cell_size / dt) / static_cast<float>(count) : 0.0f;
                    }
            });
    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 1; i < nx; ++i)
                previous_velocity_x[index_3d(i, j, k, nx + 1, ny)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i - 1, j, k, nx, ny)]) / cell_size;
    });
    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 1; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                previous_velocity_y[index_3d(i, j, k, nx, ny + 1)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j - 1, k, nx, ny)]) / cell_size;
    });
    std::for_each(std::execution::par_unseq, w_slices.begin() + 1, w_slices.end() - 1, [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                previous_velocity_z[index_3d(i, j, k, nx, ny)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j, k - 1, nx, ny)]) / cell_size;
    });
    set_boundaries(previous_velocity_x, previous_velocity_y, previous_velocity_z);

    std::memcpy(velocity_x, previous_velocity_x, velocity_x_field_bytes);
    std::memcpy(velocity_y, previous_velocity_y, velocity_y_field_bytes);
    std::memcpy(velocity_z, previous_velocity_z, velocity_z_field_bytes);
    std::memcpy(previous_density, density, cell_bytes);
    std::memcpy(previous_temperature, temperature, cell_bytes);

    std::for_each(std::execution::par_unseq, cell_slices.begin(), cell_slices.end(), [&](const int k) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float px = (static_cast<float>(i) + 0.5f) * cell_size;
                float py = (static_cast<float>(j) + 0.5f) * cell_size;
                float pz = (static_cast<float>(k) + 0.5f) * cell_size;
                float vx, vy, vz;
                sample_velocity(velocity_x, velocity_y, velocity_z, px, py, pz, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz);
                density[index_3d(i, j, k, nx, ny)] = std::max(0.0f, sample_scalar(previous_density, px, py, pz));
                temperature[index_3d(i, j, k, nx, ny)] = sample_scalar(previous_temperature, px, py, pz);
            }
    });

    return 0;
}

} // extern "C"
