#include "visual-simulation-of-smoke.h"

extern "C" {

int32_t visual_simulation_of_smoke_validate_desc(const VisualSimulationOfSmokeStepDesc* desc) {
    if (desc == nullptr) return 1000;
    if (desc->struct_size < sizeof(VisualSimulationOfSmokeStepDesc)) return 1000;
    if (desc->api_version != VISUAL_SIMULATION_OF_SMOKE_API_VERSION) return 1005;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->cell_size <= 0.0f) return 1002;
    if (desc->dt <= 0.0f) return 1003;
    if (desc->pressure_iterations <= 0) return 1004;
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
    return 0;
}

} // extern "C"
