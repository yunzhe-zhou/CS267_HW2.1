#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#define MAX_P 5

typedef struct grid_class {
    int num_p;
    particle_t* members[MAX_P];
} grid_class;


int ngrid;
grid_class* grids;


// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

static inline void apply_force_all(int gx, int gy, particle_t* part) {
    for (int px = gx; px < gx + 3; px++) {
        for (int py = gy; py < gy + 3; py++) {
            // If the neighbor is outside the grid, let's abondon it.
            if (px >= ngrid || py >= ngrid) continue;

            // Assign the grid for this neighbor.
            grid_class* grid = &grids[px * ngrid + py];

            if (grid->num_p > 0) {
                for (int j = 0; j < MAX_P; j++) {
                    if (grid->members[j] == NULL) {
                        continue;
                    }
                    // If the neighbor is the particle itself, we don't need to consider the force.
                    if (part == grid->members[j]){
                        continue;
                    } 
                    apply_force(*part, *grid->members[j]);
                }
            }
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    // Determine the number of grids
    ngrid = (int)ceil(size / cutoff);
    // Allocate the memory spaces for the whole grids
    grids = (grid_class*)calloc(ngrid * ngrid, sizeof(grid_class));

    // Assign particles to each grid
    for (int i = 0; i < num_parts; i++) {
        // Get the location of grid for the particle
        int gx = (int)(parts[i].x / cutoff);
        int gy = (int)(parts[i].y / cutoff);

        // Assign the grid object
        grid_class* grid = &grids[gx * ngrid + gy];

        // Add the info of the particle into the grid and update num_p of the grid
        for (int ii = 0; ii < MAX_P; ii++) {
            if (grid->members[ii] == NULL) {
                grid->members[ii] = &parts[i];
                grid->num_p++;
                break;
            }
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Update the grids
    for (int i = 0; i < ngrid; i++) {
        for (int j = 0; j < ngrid; j++) {
            // Consider each grid
            grid_class* grid = &grids[i * ngrid + j];
            // If it is empty, let's skip it.
            if (grid->num_p == 0) continue;
            for (int k = 0; k < MAX_P; k++) {
                // Consider each member of the grid
                particle_t* part = grid->members[k];
                if (part == NULL) continue;
                // Get the updated grid location for this member after being applied the force.
                int gx = (int)(part->x / cutoff);
                int gy = (int)(part->y / cutoff);
                // If the location remains the same, let's skip it.
                if (gx == i && gy == j) continue;
                // Update the grid of this member.
                grid_class* new_grid = &grids[gx * ngrid + gy];

                for (int ii = 0; ii < MAX_P; ii++) {
                    if (new_grid->members[ii] == NULL) {
                        // Add the member to the new updated grid.
                        new_grid->num_p++;
                        new_grid->members[ii] = grid->members[k];
                        // Remove the member from its previous grid.
                        grid->num_p--;
                        grid->members[k] = NULL;
                        break;
                    }
                }
            }
        }
    }

    // Compute Forces
    for(int i = 0; i < ngrid * ngrid; i++) {
        if (0 != grids[i].num_p) {
            for (int j = 0; j < MAX_P; j++) {
                if (grids[i].members[j] == NULL) continue;
                particle_t* part = grids[i].members[j];
                part->ax = part->ay = 0;
                int gx = (int)(part->x / cutoff) - 1;
                int gy = (int)(part->y / cutoff) - 1;
                if(gx < 0) {
                    gx = 0;
                }
                if(gy < 0) {
                    gy = 0;
                }
                apply_force_all(gx, gy, part);
            }
        }
    }

    // Move Particles
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
    }
}

