#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Representation of a body / particle in 3D space
struct Body {
  double x, y, z;       // Position coordinates
  double vx, vy, vz;    // Velocity components
  double mass;          // Mass of the body
};

// Gravitational and numerical constants
const double G = 1.0;              // Normalized gravitational constant
const double SOFTENING_SQ = 0.1;   // Softening factor squared (prevents extreme near-collision accelerations)
const double DT = 0.005;           // Time step size

// Deterministically initialize particles for reproducible benchmarks
void initBodies(vector<Body>& bodies, int n) {
  mt19937_64 rng(42);  // Fixed seed so results are identical across all implementations
  uniform_real_distribution<double> posDist(-10.0, 10.0);
  uniform_real_distribution<double> velDist(-1.0, 1.0);
  uniform_real_distribution<double> massDist(1.0, 5.0);

  for (int i = 0; i < n; i++) {
    bodies[i].x = posDist(rng);
    bodies[i].y = posDist(rng);
    bodies[i].z = posDist(rng);
    bodies[i].vx = velDist(rng);
    bodies[i].vy = velDist(rng);
    bodies[i].vz = velDist(rng);
    bodies[i].mass = massDist(rng);
  }
}

// Compute total energy (Kinetic + Potential) of the system
double computeTotalEnergy(const vector<Body>& bodies, int n) {
  double kinetic = 0.0;
  double potential = 0.0;

  for (int i = 0; i < n; i++) {
    // Kinetic energy: 0.5 * m * v^2
    double vSq = bodies[i].vx * bodies[i].vx +
                 bodies[i].vy * bodies[i].vy +
                 bodies[i].vz * bodies[i].vz;
    kinetic += 0.5 * bodies[i].mass * vSq;

    // Potential energy: - G * m_i * m_j / sqrt(r_ij^2 + softening^2)
    for (int j = i + 1; j < n; j++) {
      double dx = bodies[j].x - bodies[i].x;
      double dy = bodies[j].y - bodies[i].y;
      double dz = bodies[j].z - bodies[i].z;
      double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING_SQ);
      potential -= (G * bodies[i].mass * bodies[j].mass) / dist;
    }
  }

  return kinetic + potential;
}

int main(int argc, char* argv[]) {
  // Default benchmark configuration
  int N = 5000;         // Number of bodies
  int STEPS = 100;      // Number of simulation time steps
  int BLOCK_SIZE = 256; // Cache block / tile size (number of bodies per L1 cache tile)

  // Allow command-line overrides: ./NBody_serial_block [N] [STEPS] [BLOCK_SIZE]
  if (argc >= 2) N = atoi(argv[1]);
  if (argc >= 3) STEPS = atoi(argv[2]);
  if (argc >= 4) BLOCK_SIZE = atoi(argv[3]);

  vector<Body> bodies(N);
  initBodies(bodies, N);

  // Temporary arrays to store computed accelerations for each body
  vector<double> ax(N, 0.0);
  vector<double> ay(N, 0.0);
  vector<double> az(N, 0.0);

  // Fast L1 cache tile buffers (Structure-of-Arrays / SoA format for current tile).
  // Storing only x, y, z, mass (32 bytes instead of full 56-byte Body struct)
  // eliminates loading unused velocity fields into L1 cache and enables SIMD vectorization.
  vector<double> tile_x(BLOCK_SIZE);
  vector<double> tile_y(BLOCK_SIZE);
  vector<double> tile_z(BLOCK_SIZE);
  vector<double> tile_m(BLOCK_SIZE);

  cout << "========================================================\n";
  cout << "   N-Body Gravitational Simulation (Serial Block/Tiled) \n";
  cout << "========================================================\n";
  cout << "Number of bodies (N) : " << N << "\n";
  cout << "Time steps           : " << STEPS << "\n";
  cout << "Block / Tile size (B): " << BLOCK_SIZE << " bodies ("
       << (BLOCK_SIZE * 32.0 / 1024.0) << " KB in L1 tile)\n";
  cout << "Time step size (dt)  : " << DT << "\n";
  cout << "Softening factor^2   : " << SOFTENING_SQ << "\n";

  double initialEnergy = computeTotalEnergy(bodies, N);
  cout << fixed << setprecision(6);
  cout << "Initial Total Energy : " << initialEnergy << "\n";
  cout << "Running simulation...\n";

  auto start = chrono::high_resolution_clock::now();

  for (int step = 0; step < STEPS; step++) {
    // Reset acceleration accumulators for current time step
    fill(ax.begin(), ax.end(), 0.0);
    fill(ay.begin(), ay.end(), 0.0);
    fill(az.begin(), az.end(), 0.0);

    // 1. Cache-Blocked / Tiled Force Calculation:
    // Outer loop iterates over tiles of source bodies j.
    // Each source tile is loaded once into the L1 cache scratchpad (tile_x, tile_y, tile_z, tile_m).
    // Then all target bodies i interact with this cache-resident block.
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
      int bsize_j = min(BLOCK_SIZE, N - jj);

      // Pack active tile into contiguous L1 cache buffer
      for (int k = 0; k < bsize_j; k++) {
        tile_x[k] = bodies[jj + k].x;
        tile_y[k] = bodies[jj + k].y;
        tile_z[k] = bodies[jj + k].z;
        tile_m[k] = bodies[jj + k].mass;
      }

      // Loop over blocks of target bodies i
      for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        int i_end = min(ii + BLOCK_SIZE, N);

        for (int i = ii; i < i_end; i++) {
          double bi_x = bodies[i].x;
          double bi_y = bodies[i].y;
          double bi_z = bodies[i].z;
          double accX = 0.0;
          double accY = 0.0;
          double accZ = 0.0;

          // Inner loop: evaluated against contiguous L1 cache-resident tile
          for (int k = 0; k < bsize_j; k++) {
            double dx = tile_x[k] - bi_x;
            double dy = tile_y[k] - bi_y;
            double dz = tile_z[k] - bi_z;
            double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING_SQ;
            double invDist = 1.0 / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            double s = G * tile_m[k] * invDist3;
            accX += s * dx;
            accY += s * dy;
            accZ += s * dz;
          }

          ax[i] += accX;
          ay[i] += accY;
          az[i] += accZ;
        }
      }
    }

    // 2. Integration step: update velocities and positions (Symplectic Euler)
    for (int i = 0; i < N; i++) {
      bodies[i].vx += ax[i] * DT;
      bodies[i].vy += ay[i] * DT;
      bodies[i].vz += az[i] * DT;

      bodies[i].x += bodies[i].vx * DT;
      bodies[i].y += bodies[i].vy * DT;
      bodies[i].z += bodies[i].vz * DT;
    }
  }

  auto end = chrono::high_resolution_clock::now();
  double elapsed = chrono::duration<double>(end - start).count();

  double finalEnergy = computeTotalEnergy(bodies, N);
  double energyError = fabs((finalEnergy - initialEnergy) / initialEnergy) * 100.0;

  // Compute performance metrics
  double totalInteractions = (double)N * (double)N * (double)STEPS;
  double flopsPerInteraction = 20.0;
  double totalGFlops = (totalInteractions * flopsPerInteraction) / 1e9;
  double gflopsRate = totalGFlops / elapsed;

  cout << "========================================================\n";
  cout << "Execution time       : " << elapsed << " seconds\n";
  cout << "Time per step        : " << (elapsed / STEPS) * 1000.0 << " ms\n";
  cout << "Throughput           : " << totalInteractions / elapsed / 1e6 << " M-Interactions/sec\n";
  cout << "Compute Performance  : " << gflopsRate << " GFLOP/s\n";
  cout << "Final Total Energy   : " << finalEnergy << "\n";
  cout << "Energy Drift         : " << energyError << " %\n";
  cout << "Sample Body 0 pos    : (" << bodies[0].x << ", " << bodies[0].y << ", " << bodies[0].z << ")\n";
  cout << "Sample Body 0 vel    : (" << bodies[0].vx << ", " << bodies[0].vy << ", " << bodies[0].vz << ")\n";
  cout << "========================================================\n";

  return 0;
}
