#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <omp.h>
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
  mt19937_64 rng(42);  // Fixed seed so results are identical across serial, OpenMP, and MPI
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
  int N = 5000;      // Number of bodies
  int STEPS = 100;   // Number of simulation time steps

  // Allow command-line overrides: ./NBody_openMP [N] [STEPS] [THREADS]
  if (argc >= 2) N = atoi(argv[1]);
  if (argc >= 3) STEPS = atoi(argv[2]);
  if (argc >= 4) omp_set_num_threads(atoi(argv[3]));

  int numThreads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    numThreads = omp_get_num_threads();
  }

  vector<Body> bodies(N);
  initBodies(bodies, N);

  // Temporary arrays to store computed accelerations for each body
  vector<double> ax(N, 0.0);
  vector<double> ay(N, 0.0);
  vector<double> az(N, 0.0);

  cout << "========================================================\n";
  cout << "       N-Body Gravitational Simulation (OpenMP)         \n";
  cout << "========================================================\n";
  cout << "Number of OpenMP threads: " << numThreads << "\n";
  cout << "Number of bodies (N)    : " << N << "\n";
  cout << "Time steps              : " << STEPS << "\n";
  cout << "Time step size (dt)     : " << DT << "\n";
  cout << "Softening factor^2      : " << SOFTENING_SQ << "\n";

  double initialEnergy = computeTotalEnergy(bodies, N);
  cout << fixed << setprecision(6);
  cout << "Initial Total Energy    : " << initialEnergy << "\n";
  cout << "Running simulation...\n";

  auto start = chrono::high_resolution_clock::now();

  for (int step = 0; step < STEPS; step++) {
    // 1. Multithreaded force computation
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
      double accX = 0.0;
      double accY = 0.0;
      double accZ = 0.0;

      for (int j = 0; j < N; j++) {
        double dx = bodies[j].x - bodies[i].x;
        double dy = bodies[j].y - bodies[i].y;
        double dz = bodies[j].z - bodies[i].z;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING_SQ;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;

        double s = G * bodies[j].mass * invDist3;
        accX += s * dx;
        accY += s * dy;
        accZ += s * dz;
      }

      ax[i] = accX;
      ay[i] = accY;
      az[i] = accZ;
    }

    // 2. Multithreaded integration step
    #pragma omp parallel for schedule(static)
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
  cout << "Execution time          : " << elapsed << " seconds\n";
  cout << "Time per step           : " << (elapsed / STEPS) * 1000.0 << " ms\n";
  cout << "Throughput              : " << totalInteractions / elapsed / 1e6 << " M-Interactions/sec\n";
  cout << "Compute Performance     : " << gflopsRate << " GFLOP/s\n";
  cout << "Final Total Energy      : " << finalEnergy << "\n";
  cout << "Energy Drift            : " << energyError << " %\n";
  cout << "Sample Body 0 pos       : (" << bodies[0].x << ", " << bodies[0].y << ", " << bodies[0].z << ")\n";
  cout << "Sample Body 0 vel       : (" << bodies[0].vx << ", " << bodies[0].vy << ", " << bodies[0].vz << ")\n";
  cout << "========================================================\n";

  return 0;
}
