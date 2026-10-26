#include <mpi.h>
#include <omp.h>
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

// Ensure Body is tightly packed with 7 doubles
static_assert(sizeof(Body) == 7 * sizeof(double), "Body struct size mismatch with 7 doubles");

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
  // Initialize MPI with thread support (funneled: only master thread makes MPI calls)
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (provided < MPI_THREAD_FUNNELED) {
    if (rank == 0) {
      cerr << "Warning: MPI does not fully support MPI_THREAD_FUNNELED\n";
    }
  }

  // Default benchmark configuration
  int N = 5000;      // Number of bodies
  int STEPS = 100;   // Number of simulation time steps

  // Allow command-line overrides: mpiexec -n P ./NBody_mpi_openMP [N] [STEPS]
  if (argc >= 2) N = atoi(argv[1]);
  if (argc >= 3) STEPS = atoi(argv[2]);

  // Determine number of OpenMP threads
  int numThreads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    numThreads = omp_get_num_threads();
  }

  // Create MPI Datatype for Body struct (7 contiguous doubles)
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);

  // 1D Domain Decomposition: partition bodies among MPI processes
  vector<int> counts(size);
  vector<int> displs(size);

  for (int r = 0; r < size; r++) {
    counts[r] = N / size + (r < (N % size) ? 1 : 0);
    displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
  }

  int local_n = counts[rank];
  int local_start = displs[rank];

  // Full array of bodies (each rank maintains a copy of all positions)
  vector<Body> bodies(N);

  // Rank 0 initializes and broadcasts initial bodies to all processes
  if (rank == 0) {
    initBodies(bodies, N);
  }
  MPI_Bcast(bodies.data(), N, MPI_BODY, 0, MPI_COMM_WORLD);

  // Local arrays for accelerations of bodies owned by this rank
  vector<double> local_ax(local_n, 0.0);
  vector<double> local_ay(local_n, 0.0);
  vector<double> local_az(local_n, 0.0);

  double initialEnergy = 0.0;
  if (rank == 0) {
    cout << "========================================================\n";
    cout << "     N-Body Gravitational Simulation (Hybrid MPI+OpenMP)\n";
    cout << "========================================================\n";
    cout << "Number of MPI processes : " << size << "\n";
    cout << "OpenMP threads / process: " << numThreads << "\n";
    cout << "Total worker threads    : " << size * numThreads << "\n";
    cout << "Number of bodies (N)    : " << N << "\n";
    cout << "Bodies per process      : ~" << N / size << "\n";
    cout << "Time steps              : " << STEPS << "\n";
    cout << "Time step size (dt)     : " << DT << "\n";
    cout << "Softening factor^2      : " << SOFTENING_SQ << "\n";

    initialEnergy = computeTotalEnergy(bodies, N);
    cout << fixed << setprecision(6);
    cout << "Initial Total Energy    : " << initialEnergy << "\n";
    cout << "Running simulation...\n";
  }

  // Synchronize before timing
  MPI_Barrier(MPI_COMM_WORLD);
  double startTime = MPI_Wtime();

  double totalComputeTime = 0.0;
  double totalCommTime = 0.0;

  for (int step = 0; step < STEPS; step++) {
    double t0 = MPI_Wtime();

    // 1. Compute gravitational accelerations on local bodies using OpenMP multithreading
    // The outer loop over local bodies is divided across OpenMP threads
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_n; i++) {
      int global_i = local_start + i;
      double accX = 0.0;
      double accY = 0.0;
      double accZ = 0.0;

      for (int j = 0; j < N; j++) {
        double dx = bodies[j].x - bodies[global_i].x;
        double dy = bodies[j].y - bodies[global_i].y;
        double dz = bodies[j].z - bodies[global_i].z;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING_SQ;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;

        double s = G * bodies[j].mass * invDist3;
        accX += s * dx;
        accY += s * dy;
        accZ += s * dz;
      }

      local_ax[i] = accX;
      local_ay[i] = accY;
      local_az[i] = accZ;
    }

    // 2. Integration step: update velocities and positions of local bodies (OpenMP)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_n; i++) {
      int global_i = local_start + i;
      bodies[global_i].vx += local_ax[i] * DT;
      bodies[global_i].vy += local_ay[i] * DT;
      bodies[global_i].vz += local_az[i] * DT;

      bodies[global_i].x += bodies[global_i].vx * DT;
      bodies[global_i].y += bodies[global_i].vy * DT;
      bodies[global_i].z += bodies[global_i].vz * DT;
    }

    double t1 = MPI_Wtime();
    totalComputeTime += (t1 - t0);

    // 3. Communication: Gather updated bodies from all MPI ranks to synchronize positions
    MPI_Allgatherv(&bodies[local_start], local_n, MPI_BODY,
                   bodies.data(), counts.data(), displs.data(), MPI_BODY,
                   MPI_COMM_WORLD);

    double t2 = MPI_Wtime();
    totalCommTime += (t2 - t1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double endTime = MPI_Wtime();
  double elapsed = endTime - startTime;

  // Reduce timing metrics across all processes to find max elapsed
  double maxElapsed, maxCompute, maxComm;
  MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&totalComputeTime, &maxCompute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&totalCommTime, &maxComm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double finalEnergy = computeTotalEnergy(bodies, N);
    double energyError = fabs((finalEnergy - initialEnergy) / initialEnergy) * 100.0;

    // Performance metrics
    double totalInteractions = (double)N * (double)N * (double)STEPS;
    double flopsPerInteraction = 20.0;
    double totalGFlops = (totalInteractions * flopsPerInteraction) / 1e9;
    double gflopsRate = totalGFlops / maxElapsed;

    cout << "========================================================\n";
    cout << "Total Elapsed time      : " << maxElapsed << " seconds\n";
    cout << "  - Compute time        : " << maxCompute << " s (" << (maxCompute / maxElapsed) * 100.0 << "%)\n";
    cout << "  - MPI Comm time       : " << maxComm << " s (" << (maxComm / maxElapsed) * 100.0 << "%)\n";
    cout << "Time per step           : " << (maxElapsed / STEPS) * 1000.0 << " ms\n";
    cout << "Throughput              : " << totalInteractions / maxElapsed / 1e6 << " M-Interactions/sec\n";
    cout << "Compute Performance     : " << gflopsRate << " GFLOP/s\n";
    cout << "Final Total Energy      : " << finalEnergy << "\n";
    cout << "Energy Drift            : " << energyError << " %\n";
    cout << "Sample Body 0 pos       : (" << bodies[0].x << ", " << bodies[0].y << ", " << bodies[0].z << ")\n";
    cout << "Sample Body 0 vel       : (" << bodies[0].vx << ", " << bodies[0].vy << ", " << bodies[0].vz << ")\n";
    cout << "========================================================\n";
  }

  MPI_Type_free(&MPI_BODY);
  MPI_Finalize();
  return 0;
}
