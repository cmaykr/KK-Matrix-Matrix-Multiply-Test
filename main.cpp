#include <iostream>
#include <random>
#include <iomanip>

#include <Kokkos_Core.hpp>

#define N 10
int C[N][N], A[N][N], B[N][N], D[N][N];

struct TagA {};
struct Foo{

    KOKKOS_INLINE_FUNCTION
    void operator() (const TagA, const int i)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
};

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);

    if (argc != 2)
    {
        std::cerr << "Wrong amount of arguments, only takes array size";
        exit(1);
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 3000);

    double avgSpeedup = 0;
    for (int t = 0; t < 10; ++t)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                A[i][j] = distrib(gen);
                B[i][j] = distrib(gen);
            }
        }
        Kokkos::Timer timer;
        timer.reset();

        Kokkos::parallel_for("Outer loop", N, KOKKOS_LAMBDA (const int i) {
            Kokkos::parallel_for("Inner loop", N, KOKKOS_LAMBDA (const int j)
            {
                C[i][j] = 0;
                for (int k = 0; k < N; k++)
                {
                    C[i][j] = C[i][j] + A[i][k] * B[k][j];
                }
            });
        });

        double paraTime = timer.seconds();

        timer.reset();

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                D[i][j] = 0;
                for (int k = 0; k < N; k++)
                {
                    D[i][j] = D[i][j] + A[i][k] * B[k][j];
                }
            }
        }

        double time = timer.seconds();

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (C[i][j] != D[i][j])
                {
                    std::cerr << "Wrong multiply" << std::endl;
                    exit(1);
                }
            }
        }

        printf("Sequential time: %10.6f\n", time);
        printf("Parallel time: %10.6f\n", paraTime);
        double speedup = time / paraTime;
        printf("Speedup for an array size of %i: %f\n", N, speedup);

        avgSpeedup += speedup / 10;
    }
    std::cout << "Average Speedup: " << avgSpeedup << std::endl;
        Kokkos::finalize();
    return 0;
}