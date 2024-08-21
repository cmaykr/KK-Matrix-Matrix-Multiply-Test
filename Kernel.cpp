#include<Kokkos_Core.hpp>
#include<KokkosBlas3_gemm.hpp>

int main(int argc, char* argv[]) {
   Kokkos::ScopeGuard guard;

   int M = atoi(argv[1]);
   int N = atoi(argv[2]);

   Kokkos::View<double**> A("A",M,N);
   Kokkos::View<double**> B("B",N,M);
   Kokkos::View<double**> C("C",M,M);
   
   Kokkos::deep_copy(A,1.0);
   Kokkos::deep_copy(B,2.0);

   const double alpha = double(1.0);
   const double beta = double(0.0);
   
   KokkosBlas::gemm("N","N",alpha,A,B,beta,C);

    Kokkos::fence();
}