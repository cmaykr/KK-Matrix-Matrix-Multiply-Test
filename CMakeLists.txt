cmake_minimum_required(VERSION 3.16)

project(kernel_Test)

find_package(Kokkos PATHS /home/david/Kokkos/kokkos/build REQUIRED)

add_executable(speedup_Test main.cpp)
target_link_libraries(speedup_Test Kokkos::kokkos)

find_package(KokkosKernels PATHS /home/david/Kokkos/kokkos-kernels/install REQUIRED)
add_executable(kernel_Test Kernel.cpp)
target_link_libraries(kernel_Test Kokkos::kokkoskernels)

