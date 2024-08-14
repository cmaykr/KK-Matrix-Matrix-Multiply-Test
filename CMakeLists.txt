cmake_minimum_required(VERSION 3.16)

project(speedup_Test)

find_package(Kokkos PATHS /home/david/Kokkos/kokkos/build REQUIRED)

add_executable(speedup_Test main.cpp)
target_link_libraries(speedup_Test Kokkos::kokkos)

