cmake_minimum_required(VERSION 3.16)

project(test_pusher)
find_package(Kokkos REQUIRED)
add_executable(test_pusher test.cpp)
target_link_libraries(test_pusher Kokkos::kokkos)
target_include_directories(test_pusher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..    # src/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/# src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/kernels# src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/global# src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../plog/include# src
)


