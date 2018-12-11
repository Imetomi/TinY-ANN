cmake_minimum_required(VERSION 3.12)
project(Neural_Network_in_C C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_FLAGS -Wall)
set(CMAKE_CXX_FLAGS -pedantic)
set(CMAKE_CXX_FLAGS -g)
set(CMAKE_CXX_FLAGS -Werror)

add_definitions("-Wall")
add_definitions("-Werror")
add_definitions("-pedantic")
add_definitions("-g")


add_executable(Neural_Network_in_C perceptron.h perceptron.c perceptron_libs.c perceptron_plotter.c
               example_spiral.c debugmalloc.h debugmalloc.c)
target_link_libraries(Neural_Network_in_C -lmingw32 -lSDL2main -lSDL2 -lSDL2_gfx -lSDL2_ttf -lSDL2_image -lSDL2_mixer
                -static-libgcc)

message(none)
