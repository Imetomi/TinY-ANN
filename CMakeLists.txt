cmake_minimum_required(VERSION 3.12)
project(Neural_Network_in_C C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_EXE_LINKER_FLAGS  -Wall)
set(CMAKE_EXE_LINKER_FLAGS  -pedantic)
set(CMAKE_EXE_LINKER_FLAGS  -g)
set(CMAKE_EXE_LINKER_FLAGS  -Werror)

add_executable(Neural_Network_in_C perceptron.h perceptron.c perceptron_libs.c perceptron_plotter.c example_wine.c)
target_link_libraries(Neural_Network_in_C -lmingw32 -lSDL2main -lSDL2 -lSDL2_gfx -lSDL2_ttf -lSDL2_image -lSDL2_mixer)