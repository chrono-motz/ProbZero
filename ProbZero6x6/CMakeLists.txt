cmake_minimum_required(VERSION 3.10)
project(c4_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 1. Training Engine
add_executable(c4_engine
    src/main.cpp
    src/mcts.cpp
    src/game.hpp
    src/mcts.hpp
    src/node.hpp
)
target_link_libraries(c4_engine ${TORCH_LIBRARIES})
set_property(TARGET c4_engine PROPERTY CXX_STANDARD 17)

# 2. Playing Arena
add_executable(play
    src/play.cpp
    src/mcts.cpp
    src/game.hpp
    src/mcts.hpp
    src/node.hpp
)
target_link_libraries(play ${TORCH_LIBRARIES})
set_property(TARGET play PROPERTY CXX_STANDARD 17)
