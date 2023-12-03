cmake_minimum_required(VERSION 3.22)
project(HITS)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_FLAGS_DEBUG "-march=native ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native  ${CMAKE_CXX_FLAGS_RELEASE}")

set(gpu_device torch::kCUDA)
set(data_father_path "\"/home/redamancyguy/Desktop/buffer/data/\"")

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(Torch_DIR /home/redamancyguy/Desktop/libtorch/share/cmake/Torch)

configure_file(
        "${PROJECT_SOURCE_DIR}/include/DEFINE.h.in"
        "${PROJECT_SOURCE_DIR}/include/DEFINE.h"
)
configure_file(
        "${PROJECT_SOURCE_DIR}/index/include/Parameter.h.in"
        "${PROJECT_SOURCE_DIR}/index/include/Parameter.h"
)

#execute_process(COMMAND "conda deactivate")
find_package(PythonInterp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS fiber date_time system filesystem)
set(gpu_device torch::kCUDA)
set(data_father_path "\"/home/redamancyguy/Desktop/buffer/data/\"")

link_libraries("${TORCH_LIBRARIES}" Python3::Python Python3::NumPy)

#set(MY_STACK_SIZE 1048576k)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MY_STACK_SIZE}")

add_executable(main main.cpp
#        index/include/Configuration.hpp
#        index/include/Parameter.h index/include/Controller.hpp
#        index/include/experience.hpp index/include/Index.hpp
#        index/include/RL_network.hpp index/workspace/gen_exp.hpp
#        index/workspace/global_on_policy.cpp index/workspace/train.cpp
#        index/workspace/train_tree.cpp
        )

add_executable(train index/workspace/train.cpp)

add_executable(train_TSMDP index/workspace/train_TSMDP.cpp)

add_executable(train_DARE index/workspace/train_DARE.cpp)

add_executable(dynamic_data benchmark/dynamic_data.cpp)

add_executable(dynamic_data2 benchmark/dynamic_data2.cpp)

add_executable(add_get_erase benchmark/add_get_erase.cpp)

add_executable(read_only benchmark/read_only.cpp)

add_executable(parameters benchmark/parameters.cpp)

add_executable(test benchmark/test.cpp )

add_executable(opration_latency benchmark/opration_latency.cpp)

add_executable(scalability benchmark/scalability.cpp)

add_executable(why_retraining_is_needed benchmark/why_retraining_is_needed.cpp)
find_package(Threads REQUIRED)
add_executable(concurrency_test benchmark/concurrency_test.cpp)
target_link_libraries(concurrency_test PUBLIC Threads::Threads)
