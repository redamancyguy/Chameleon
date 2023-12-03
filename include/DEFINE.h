//
// Created by wenli on 2023/1/11.
//

#ifndef DEFINES_H
#define DEFINES_H

#include <random>
#include <ctime>

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

#include <iostream>


auto e = std::mt19937(std::random_device()());

inline double random_u_0_1() {
    return (double) (e() % std::numeric_limits<int>::max()) / (double) std::numeric_limits<int>::max();
}

double random_u_0_1_skew(double skew_rate) {
    return 1 - std::pow(random_u_0_1(),skew_rate);
}

#include <torch/torch.h>
torch::Device GPU_DEVICE = torch::kCUDA;
torch::Device CPU_DEVICE = torch::kCPU;

#include <sys/types.h>
#include <dirent.h>
#include <cstdio>
#include <cerrno>
#include <iostream>
#include <regex>
#include <csignal>

#include <exception>
#include <string>

class MyException : public std::exception {
public:
    MyException() : message("Error.") {}

    explicit MyException(const std::string &str) : message("Error : " + str) {}

    ~MyException() noexcept override = default;

    [[nodiscard]] const char *what() const noexcept override {
        return message.c_str();
    }

private:
    std::string message;
};

std::vector<std::string> scanFiles(std::string inputDirectory) {
    std::vector<std::string> fileList;
    inputDirectory = inputDirectory.append("/");

    DIR *p_dir;
    const char *str = inputDirectory.c_str();

    p_dir = opendir(str);
    if (p_dir == nullptr) {
        std::cout << "can't open :" << inputDirectory << std::endl;
    }

    struct dirent *p_dirent;

    while ((p_dirent = readdir(p_dir))) {
        std::string tmpFileName = p_dirent->d_name;
        if (tmpFileName == "." || tmpFileName == "..") {
            continue;
        } else {
            fileList.push_back(tmpFileName);
        }
    }
    closedir(p_dir);
    return fileList;
}

std::vector<std::string> like_filter(std::vector<std::string> input, const std::string &limit) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!regex_match(input[i], std::regex(".*" + limit + ".*"))) {
            input.erase(input.begin() + (int) i);
            --i;
        }
    }
    return input;
}

auto random_seed() {
    auto ccc = 0;
    for (int j = 0, end = int(e()); j < end; j++) {
        ccc += int(e());
    }
    return ccc;
}

bool IsFileExist(const char* path)
{
    return -1 != access(path, F_OK);
}


template<class CLS>
void load_in(CLS &cls, const std::string &filename = "RunningStatus.bin") {
    auto file = std::fopen(filename.c_str(), "r");
    if (std::fread(&cls, sizeof(CLS), 1, file) == 0) {
        throw MyException("bad loading file");
    }
    std::fclose(file);
}

template<class CLS>
void load_out(CLS &cls, const std::string &filename = "RunningStatus.bin") {
    auto file = std::fopen(filename.c_str(), "w");
    if (std::fwrite(&cls, sizeof(CLS), 1, file) == 0) {
        throw MyException("bad loading file");
    }
    std::fclose(file);
}


#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include "TimerClock.hpp"

template<class T>
T *create_shared_memory(std::string path = "./",int id = 2023) {
    key_t key = ftok("./", 2023);
    if (key == -1) { perror("ftok"); }
    int shared_id = shmget(key, sizeof(T), IPC_CREAT | 0666);
    if (shared_id < 0) { throw MyException("shmget"); }
    auto *pointer = (T*)shmat(shared_id, nullptr, 0);
    if (pointer == nullptr) {throw MyException("shmat");}
    bzero(pointer, sizeof(T)); // set memory to zero
    return pointer;
}

template<class T>
T *get_shared_memory(std::string path = "./", int id = 2023) {
    key_t key = ftok("./", 2023);
    if (key == -1) { perror("ftok"); }
    int shared_id = shmget(key, sizeof(T), IPC_CREAT | 0666);
    if (shared_id < 0) { throw MyException("shmget"); }
    auto *pointer = (T *) shmat(shared_id, nullptr, 0);
    if (pointer == nullptr) { throw MyException("shmat"); }
    return pointer;
}
TimerClock tc;
#endif //DEFINES_H
