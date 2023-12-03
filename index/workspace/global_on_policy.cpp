//
// Created by redamancyguy on 23-5-18.
//
#include <iostream>
#include <unistd.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/mman.h>
#define using_semaphore
#ifdef using_semaphore
sem_t* q_semaphore;
sem_t* pai_semaphore;
#endif
#include "../../include/DEFINE.h"
#include <c10/cuda/CUDAGuard.h>
//#define CB
//#define using_small_network
#include "gen_exp.hpp"

int main(int argc, char const *argv[]) {

#ifdef using_semaphore
    q_semaphore = (sem_t*)mmap(nullptr, sizeof(sem_t), PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANON, -1, 0);
    pai_semaphore = (sem_t*)mmap(nullptr, sizeof(sem_t), PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANON, -1, 0);
    sem_init(q_semaphore, 1, 1);
    sem_init(pai_semaphore, 1, 1);
#endif
//    double random_rate_discount_rate = 0.997;
    double random_rate_discount_rate = 0.99;
    auto *rs = create_shared_memory<RunningStatus>();
    rs->random_rate = 1;
    std::cout << "rs->random_rate:" <<rs->random_rate << std::endl;
    const int process_count = 4;
    const int process_count2= 2;
    const std::size_t sample_batch = 33;
    clear_exp(experience_father_path,scanFiles(experience_father_path));
//    clear_exp(model_father_path,scanFiles(model_father_path));
    rs->exp_num = max_exp_number(scanFiles(experience_father_path)) + 1;
    std::cout << "start_exp:" << rs->exp_num << std::endl;
    std::cout << "process_count:" << process_count << std::endl;
    /////////////////////////
    GPU_DEVICE = torch::Device(torch::DeviceType::CUDA,1);

    int pid = 0;
    for ( auto i = 0; i < process_count; i++) {
        random_seed();
        if ((pid = fork()) == 0) {
            random_seed();
            gen(i + 1, sample_batch);
        }
        std::cout << "gen pid : " << pid << std::endl;
        sleep(1);
        random_seed();
    }
    random_seed();
    GPU_DEVICE = torch::Device(torch::DeviceType::CUDA,0);
    for (auto i = 0; i < process_count2; i++) {
        random_seed();
        if ((pid = fork()) == 0) {
            random_seed();
            gen(process_count + i + 1, sample_batch);
        }
        std::cout << "gen pid : " << pid << std::endl;
        sleep(1);
        random_seed();
    }
    std::cout << "train pid: " << pid << std::endl;
    while(count_exp(scanFiles(experience_father_path)) < 100){ puts("waiting for more samples !");sleep(10);}
    break_times = 10;
    auto q_model = std::make_shared<Global_Q_network>(Global_Q_network());
    q_model->to(GPU_DEVICE);
    q_model->train();
    auto q_optimizer = torch::optim::Adam(q_model->parameters(),
                                          torch::optim::AdamOptions(train_lr).
                                                  weight_decay(train_wd));

    std::size_t sample_count = 0;
    while (rs->random_rate > 0) {
        std::cout << "random rate:" << "  " << std::setprecision(30)
                  << rs->random_rate << std::setprecision(6) << std::endl;
        if (sample_count + sample_batch + BATCH_SIZE < count_exp(scanFiles(experience_father_path))) {
            std::cout << "writing shared memory:" << rs->random_rate << std::endl;
            rs->random_rate *= random_rate_discount_rate;
            sample_count += sample_batch;
            training_Q(*q_model,q_optimizer);
            sem_wait(q_semaphore);
            q_model->to(CPU_DEVICE);
            torch::save(q_model, q_model_path);
            q_model->to(GPU_DEVICE);
            std::cout <<GREEN<<"Q finished:"<<sample_count / sample_batch<<RESET<< std::endl;
            sem_post(q_semaphore);
            continue;
        }
        sleep(10);
    }

#ifdef using_semaphore
    sem_close(q_semaphore);
    sem_destroy(q_semaphore);
    munmap(q_semaphore, sizeof(sem_t));
    q_semaphore = nullptr;
    sem_close(pai_semaphore);
    sem_destroy(pai_semaphore);
    munmap(pai_semaphore, sizeof(sem_t));
    pai_semaphore = nullptr;
#endif
    return 0;
}