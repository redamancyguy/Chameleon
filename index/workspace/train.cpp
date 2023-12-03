//
// Created by wenli on 2023/2/2.
//



//#define show_loss
#include "torch/torch.h"
#include "../../include/DataSet.hpp"

#include <iostream>
#include <unistd.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/mman.h>
#include<functional>
#include<vector>
#include<ctime>
#include "../include/experience.hpp"
#include "../../include/TimerClock.hpp"
#include "../../include/StandardScalar.hpp"
#include "../include/experience.hpp"
#include "../include/RL_network.hpp"

[[noreturn]] void train(){
    auto q_model = std::make_shared<Global_Q_network>(Global_Q_network());
    q_model->to(GPU_DEVICE);
    q_model->train();
    auto q_optimizer = torch::optim::Adam(q_model->parameters(),
                                          torch::optim::AdamOptions(train_lr).
                                                  weight_decay(train_wd));
    int break_times = 10;
    ExpGenerator exp_g;
    RewardScalar scalar(2);

//    torch::load(q_model, q_model_path);
//    scalar.load(q_scalar_path);
//    scalar.to_GPU();
//    q_model->to(GPU_DEVICE);
    for(int steps = 0;;++steps){
        float loss_count = 0;
        for (int flush_count = 0;flush_count < break_times; flush_count++) {
            for (int i = 0; i < train_steps; ++i) {
                auto exp_tensor_batch = exp_g.exp_batch(2 * BATCH_SIZE);
                auto pdf = std::get<0>(exp_tensor_batch).to(GPU_DEVICE);
                auto value = std::get<1>(exp_tensor_batch).to(GPU_DEVICE);
                auto root_fanout = std::get<2>(exp_tensor_batch).to(GPU_DEVICE);
                auto inner_fanout = std::get<3>(exp_tensor_batch).to(GPU_DEVICE);
                auto reward = std::get<4>(exp_tensor_batch).to(GPU_DEVICE);
                auto std_reward  = scalar.forward_and_fit(reward);
                auto pred = q_model->forward(pdf, value, root_fanout, inner_fanout);
                auto loss = torch::nn::L1Loss()->forward(pred,std_reward);
                q_optimizer.zero_grad();
                loss.backward();
                q_optimizer.step();
                fflush(stdout);
                tc.synchronization();
                auto loss_abs = torch::nn::L1Loss()->forward(scalar.inverse(pred),reward).item().toFloat();
                loss_count += loss_abs;
                printf("loss:%f\r",loss_abs);
            }
        }
        printf("loss_count:%f\n",loss_count / float(train_steps * break_times));
        q_model->to(CPU_DEVICE);
        torch::save(q_model, q_model_path);
        scalar.save(q_scalar_path);
        q_model->to(GPU_DEVICE);
        q_optimizer.param_groups()[0].options().set_lr(q_optimizer.param_groups()[0].options().get_lr() * 0.99);
        std::cout <<q_optimizer.param_groups()[0].options().get_lr()<< std::endl;
    }
}
int main(int argc, char const *argv[]) {
    std::vector<int> a;
    GPU_DEVICE = torch::Device(torch::DeviceType::CUDA,1);
    train();
    return 0;
//    if(fork() == 0)

    sleep(20);
//    if(fork() == 0){
//    {
//        break_times = 6;
//        auto pai_model = std::make_shared<PAI_network>(PAI_network());
//        pai_model->to(GPU_DEVICE);
//        pai_model->train();
//        auto q_model = std::make_shared<Q_network>(Q_network());
//        q_model->eval();
//        for(int i = 0;;++i){
//            auto pai_optimizer = torch::optim::Adam(
//                    pai_model->parameters(),
//                    torch::optim::AdamOptions(train_lr*100).
//                            weight_decay(train_wd).betas(std::make_tuple(0.85, 0.98)));
//            sem_wait(q_semaphore);
//            torch::load(q_model, q_model_path);
//            usleep(1000);
//            if(q_model->is_training()){
//                throw MyException("must be eval status !");
//            }
//            sem_post(q_semaphore);
//            ///////////
//            training_PAI(*pai_model,*q_model,pai_optimizer);
//            sem_wait(pai_semaphore);
//            torch::save(pai_model, pai_model_path);
//            std::cout <<"PAI finished:"<<i<< std::endl;
//            usleep(1000);
//            sem_post(pai_semaphore);
//        }
//    }

    while (true) {
        sleep(100);
    }
    return 0;
}