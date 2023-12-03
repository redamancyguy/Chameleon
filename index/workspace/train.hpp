//
// Created by redamancyguy on 23-5-11.
//

#ifndef HITS_TRAIN_HPP
#define HITS_TRAIN_HPP

#include <torch/torch.h>
#include "../../include/DEFINE.h"
#include <iostream>
#include "../../include/Model.hpp"
#include<functional>
#include<vector>
#include<ctime>
#include "../include/experience.hpp"
#include "../../include/TimerClock.hpp"
#include "../../include/StandardScalar.hpp"
#include "../include/experience.hpp"
#include "../include/RL_network.hpp"

int break_times = 10;

#define buffer_size 350

ExpGenerator exp_g;
void training_Q(Global_Q_network &q_model, torch::optim::Adam &q_optimizer) {
    std::cout << "exp-num:" << exp_g.exps.size() << "  " << std::endl;
    float loss_count = 0;
    for (int flush_count = 0;flush_count < break_times; flush_count++) {
        for (int i = 0; i < train_steps; ++i) {
            auto exp_tensor_batch = exp_g.exp_batch(BATCH_SIZE);
            auto pdf = std::get<0>(exp_tensor_batch).to(GPU_DEVICE);
            auto value = std::get<1>(exp_tensor_batch).to(GPU_DEVICE);
            auto root_fanout = std::get<2>(exp_tensor_batch).to(GPU_DEVICE);
            auto inner_fanout = std::get<3>(exp_tensor_batch).to(GPU_DEVICE);
            auto reward = std::get<4>(exp_tensor_batch).to(GPU_DEVICE);
            auto pred = q_model.forward(pdf, value, root_fanout, inner_fanout);
            auto loss = torch::nn::L1Loss()->forward(pred,reward);
//            auto loss = torch::nn::MSELoss()->forward(pred,reward);
            q_optimizer.zero_grad();
            loss.backward();
            q_optimizer.step();
            fflush(stdout);
            tc.synchronization();
            loss_count += loss.to(CPU_DEVICE).item().toFloat();
            printf("loss:%f\r",loss.to(CPU_DEVICE).item().toFloat());
        }
    }
    printf("loss_count:%f\n",loss_count / float(train_steps * break_times));
}


void training_PAI(Global_PAI_network& pai_model, Global_Q_network& q_model,torch::optim::Adam &pai_optimizer) {
    std::cout << "exp-num:" << exp_g.exps.size() << "  " << std::endl;
    for (int flush_count = 0; flush_count < break_times ;flush_count++) {
        float cost_count = 0;
        for (int i = 0; i < train_steps; ++i) {
            auto exp_tensor_batch = exp_g.exp_batch(BATCH_SIZE);
            auto pdf = std::get<0>(exp_tensor_batch).to(GPU_DEVICE);
            auto value = std::get<1>(exp_tensor_batch).to(GPU_DEVICE);
            auto inner_fanout = std::get<2>(exp_tensor_batch).to(GPU_DEVICE);
            auto other_fanout = std::get<3>(exp_tensor_batch).to(GPU_DEVICE);
            std::vector<float> memory_weights;
            std::vector<float> height_weights;
            for (int j = 0; j < BATCH_SIZE; ++j) {
                memory_weights.push_back(random_memory_weight());
                height_weights.push_back(random_height_weight());
            }
            auto weight_tensor = torch::hstack({torch::tensor(memory_weights).view({-1, 1}),
                                                torch::tensor(height_weights).view({-1, 1})}).to(GPU_DEVICE);
            auto pred_fanout = pai_model.forward(pdf,value,weight_tensor);
            auto pred_cost = q_model.forward(pdf,value,pred_fanout.first,pred_fanout.second);
            pred_cost = pred_cost.mul(weight_tensor).mean();
            auto punish_root = -((pred_fanout.first < min_root_fan_out).to(torch::kFloat32).mul(pred_fanout.first - min_root_fan_out)) +
                               (pred_fanout.first > max_root_fan_out).to(torch::kFloat32).mul(pred_fanout.first - max_root_fan_out);
            auto punish_inner = -((pred_fanout.second < min_inner_fan_out).to(torch::kFloat32).mul(pred_fanout.second - min_inner_fan_out)) +
                                (pred_fanout.second > max_inner_fan_out).to(torch::kFloat32).mul(pred_fanout.second - max_inner_fan_out);
            punish_root = punish_root.mean();
            punish_inner = punish_inner.mean();
            auto cost = (pred_cost + (punish_root + ((min_root_fan_out + max_root_fan_out) / (min_inner_fan_out + max_inner_fan_out)) * punish_inner));
            pai_optimizer.zero_grad();
            cost.backward();
            pai_optimizer.step();
            cost_count += pred_cost.to(CPU_DEVICE).item().toFloat();
            printf("%s calc_t:[%.5fms]punish_root %f punish_inner %.6f root_mean %.6f inner_mean %f pred_cost %f %s\r", GREEN, tc.get_timer_milliSec(),
                   punish_root.to(CPU_DEVICE).item().toFloat() ,
                   punish_inner.to(CPU_DEVICE).item().toFloat() ,
                   pred_fanout.first.mean().to(CPU_DEVICE).item().toFloat() ,
                   pred_fanout.second.mean().to(CPU_DEVICE).item().toFloat() ,
                   cost_count / float(i + 1) , RESET);
            fflush(stdout);
            tc.synchronization();
        }
    }
}


#endif //HITS_TRAIN_HPP
