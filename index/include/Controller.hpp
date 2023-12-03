//
// Created by redamancyguy on 23-4-5.
//

#ifndef HITS_CONTROLLER_HPP
#define HITS_CONTROLLER_HPP

#include <iostream>
#include<functional>
#include<vector>
#include <set>
#include "../../include/DEFINE.h"
#include "experience.hpp"
#include "../../include/TimerClock.hpp"
#include "RL_network.hpp"
#include <boost/math/distributions/normal.hpp>

class GlobalController {
public:
    float memory_weight = MEMORY_WEIGHT;
    float query_weight = QUERY_WEIGHT;
    std::shared_ptr<Global_Q_network> q_model;
    std::shared_ptr<RewardScalar> scalar;
public:
    explicit GlobalController()  {
        q_model = std::make_shared<Global_Q_network>(Global_Q_network());
        scalar = std::make_shared<RewardScalar>(RewardScalar(2));
        q_model->eval();
        q_model->to(GPU_DEVICE);
    }

    void load_in(){
        if (IsFileExist(q_model_path.c_str())) {
            torch::load(q_model, q_model_path);
            scalar->load(q_scalar_path);
            q_model->to(GPU_DEVICE);
        } else {
            scalar->to_GPU();
            q_model->to(GPU_DEVICE);
        }

    }

    void random_weight() {
        memory_weight = random_memory_weight();
        query_weight = random_height_weight();
    }

    static float rate_shrink(float rate, float shrink_rate = 0.95) {
        return rate * shrink_rate;
    }

    static float size_shrink(float size, float shrink_rate = 0.95) {
        return size * shrink_rate;
    }

    [[nodiscard]] float reward_func(const Cost &conf) const {
        return -(conf.memory * memory_weight +
                 conf.get * query_weight);
    }

    [[nodiscard]] experience_t get_best_action_GA(experience_t buffer)const {
        std::vector<experience_t> gens;
        float mutation_rate = 0.5;
        float mutation_size_root = max_root_fan_out / 256;
        float mutation_size_inner = max_inner_fan_out / 256;


        int eq = 0;
        int max_gen = MAX_GEN;
        int eq_max = 3;
        int min_iteration = 10;
        float last_reward = -std::numeric_limits<float>::max();
        for (int _ = 0; ; _++) {
            mutation_rate = rate_shrink(mutation_rate);
            mutation_size_root = size_shrink(mutation_size_root);
            mutation_size_inner = size_shrink(mutation_size_inner);

            for (int i = 0; i < max_gen; i++) {
                buffer.conf = CHA::Configuration::random_configuration();
                gens.push_back(buffer);
            }
            for (int i = 0; i < max_gen; i++) {
                for (int ii = 0; ii < 3; ii++) {
                    switch (e() % 4) {
                        case 0: {
                            buffer.conf.root_fan_out = shrink_root_fan_out(
                                    gens[i].conf.root_fan_out * (1+mutation_rate));
                            break;
                        }
                        case 1: {
                            buffer.conf.root_fan_out = shrink_root_fan_out(
                                    gens[i].conf.root_fan_out * (1-mutation_rate));
                            break;
                        }
                        case 2: {
                            buffer.conf.root_fan_out = shrink_root_fan_out(
                                    gens[i].conf.root_fan_out + mutation_size_root);
                            break;
                        }
                        case 3: {
                            buffer.conf.root_fan_out = shrink_root_fan_out(
                                    gens[i].conf.root_fan_out - mutation_size_root);
                            break;
                        }
                    }
                    for (int j = 0; j < INNER_FANOUT_ROW; j++) {
                        for (int k = 0; k < INNER_FANOUT_COLUMN; k++) {
                            switch (e() % 4) {
                                case 0: {
                                    buffer.conf.fan_outs[j][k] =
                                            shrink_inner_fan_out(
                                                    gens[i].conf.fan_outs[j][k] * (1+mutation_rate));
                                    break;
                                }
                                case 1: {
                                    buffer.conf.fan_outs[j][k] =
                                            shrink_inner_fan_out(
                                                    gens[i].conf.fan_outs[j][k] * (1-mutation_rate));
                                    break;
                                }
                                case 2: {
                                    buffer.conf.fan_outs[j][k] =
                                            shrink_inner_fan_out(
                                                    gens[i].conf.fan_outs[j][k] + mutation_size_inner);
                                    break;
                                }
                                case 3: {
                                    buffer.conf.fan_outs[j][k] =
                                            shrink_inner_fan_out(
                                                    gens[i].conf.fan_outs[j][k] - mutation_size_inner);
                                    break;
                                }
                            }
                        }
                    }
                    gens.push_back(buffer);
                }
            }
            for (auto extract_batch: std::vector<int>({2, 3, 4})) {
                for (int i = 0; i < max_gen - extract_batch; ++i) {
                    buffer.conf.root_fan_out = shrink_root_fan_out(
                            gens[(e() % extract_batch) + i].conf.root_fan_out);
                    for (int j = 0; j < INNER_FANOUT_ROW; j++) {
                        for (int k = 0; k < INNER_FANOUT_COLUMN; k++) {
                            buffer.conf.fan_outs[j][k] = shrink_inner_fan_out(
                                    gens[(e() % extract_batch) + i].conf.fan_outs[j][k]);
                        }
                    }
                    gens.push_back(buffer);
                }
            }
            for (auto extract_batch: std::vector<int>({2, 3, 4})) {
                for (int i = 0; i < max_gen - extract_batch; ++i) {
                    auto mean = CHA::Configuration::zeros();
                    auto var = CHA::Configuration::zeros();
                    for (int ii = 0; ii < extract_batch; ii++) {
                        mean = mean + gens[ii + i].conf;
                    }
                    mean = mean / float(extract_batch);
                    for (int ii = 0; ii < extract_batch; ii++) {
                        auto tmp = gens[ii + i].conf - mean;
                        var = var + (tmp * tmp);
                    }
                    var = var / float(extract_batch - 1);
                    var = var.sqrt_invert();
                    buffer.conf.root_fan_out = shrink_root_fan_out( float(boost::math::quantile(boost::math::normal_distribution<>(mean.root_fan_out, var.root_fan_out), 0.001 + random_u_0_1() * 0.998)));
                    for(int j = 0; j < INNER_FANOUT_ROW; ++j){
                        for(int jj = 0; jj < INNER_FANOUT_COLUMN; ++jj){
                            buffer.conf.fan_outs[j][jj] = shrink_inner_fan_out(float(boost::math::quantile(boost::math::normal_distribution<>(mean.fan_outs[j][jj], var.fan_outs[j][jj]), 0.001 + random_u_0_1() * 0.998)));
                        }
                    }
                    gens.push_back(buffer);
                }
            }
            auto exp_tensor_batch = ExpGenerator::experience_to_tensor(gens);
            auto pdf = std::get<0>(exp_tensor_batch).to(GPU_DEVICE);
            auto value = std::get<1>(exp_tensor_batch).to(GPU_DEVICE);
            auto root_fanout = std::get<2>(exp_tensor_batch).to(GPU_DEVICE);
            auto inner_fanout = std::get<3>(exp_tensor_batch).to(GPU_DEVICE);
            auto pred = torch::Tensor();
            {
                torch::NoGradGuard no_grad;
                pred = q_model->forward(pdf, value, root_fanout, inner_fanout);
                pred = scalar->inverse(pred).to(CPU_DEVICE);
            }
            auto reward_arr = pred.data_ptr<float>();
            for (auto &gen: gens) {
                gen.cost.memory = reward_arr[0];
                gen.cost.get = reward_arr[1];
                reward_arr += REWARD_SIZE;
            }
            std::sort(gens.begin(), gens.end(), [=](experience_t &a, experience_t &b) {
                return reward_func(a.cost) > reward_func(b.cost);
            });
            if (reward_func(gens[0].cost) == last_reward) {
                ++eq;
                if (eq >= eq_max && _ > min_iteration) {
                    break;
                }
            } else {
                last_reward = reward_func(gens[0].cost);
                eq = 0;
            }
            gens.resize(max_gen);
        }
        return gens[0];
    }
};

#endif //HITS_CONTROLLER_HPP