//
// Created by wenli on 2023/1/26.
//

#ifndef EXPERIENCE_HPP
#define EXPERIENCE_HPP


#include <unistd.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "../../include/DEFINE.h"
#include "../../include/DataSet.hpp"
#include "Configuration.hpp"
#include "Parameter.h"

class Cost {
public:
    float memory;
    float add;
    float get;
    float erase;

    Cost() {
        memory = 0;
        add = 0;
        get = 0;
        erase = 0;
    }

    friend std::ostream &operator<<(std::ostream &out, Cost &input)
    {
        out << " " << std::setprecision(12) << input.memory <<std::setprecision(6)<< " : "
            << std::setw(6) << input.add << " : " << std::setw(6) << input.get << " : " << std::setw(6) << input.erase;
        return out;
    }
};


class experience_t {
public:
    float distribution[BUCKET_SIZE]{};
    float data_size;
    CHA::Configuration conf;
    Cost cost;

public:
    experience_t() {
        for (auto &i: distribution) {
            i = 0;
        }
        data_size = 0;
        conf.root_fan_out = 0;
        for (auto &i: conf.fan_outs) {
            for (auto &j: i) {
                j = 0;
            }
        }
        cost.memory = 0;
        cost.add = 0;
        cost.get = 0;
        cost.erase = 0;
    }

    friend std::ostream &operator<<(std::ostream &out, experience_t &input)
    {
        out << input.conf << " ";
        out << input.cost << " ";
        out << std::endl;
        return out;
    }

    bool operator<(const experience_t &another) const {
        return this->conf < another.conf;
    }

    bool operator>(const experience_t &another) const {
        return this->conf > another.conf;
    }

    bool operator==(const experience_t &another) const {
        return this->conf == another.conf;
    }
};

int clear_exp(const std::string &parent_path, const std::vector<std::string> &files) {
    for (auto &i: files) {
        if (remove((parent_path + i).c_str()) == 0)
            printf("Removed %s.", i.c_str());
        else
            perror("remove");
    }
    return 0;
}

int max_exp_number(const std::vector<std::string> &files) {
    int result = 0;
    for (auto &i: files) {
        result = std::max(result, std::stoi(i.substr(0, i.size() - 4)));
    }
    return result;
}

std::size_t count_exp(const std::vector<std::string> &files) {
    std::size_t result = 0;
    for (auto &i: files) {
        std::FILE *exp_file = std::fopen((experience_father_path + i).c_str(), "rb");
        std::fseek(exp_file, 0, SEEK_END);
        result += std::ftell(exp_file) / sizeof(experience_t);
        std::fclose(exp_file);
    }
    return result;
}

std::vector<experience_t> read_exp(const std::vector<std::string> &files) {
    std::vector<experience_t> all_exps;
    auto all_size = int(count_exp(files));
    all_exps.resize(all_size);
    std::size_t cursor = 0;
    for (auto &i: files) {
        std::FILE *exp_file = std::fopen((experience_father_path + i).c_str(), "rb");
        std::fseek(exp_file, 0, SEEK_END);
        std::size_t size = std::ftell(exp_file) / sizeof(experience_t);
        std::fseek(exp_file, 0, SEEK_SET);
        if (std::fread(all_exps.data() + cursor, sizeof(experience_t), size, exp_file) == 0) {
            std::fclose(exp_file);
            continue;
        }
        std::fclose(exp_file);
        cursor += size;
    }
    if (cursor != all_exps.size()) {
        if (cursor < all_exps.size()) {
            all_exps.resize(cursor);
        }
    }
    return all_exps;
}

std::pair<std::vector<experience_t>, std::vector<experience_t>>
read_exp_and_split(const std::vector<std::string> &files) {
    auto temp_e = e;
    temp_e.seed(1000);
    std::vector<experience_t> all_exps;
    std::vector<experience_t> train;
    std::vector<experience_t> test;
    auto all_size = int(count_exp(files));
    auto test_size = int(0.25 * all_size / BATCH_SIZE) * BATCH_SIZE + BATCH_SIZE;
    if (all_size < test_size) {
        return {train, test};
    }
    all_exps.resize(all_size);
    std::size_t cursor = 0;
    for (auto &i: files) {
        std::cout << "loading:" << i << std::endl;
        std::FILE *exp_file = std::fopen((experience_father_path + i).c_str(), "rb");
        std::fseek(exp_file, 0, SEEK_END);
        std::size_t size = std::ftell(exp_file) / sizeof(experience_t);
        std::fseek(exp_file, 0, SEEK_SET);
        if (std::fread(all_exps.data() + cursor, sizeof(experience_t), size, exp_file) == 0) {
            std::cout << "zero:" << i << std::endl;
            continue;
        }
        cursor += size;
        std::fclose(exp_file);
    }
    if (cursor != all_exps.size()) {
        if (cursor < all_exps.size()) {
            all_exps.resize(cursor);
        }
    }
    std::vector<int> indices;
    for (int i = 0; i < int(all_exps.size()); ++i) {
        indices.push_back(i);
    }
    std::shuffle(indices.begin(), indices.end(), temp_e);
    for (int i = 0; i < test_size; ++i) {
        test.push_back(all_exps[indices[i]]);
    }
    for (int i = test_size; i < int(all_exps.size()); ++i) {
        train.push_back(all_exps[indices[i]]);
    }
    return {train, test};
}


#include <regex>
#include "../../include/TimerClock.hpp"
#include "../../include/StandardScalar.hpp"

class ExpGenerator {
public:
    std::vector<experience_t> exps;
    TimerClock tc_;
    void check_nan(){
        for(auto &i:exps){
            for(int j = 0;j<BUCKET_SIZE;++j){
                if(std::isnan(i.distribution[j])){
                    throw MyException("nan value 1!");
                }
            }
            for(int j = 0;j<INNER_FANOUT_ROW;++j){
                for(int k = 0;k<INNER_FANOUT_COLUMN;++k){
                    if(std::isnan(i.conf.fan_outs[j][k])){
                        throw MyException("nan value 2!");
                    }
                }
            }
            if(std::isnan(i.conf.root_fan_out)){
                throw MyException("nan value 3!");
            }
            if(std::isnan(i.data_size)){
                throw MyException("nan value 4!");
            }
            if(std::isnan(i.cost.memory)){
                throw MyException("nan value 5!");
            }
            if(std::isnan(i.cost.get)){
                throw MyException("nan value 6!");
            }
        }
    }
    ExpGenerator(){
        auto files = scanFiles(experience_father_path);
        exps = read_exp(files);
        check_nan();
        tc_.synchronization();
    }
    inline static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    experience_to_tensor(const std::vector<experience_t> &exps) {
        auto result = std::make_tuple(
                torch::rand({int(exps.size()), PDF_SIZE}),
                torch::rand({int(exps.size()), VALUE_SIZE}),
                torch::rand({int(exps.size()), 1}),
                torch::rand({int(exps.size()), INNER_FANOUT_ROW, INNER_FANOUT_COLUMN}),
                torch::rand({int(exps.size()), REWARD_SIZE})
        );
        auto dis_state_ptr = std::get<0>(result).data_ptr<float>();
        auto v_state_ptr = std::get<1>(result).data_ptr<float>();
        auto root_fanout_ptr = std::get<2>(result).data_ptr<float>();
        auto inner_fanout_ptr = std::get<3>(result).data_ptr<float>();
        auto reward_ptr = std::get<4>(result).data_ptr<float>();
        for (auto &i:exps) {
            std::copy(i.distribution, i.distribution + BUCKET_SIZE, dis_state_ptr);
            dis_state_ptr += BUCKET_SIZE;
            ///////////////////////////
            v_state_ptr[0] = i.data_size;
            v_state_ptr += VALUE_SIZE;
            ////////////////////////////
            std::copy((float *) i.conf.fan_outs, ((float *) (i.conf.fan_outs) + INNER_FANOUT_SIZE), inner_fanout_ptr);
            inner_fanout_ptr += INNER_FANOUT_SIZE;
            root_fanout_ptr[0] = i.conf.root_fan_out ;
            root_fanout_ptr += 1;
            reward_ptr[0] = i.cost.memory;
            reward_ptr[1] = i.cost.get;
            reward_ptr += REWARD_SIZE;
        }
        return result;
    }

    inline
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    exp_batch(int batch_size) {
        std::vector<experience_t> result(batch_size);
        for (int i = 0;i<batch_size;++i) {
            result[i] = exps[e() % exps.size()];
            if(tc_.get_timer_second() > 30){
                auto files = scanFiles(experience_father_path);
                exps = read_exp(files);
                std::shuffle(exps.begin(), exps.end(),e);
                tc_.synchronization();
                check_nan();
                std::cout <<"exp num:"<<exps.size()<<" check using:"<<tc_.get_timer_second()<< std::endl;
                tc_.synchronization();
            }
        }
        return experience_to_tensor(result);
    }
};

void sort_all_dataset(){
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    for(auto &dataset_name:dataset_names){
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
                data_father_path + dataset_name);
        std::sort(dataset.begin(), dataset.end());
        dataset_source::set_dataset(data_father_path + dataset_name,dataset);
    }
}

void shuffle_all_dataset(){
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    for(auto &dataset_name:dataset_names){
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(
                data_father_path + dataset_name);
        std::shuffle(dataset.begin(), dataset.end(),e);
        dataset_source::set_dataset(data_father_path + dataset_name,dataset);
    }
}



#endif //EXPERIENCE_HPP
