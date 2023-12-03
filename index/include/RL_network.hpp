//
// Created by redamancyguy on 23-8-2.
//

#ifndef HITS_Q_MODEL_H
#define HITS_Q_MODEL_H

#include <utility>

#include "../../include/Model.hpp"
#include "../../include/StandardScalar.hpp"
#include "experience.hpp"

#include "../include/Parameter.h"
#define fc_neuron_size 256
std::string q_scalar_path = model_father_path + "AC" + "_Q_Scalar.pt";
std::string q_model_path = model_father_path + "AC" + "_Q_Net.pt";
std::string pai_model_path = model_father_path + "AC" + "_PAI_Net.pt";


#define weight_skew 2

float random_memory_weight() {
    return float(std::pow(random_u_0_1(),weight_skew) * float(weight_skew+1) + 0.03);
}

float random_height_weight() {
    return float(std::pow(random_u_0_1(),weight_skew) * float(weight_skew+1) + 0.03);
}

#define min_scalar_eps float(1e-5)
#define min_var float(1e-7)

class RewardScalar {
public:
    torch::Tensor mean;
    torch::Tensor var;
    torch::Tensor std_var;
    float scalar_eps = 1;

    explicit RewardScalar(int size) {
        mean = torch::zeros({size}).to(GPU_DEVICE);
        var = torch::ones({size}).to(GPU_DEVICE);
        std_var = torch::sqrt(var).to(GPU_DEVICE);
    }

    void to_CPU() {
        mean = mean.to(CPU_DEVICE);
        var = var.to(CPU_DEVICE);
        std_var = std_var.to(CPU_DEVICE);
    }

    void to_GPU() {
        mean=  mean.to(GPU_DEVICE);
        var = var.to(GPU_DEVICE);
        std_var = std_var.to(GPU_DEVICE);
    }

    torch::Tensor forward_and_fit(torch::Tensor &x) {
        mean = mean * (float(1.0) - scalar_eps) + scalar_eps * x.mean(0);
        /////////////
        var = var * (float(1.0) - scalar_eps) + scalar_eps * x.var(0);
        std_var = torch::sqrt(var);
        if (scalar_eps > min_scalar_eps) { scalar_eps *= 0.97; }
        return forward(x);
    }

    torch::Tensor forward(torch::Tensor &x) const {
        return (x - mean) / std_var;
    }

    torch::Tensor inverse(torch::Tensor &x) const {
        return std_var * x + mean;
    }

    void save(const std::string &model_name) {
        mean = mean.to(CPU_DEVICE);
        var = var.to(CPU_DEVICE);
        std::ofstream file(model_name);
        file << std::setprecision(40);
        for (auto i: std::vector<float>(mean.data_ptr<float>(),
                                        mean.data_ptr<float>() + mean.numel())) {
            file << i << " ";
        }
        for (auto i: std::vector<float>(var.data_ptr<float>(),
                                        var.data_ptr<float>() + var.numel())) {
            file << i << " ";
        }
        file.close();
        mean = mean.to(GPU_DEVICE);
        var = var.to(GPU_DEVICE);
    }

    void load(const std::string &model_name) {
        mean = mean.to(CPU_DEVICE);
        var = var.to(CPU_DEVICE);
        if (IsFileExist((model_name).c_str())) {
            std::ifstream file(model_name);
            for (int i = 0; i < mean.numel(); ++i) {
                float t;
                file >> t;
                mean.data_ptr<float>()[i] = t;
            }
            for (int i = 0; i < var.numel(); ++i) {
                float t;
                file >> t;
                var.data_ptr<float>()[i] = t;
            }
            file.close();
        } else {
            mean = torch::zeros(mean.sizes());
            var = torch::ones(var.sizes());
        }
        mean = mean.to(GPU_DEVICE);
        var = var.to(GPU_DEVICE);
        std_var = torch::sqrt(var);
    }
};


class Global_Q_network : public torch::nn::Module {
private:
    ////////////////////////
    std::vector<int> pdf_cnn_channel = {1,16, 32,64, 128,192,256};
    std::vector<int> pdf_cnn_kernel = {6, 6, 7, 6, 4, 3};
    std::vector<int> pdf_cnn_stride = {5, 5, 5, 5, 2, 2};
    std::vector<int> pdf_cnn_padding = {1, 2, 1, 0, 1, 0};
    std::vector<int> pdf_pool_kernel = {2, 2, 2, 2, 3, 2};
    std::vector<int> pdf_pool_stride = {1, 1, 1, 1, 2, 1};
    std::vector<int> pdf_pool_padding = {1, 0, 1, 1, 0, 0};
    std::shared_ptr<CV_1D_STACK> pdf_cnn_stack;
    ///////////////////////////////////////////////////////
    std::vector<int> fanout_cnn_channel = {1, 16, 32, 64, 128};
    std::vector<int> fanout_cnn_kernel_x = {2, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int> fanout_cnn_stride_x = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int> fanout_cnn_padding_x = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> fanout_pool_kernel_x = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int> fanout_pool_stride_x = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int> fanout_pool_padding_x = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> fanout_cnn_kernel_y =   {4, 4, 4, 3};
    std::vector<int> fanout_cnn_stride_y =   {3, 3, 3, 2};
    std::vector<int> fanout_cnn_padding_y =  {0, 0, 0, 0};
    std::vector<int> fanout_pool_kernel_y =  {2, 2, 2, 2};
    std::vector<int> fanout_pool_stride_y =  {1, 1, 1, 1};
    std::vector<int> fanout_pool_padding_y = {1, 1, 1, 0};
    std::shared_ptr<CV_2D_STACK> fanout_cnn_stack;
    ////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    std::vector<int> fc_sizes = {fc_neuron_size, fc_neuron_size};
    std::vector<std::shared_ptr<LN>> fcs;
    torch::nn::Linear fc_out = nullptr;
    torch::nn::BatchNorm1d root_fanout_bn = nullptr;
    torch::nn::BatchNorm2d inner_fanout_bn = nullptr;
    torch::nn::BatchNorm1d value_bn = nullptr;
public:
    Global_Q_network() {
        int pdf_size = PDF_SIZE;
        pdf_cnn_stack = register_module(
                "pdf_cnn",
                std::make_shared<CV_1D_STACK>(CV_1D_STACK(pdf_cnn_channel,
                                                          pdf_cnn_kernel,
                                                          pdf_cnn_stride,
                                                          pdf_cnn_padding,
                                                          pdf_pool_kernel,
                                                          pdf_pool_stride,
                                                          pdf_pool_padding,
                                                          pdf_size
                )));
        int fanout_row = INNER_FANOUT_ROW + 1;
        int fanout_column = INNER_FANOUT_COLUMN;
        fanout_cnn_stack = register_module(
                "fanout_cnn",
                std::make_shared<CV_2D_STACK>(CV_2D_STACK(fanout_cnn_channel,
                                                          fanout_cnn_kernel_x,
                                                          fanout_cnn_stride_x,
                                                          fanout_cnn_padding_x,
                                                          fanout_pool_kernel_x,
                                                          fanout_pool_stride_x,
                                                          fanout_pool_padding_x,
                                                          fanout_row,
                                                          fanout_cnn_kernel_y,
                                                          fanout_cnn_stride_y,
                                                          fanout_cnn_padding_y,
                                                          fanout_pool_kernel_y,
                                                          fanout_pool_stride_y,
                                                          fanout_pool_padding_y,
                                                          fanout_column
                )));
        root_fanout_bn = register_module("root_fanout_bn", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1)));
        inner_fanout_bn = register_module("inner_fanout_bn", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1)));
        value_bn = register_module("value_bn", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1)));
        int out_size = pdf_size * pdf_cnn_channel.back() + VALUE_SIZE + fanout_row * fanout_column * fanout_cnn_channel.back();
        for (std::size_t i = 0; i < fc_sizes.size(); i++) {
            fcs.push_back(register_module("fc_" + std::to_string(i),
                                          std::make_shared<LN>(LN(out_size, fc_sizes[i]))));
            out_size = fc_sizes[i];
        }
        fc_out = register_module("fcs", torch::nn::Linear(out_size, REWARD_SIZE));
    }

    auto
    forward(torch::Tensor pdf, torch::Tensor value, torch::Tensor root_fanout, torch::Tensor inner_fanout) {
        auto sample_number = pdf.size(0);
        pdf = pdf.view({pdf.size(0), 1, pdf.size(1)});
        pdf = pdf_cnn_stack->forward(pdf);
        pdf = pdf.view({sample_number, -1});
        root_fanout = root_fanout.view({sample_number, 1, root_fanout.size(1)});
        root_fanout = root_fanout_bn->forward(root_fanout);
        root_fanout = root_fanout.view({sample_number, 1, root_fanout.size(2), 1});
        root_fanout = root_fanout.mul(
//                torch::ones({sample_number, 1, root_fanout.size(2), INNER_FANOUT_COLUMN}).to(torch::Device(torch::DeviceType::CUDA,1)));
                torch::ones({sample_number, 1, root_fanout.size(2), INNER_FANOUT_COLUMN}).to(GPU_DEVICE));
        inner_fanout = inner_fanout.view({sample_number, 1, inner_fanout.size(1), inner_fanout.size(2)});
        inner_fanout = inner_fanout_bn->forward(inner_fanout);
        auto all_fanout = torch::cat({root_fanout, inner_fanout}, 2);
        all_fanout = fanout_cnn_stack->forward(all_fanout);
        all_fanout = all_fanout.view({sample_number, -1});//decrease one dimension
        ////////////////////////
        value = value.view({sample_number, 1, value.size(1)});
        value = value_bn->forward(value);
        value = value.view({sample_number, 1});
        auto concat = torch::hstack({pdf, all_fanout, value});
        for (auto &fc: fcs) {
            concat = fc->forward(concat);
        }
        concat = fc_out->forward(concat);
        return concat;
    }
};

std::vector<float> action_space = {0, 1, 2, 3, 4, 5 ,6, 7, 8};
//std::vector<float> action_space = {0, 1, 2, 3, 4, 5 ,6, 7, 8,9,10};
class Small_Q_network: public torch::nn::Module {
    std::vector<int> pdf_cnn_channel =  {1,  16,  32, 64};
    std::vector<int> pdf_cnn_kernel =   {15, 15, 5};
    std::vector<int> pdf_cnn_stride =   {1,  1,  1};
    std::vector<int> pdf_cnn_padding =  {7,  7,  0};
    std::vector<int> pdf_pool_kernel =  {12, 11, 1};
    std::vector<int> pdf_pool_stride =  {8,  8,  1};
    std::vector<int> pdf_pool_padding = {6,  5,  0};
    std::vector<std::shared_ptr<LN>> fcs;
//    std::vector<int> fcs_sizes = {fc_neuron_size * 2,fc_neuron_size * 4,fc_neuron_size * 2,fc_neuron_size};
//    std::vector<int> fcs_sizes = {fc_neuron_size * 2,fc_neuron_size};
    std::vector<int> fcs_sizes = {fc_neuron_size};
    torch::nn::Linear fc_out = nullptr;
    std::shared_ptr<CV_1D_STACK> pdf_cnn_stack;
public:
    Small_Q_network() {
        int pdf_size = SMALL_PDF_SIZE;
        pdf_cnn_stack = register_module(
                "pdf_cnn",
                std::make_shared<CV_1D_STACK>(CV_1D_STACK(pdf_cnn_channel,
                                                          pdf_cnn_kernel,
                                                          pdf_cnn_stride,
                                                          pdf_cnn_padding,
                                                          pdf_pool_kernel,
                                                          pdf_pool_stride,
                                                          pdf_pool_padding,
                                                          pdf_size
                )));
        int out_size = pdf_size * pdf_cnn_channel.back() + 1 + 1;
        for (std::size_t i = 0; i < fcs_sizes.size(); i++) {
            fcs.push_back(register_module("fc_" + std::to_string(i),
                                                      std::make_shared<LN>(LN(out_size, fcs_sizes[i]))));
            out_size = fcs_sizes[i];
        }
        fc_out = register_module("root_fanout_fc_out", torch::nn::Linear(out_size, 1));
    }
    auto
    forward(torch::Tensor pdf, torch::Tensor value, torch::Tensor action) {
        pdf = pdf.view({pdf.size(0),1,pdf.size(1)});
        pdf = pdf_cnn_stack->forward(pdf);
        pdf = pdf.view({pdf.size(0),-1});
        auto concat = torch::hstack({std::move(pdf), std::move(value),std::move(action)});
        for (auto &fc: fcs) {
            concat = fc->forward(concat);
        }
        return fc_out->forward(concat);
    }
};


class Small_PAI_network: public torch::nn::Module {
    std::vector<int> pdf_cnn_channel =  {1,  16,  32, 64};
    std::vector<int> pdf_cnn_kernel =   {15, 15, 5};
    std::vector<int> pdf_cnn_stride =   {1,  1,  1};
    std::vector<int> pdf_cnn_padding =  {7,  7,  0};
    std::vector<int> pdf_pool_kernel =  {12, 11, 1};
    std::vector<int> pdf_pool_stride =  {8,  8,  1};
    std::vector<int> pdf_pool_padding = {6,  5,  0};
    std::vector<std::shared_ptr<LN>> fcs;
    std::vector<int> fcs_sizes = {fc_neuron_size};
    torch::nn::Linear fc_out = nullptr;
    std::shared_ptr<CV_1D_STACK> pdf_cnn_stack;
public:
    Small_PAI_network() {
        int pdf_size = SMALL_PDF_SIZE;
        pdf_cnn_stack = register_module(
                "pdf_cnn",
                std::make_shared<CV_1D_STACK>(CV_1D_STACK(pdf_cnn_channel,
                                                          pdf_cnn_kernel,
                                                          pdf_cnn_stride,
                                                          pdf_cnn_padding,
                                                          pdf_pool_kernel,
                                                          pdf_pool_stride,
                                                          pdf_pool_padding,
                                                          pdf_size
                )));
        int out_size = pdf_size * pdf_cnn_channel.back() + 1;
        for (std::size_t i = 0; i < fcs_sizes.size(); i++) {
            fcs.push_back(register_module("fc_" + std::to_string(i),
                                          std::make_shared<LN>(LN(out_size, fcs_sizes[i]))));
            out_size = fcs_sizes[i];
        }
        fc_out = register_module("root_fanout_fc_out", torch::nn::Linear(out_size, 1));
    }
    auto
    forward(torch::Tensor pdf, torch::Tensor value) {
        pdf = pdf.view({pdf.size(0),1,pdf.size(1)});
        pdf = pdf_cnn_stack->forward(pdf);
        pdf = pdf.view({pdf.size(0),-1});
        auto concat = torch::hstack({std::move(pdf), std::move(value)});
        for (auto &fc: fcs) {
            concat = fc->forward(concat);
        }
        return fc_out->forward(concat);
    }
};

class Global_PAI_network : public torch::nn::Module {
private:
    ////////////////////////
    std::vector<int> pdf_cnn_channel =  {1, 8, 16, 32, 64, 128, 256};
    std::vector<int> pdf_cnn_kernel =   {15, 15, 11, 11, 5, 2};
    std::vector<int> pdf_cnn_stride =   {1,  1,  1,  1,  1, 1};
    std::vector<int> pdf_cnn_padding =  {7,  7,  5,  5,  2, 0};
    std::vector<int> pdf_pool_kernel =  {12, 11, 11, 9,  7, 1};
    std::vector<int> pdf_pool_stride =  {8,  8,  8,  6,  3, 1};
    std::vector<int> pdf_pool_padding = {6,  5,  5,  3,  2, 0};
    std::shared_ptr<CV_1D_STACK> pdf_cnn_stack;
    ///////////////////////////////////////////////////////
    std::vector<int> inner_fanout_fc_sizes = {fc_neuron_size * 3, fc_neuron_size * 3, fc_neuron_size * 3};
    std::vector<int> root_fanout_fc_sizes = {fc_neuron_size, fc_neuron_size};
    std::vector<std::shared_ptr<LN>> inner_fanout_fcs;
    std::vector<std::shared_ptr<LN>> root_fanout_fcs;
    torch::nn::Linear inner_fanout_fc_out = nullptr;
    torch::nn::Linear root_fanout_fc_out = nullptr;
    torch::nn::BatchNorm1d value_bn = nullptr;
    std::vector<int> value_in_sizes = {fc_neuron_size};
    std::vector<std::shared_ptr<LN>> value_in_connect;
    std::vector<int> weight_in_sizes = {fc_neuron_size};
    std::vector<std::shared_ptr<LN>> weight_in_connect;
public:
    Global_PAI_network() {
        int pdf_size = PDF_SIZE;
        pdf_cnn_stack = register_module(
                "pdf_cnn",
                std::make_shared<CV_1D_STACK>(CV_1D_STACK(pdf_cnn_channel,
                                                          pdf_cnn_kernel,
                                                          pdf_cnn_stride,
                                                          pdf_cnn_padding,
                                                          pdf_pool_kernel,
                                                          pdf_pool_stride,
                                                          pdf_pool_padding,
                                                          pdf_size
                )));
        value_bn = register_module("value_bn", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1)));
        int value_size = VALUE_SIZE;
        for (int i = 0; i < int(value_in_sizes.size()); ++i) {
            value_in_connect.push_back(register_module("value_input_" + std::to_string(i),
                                                       std::make_shared<LN>(LN(value_size, value_in_sizes[i]))));
            value_size = value_in_sizes[i];
        }
        int weight_size = 2;
        for (int i = 0; i < int(weight_in_sizes.size()); ++i) {
            weight_in_connect.push_back(register_module("weight_input_" + std::to_string(i),
                                                       std::make_shared<LN>(LN(weight_size, weight_in_sizes[i]))));
            weight_size = weight_in_sizes[i];
        }
        int out_size = pdf_size * pdf_cnn_channel.back() + value_size + weight_size;
        for (std::size_t i = 0; i < inner_fanout_fc_sizes.size(); i++) {
            inner_fanout_fcs.push_back(register_module("inner_fc_" + std::to_string(i),
                                                       std::make_shared<LN>(LN(out_size, inner_fanout_fc_sizes[i]))));
            out_size = inner_fanout_fc_sizes[i];
        }
        inner_fanout_fc_out = register_module("inner_fanout_fc_out", torch::nn::Linear(out_size, INNER_FANOUT_ROW * INNER_FANOUT_COLUMN));
        out_size = pdf_size * pdf_cnn_channel.back() + value_size + weight_size;
        for (std::size_t i = 0; i < root_fanout_fc_sizes.size(); i++) {
            root_fanout_fcs.push_back(register_module("root_fc_" + std::to_string(i),
                                                       std::make_shared<LN>(LN(out_size, root_fanout_fc_sizes[i]))));
            out_size = root_fanout_fc_sizes[i];
        }
        root_fanout_fc_out = register_module("root_fanout_fc_out", torch::nn::Linear(out_size, 1));
    }

    auto
    forward(torch::Tensor pdf, torch::Tensor value, torch::Tensor weights) {
        auto sample_number = pdf.size(0);
        pdf = pdf.view({sample_number, 1, pdf.size(1)});
        pdf = pdf_cnn_stack->forward(pdf);
        pdf = pdf.view({sample_number, -1});
        value = value.view({sample_number,1,value.size(1)});
        value = value_bn->forward(value);
        value = value.view({sample_number,1});
        for (auto &fc: value_in_connect) {
            value = fc->forward(value);
        }
        for (auto &fc: weight_in_connect) {
            weights = fc->forward(weights);
        }
        auto concat = torch::hstack({pdf, value, weights});
        auto inner_fanout = concat;
        for (auto &fc: inner_fanout_fcs) {
            inner_fanout = fc->forward(inner_fanout);
        }
        inner_fanout = inner_fanout_fc_out->forward(inner_fanout);
        inner_fanout = inner_fanout.view({sample_number, INNER_FANOUT_ROW, INNER_FANOUT_COLUMN});
        inner_fanout *= (max_inner_fan_out - min_inner_fan_out);
        inner_fanout += (max_inner_fan_out + min_inner_fan_out) * float(0.5);
        auto root_fanout = concat;
        for (auto &fc: root_fanout_fcs) {
            root_fanout = fc->forward(root_fanout);
        }
        root_fanout = root_fanout_fc_out->forward(root_fanout);
        root_fanout *= (max_root_fan_out - min_root_fan_out);
        root_fanout += (max_root_fan_out + min_root_fan_out) * float(0.5);
        return std::pair<torch::Tensor, torch::Tensor>(root_fanout, inner_fanout);
    }
};



#endif //HITS_Q_MODEL_H
