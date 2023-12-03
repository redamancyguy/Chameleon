
#ifndef Q_MODEL_HPP
#define Q_MODEL_HPP


#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/nn/module.h>
#include <utility>
#include <fstream>
#include <ostream>
#include "DEFINE.h"
#include "FIleLock.hpp"



#define ac_func torch::relu
//#define dropout_rate 0.4

class LN : public torch::nn::Module {
public:
    torch::nn::BatchNorm1d bn = nullptr;
    torch::nn::Linear ln = nullptr;

    LN(std::int_fast32_t input_size, std::int_fast32_t output_size) {
        bn = register_module("b", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(input_size)));
        ln = register_module("l", torch::nn::Linear(input_size, output_size));
    }

    torch::Tensor forward(torch::Tensor &x) {
        x = bn->forward(x);
        x = ln->forward(x);
        x = ac_func(x);
        return x;
    }
};


class CV_1d : public torch::nn::Module {
public:
    torch::nn::Conv1d cv = nullptr;
    torch::nn::BatchNorm1d bn = nullptr;
    torch::nn::MaxPool1d pool = nullptr;

    CV_1d(int input_channel, int output_channel,
          int cnn_kernel, int cnn_stride, int cnn_padding,
          int pool_kernel, int pool_stride, int pool_padding,
          int &size
    ) {
        bn = register_module("b", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(input_channel)));
        cv = register_module("c",
                             torch::nn::Conv1d(
                                     torch::nn::Conv1dOptions(
                                             input_channel,
                                             output_channel,
                                             cnn_kernel)
                                             .stride(cnn_stride)
                                             .padding(cnn_padding)));
        size = (size + 2 * cnn_padding - cnn_kernel) / cnn_stride + 1;
        pool = register_module(
                "p",
                torch::nn::MaxPool1d(
                        torch::nn::MaxPool1dOptions(
                                pool_kernel)
                                .stride(pool_stride)
                                .padding(pool_padding)));
        size = (size + 2 * pool_padding - pool_kernel) / pool_stride + 1;
    }

    torch::Tensor forward(torch::Tensor &x) {
        x = bn->forward(x);
        x = cv->forward(x);
        x = ac_func(x);
        x = pool->forward(x);
        return x;
    }
};

class CV_2d : public torch::nn::Module {
public:
    torch::nn::Conv2d cv = nullptr;
    torch::nn::BatchNorm2d bn = nullptr;
    torch::nn::MaxPool2d pool = nullptr;

    CV_2d(int input_channel, int output_channel,
          int cnn_kernel_x, int cnn_stride_x, int cnn_padding_x,
          int pool_kernel_x, int pool_stride_x, int pool_padding_x,
          int &size_x,
          int cnn_kernel_y, int cnn_stride_y, int cnn_padding_y,
          int pool_kernel_y, int pool_stride_y, int pool_padding_y,
          int &size_y
    ) {

        bn = register_module("b", torch::nn::BatchNorm2d(torch::nn::BatchNorm1dOptions(input_channel)));
        cv = register_module("c",
                             torch::nn::Conv2d(
                                     torch::nn::Conv2dOptions(
                                             input_channel,
                                             output_channel,
                                             {cnn_kernel_x, cnn_kernel_y})
                                             .stride({cnn_stride_x, cnn_stride_y})
                                             .padding({cnn_padding_x, cnn_padding_y})));
        size_x = (size_x + 2 * cnn_padding_x - cnn_kernel_x) / cnn_stride_x + 1;
        size_y = (size_y + 2 * cnn_padding_y - cnn_kernel_y) / cnn_stride_y + 1;
        pool = register_module(
                "p",
                torch::nn::MaxPool2d(
                        torch::nn::MaxPool2dOptions(
                                {pool_kernel_x, pool_kernel_y})
                                .stride({pool_stride_x, pool_stride_y})
                                .padding({pool_padding_x, pool_padding_y})));
        size_x = (size_x + 2 * pool_padding_x - pool_kernel_x) / pool_stride_x + 1;
        size_y = (size_y + 2 * pool_padding_y - pool_kernel_y) / pool_stride_y + 1;
    }

    torch::Tensor forward(torch::Tensor &x) {
        x = bn->forward(x);
        x = cv->forward(x);
        x = ac_func(x);
        x = pool->forward(x);
        return x;
    }
};


class CV_2D_STACK : public torch::nn::Module {
    std::vector<std::shared_ptr<CV_2d>> cnn_stack;
public:
    CV_2D_STACK(const std::vector<int> &cnnChannel, const std::vector<int> &cnnKernel_x,
                const std::vector<int> &cnnStride_x, const std::vector<int> &cnnPadding_x,
                const std::vector<int> &poolKernel_x,
                const std::vector<int> &poolStride_x, const std::vector<int> &poolPadding_x, int &inputSize_x,
                const std::vector<int> &cnnKernel_y,
                const std::vector<int> &cnnStride_y, const std::vector<int> &cnnPadding_y,
                const std::vector<int> &poolKernel_y,
                const std::vector<int> &poolStride_y, const std::vector<int> &poolPadding_y, int &inputSize_y) {
        for (std::size_t i = 0, end = cnnChannel.size() - 1; i < end; i++) {
            cnn_stack.push_back(register_module(
                    "c" + std::to_string(i),
                    std::make_shared<CV_2d>(CV_2d(cnnChannel[i],
                                                  cnnChannel[i + 1],
                                                  cnnKernel_x[i],
                                                  cnnStride_x[i],
                                                  cnnPadding_x[i],
                                                  poolKernel_x[i],
                                                  poolStride_x[i],
                                                  poolPadding_x[i],
                                                  inputSize_x,
                                                  cnnKernel_y[i],
                                                  cnnStride_y[i],
                                                  cnnPadding_y[i],
                                                  poolKernel_y[i],
                                                  poolStride_y[i],
                                                  poolPadding_y[i],
                                                  inputSize_y
                    ))));
        }
    }

    torch::Tensor forward(torch::Tensor &x) {
        for (auto &i: cnn_stack) {
            x = i->forward(x);
        }
        return x;
    }
};

float accuracy(const torch::Tensor &pred, const torch::Tensor &label, float error_bound = 0.1) {
    auto calc = pred.sub(label).abs().div(label.abs());
    return (float) (calc.le(error_bound).sum().template item<float>()) / (float) calc.size(0);
}

class CV_1D_STACK : public torch::nn::Module {
    std::vector<std::shared_ptr<CV_1d>> cnn_stack;
public:
    CV_1D_STACK(const std::vector<int> &cnnChannel, const std::vector<int> &cnnKernel,
                const std::vector<int> &cnnStride, const std::vector<int> &cnnPadding,
                const std::vector<int> &poolKernel,
                const std::vector<int> &poolStride, const std::vector<int> &poolPadding, int &inputSize) {
        for (std::size_t i = 0, end = cnnChannel.size() - 1; i < end; i++) {
            cnn_stack.push_back(register_module(
                    "c" + std::to_string(i),
                    std::make_shared<CV_1d>(CV_1d(cnnChannel[i],
                                                  cnnChannel[i + 1],
                                                  cnnKernel[i],
                                                  cnnStride[i],
                                                  cnnPadding[i],
                                                  poolKernel[i],
                                                  poolStride[i],
                                                  poolPadding[i],
                                                  inputSize
                    ))));
        }
    }

    torch::Tensor forward(torch::Tensor &x) {
        for (auto &i: cnn_stack) {
            x = i->forward(x);
        }
        return x;
    }
};



#endif //Q_MODEL_HPP