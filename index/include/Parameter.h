//
// Created by redamancyguy on 23-8-2.
//

#ifndef HITS_PARAMETER_IN
#define HITS_PARAMETER_IN


#include <algorithm>
#include <string>

typedef double KEY_TYPE;
//typedef long long KEY_TYPE;
typedef double VALUE_TYPE;
static const std::string father_path = "/home/redamancyguy/Desktop/buffer/data/";
static const std::string data_father_path = father_path + "data_set/";
static const std::string train_dataset_path = father_path + "train_dataset/";
static const std::string model_father_path = father_path + "model/";
static const std::string experience_father_path = father_path + "experience/";

#define BATCH_SIZE 512
#define train_steps 20
#define test_steps 5
#define train_lr double(0.0000885396)
#define train_wd double(0.001)
//#define train_wd double(0.00)


#define linear_epsilon 0

#define default_error_bound 0.13

#define min_root_fan_out float(1.0)
#define max_root_fan_out float(256.0 * 1024.0)
#define min_inner_fan_out float(1.0)
//#define max_inner_fan_out 333
#define max_inner_fan_out float(1024.0)
#define min_data_set_size 10000000
#define max_data_set_size 50000000
//#define max_data_set_size 100000000

inline float shrink_root_fan_out(float x) {
    return std::max(min_root_fan_out, std::min(max_root_fan_out, x));
}

inline float shrink_inner_fan_out(float x) {
    return std::max(min_inner_fan_out, std::min(max_inner_fan_out, x));
}

inline int shrink_dataset_size(int x) {
    return std::max(min_data_set_size, std::min(max_data_set_size, x));
}

#define  default_inner_fan_out float(10.0)
#define  default_root_fan_out float(333.0)
//#define show_curve
#define INNER_FANOUT_COLUMN 256
#define INNER_FANOUT_ROW 1
#define REWARD_SIZE 2
#define BUCKET_SIZE (16 * 1024)
#define INNER_FANOUT_SIZE (INNER_FANOUT_ROW * INNER_FANOUT_COLUMN)
#define PDF_SIZE BUCKET_SIZE
#define VALUE_SIZE 1
#define MEMORY_WEIGHT 1
#define QUERY_WEIGHT 1
//#define QUERY_WEIGHT 30
#define SMALL_PDF_SIZE 1024
int MAX_GEN = 50;

#endif //HITS_PARAMETER_IN
