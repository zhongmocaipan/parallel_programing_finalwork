include <iostream>
include <vector>
include <cmath>
include <pthread.h>

define NUM_THREADS 8

// 假设我们有一个简单的图像类
class Image {
public:
    int width, height;
    std::vector<float> data;

    Image(int w, int h) : width(w), height(h), data(w * h) {}

    float& at(int x, int y) {
        return data[y * width + x];
    }

    const float& at(int x, int y) const {
        return data[y * width + x];
    }
};

struct ThreadData {
    Image* img;
    Image* temp;
    std::vector<float>* kernel;
    int start;
    int end;
};

// 水平高斯滤波 - Pthreads 实现
void* horizontalBlur(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    Image* img = data->img;
    Image* temp = data->temp;
    std::vector<float>* kernel = data->kernel;
    int radius = (kernel->size() - 1) / 2;

    for (int y = data->start; y < data->end; ++y) {
        for (int x = 0; x < img->width; ++x) {
            float value = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int ix = std::clamp(x + k, 0, img->width - 1);
                value += img->at(ix, y) * (*kernel)[k + radius];
            }
            temp->at(x, y) = value;
        }
    }

    pthread_exit(nullptr);
}

// 垂直高斯滤波 - Pthreads 实现
void* verticalBlur(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    Image* img = data->img;
    Image* temp = data->temp;
    std::vector<float>* kernel = data->kernel;
    int radius = (kernel->size() - 1) / 2;

    for (int y = data->start; y < data->end; ++y) {
        for (int x = 0; x < img->width; ++x) {
            float value = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int iy = std::clamp(y + k, 0, img->height - 1);
                value += temp->at(x, iy) * (*kernel)[k + radius];
            }
            img->at(x, y) = value;