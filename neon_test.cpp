#include <iostream>
#include <vector>
#include <cmath>
#include <arm_neon.h>

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

// NEON SIMD 高斯滤波器
void neonGaussianBlur(Image& img, float sigma) {
    int radius = static_cast<int>(std::ceil(3 * sigma));
    std::vector<float> kernel(2 * radius + 1);
    float sum = 0.0f;

    for (int i = -radius; i <= radius; ++i) {
        kernel[i + radius] = std::exp(-(i * i) / (2 * sigma * sigma));
        sum += kernel[i + radius];
    }

    for (auto& k : kernel) {
        k /= sum;
    }

    // NEON 指令集需要使用 float32x4_t 类型处理四个浮点数的 SIMD 计算
    int width = img.width;
    int height = img.height;

    float32x4_t neon_kernel[radius + 1];
    for (int i = 0; i <= radius; ++i) {
        neon_kernel[i] = vdupq_n_f32(kernel[i]);
    }

    float *input = img.data.data();
    float *output = new float[img.data.size()];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int k = -radius; k <= radius; k += 4) {
                int ix0 = std::max(std::min(x + k + 0, width - 1), 0);
                int ix1 = std::max(std::min(x + k + 1, width - 1), 0);
                int ix2 = std::max(std::min(x + k + 2, width - 1), 0);
                int ix3 = std::max(std::min(x + k + 3, width - 1), 0);

                float32x4_t data0 = vld1q_f32(&input[y * width + ix0]);
                float32x4_t data1 = vld1q_f32(&input[y * width + ix1]);
                float32x4_t data2 = vld1q_f32(&input[y * width + ix2]);
                float32x4_t data3 = vld1q_f32(&input[y * width + ix3]);

                float32x4_t kernel0 = neon_kernel[k + radius];
                float32x4_t kernel1 = neon_kernel[k + radius + 1];
                float32x4_t kernel2 = neon_kernel[k + radius + 2];
                float32x4_t kernel3 = neon_kernel[k + radius + 3];

                sum = vmlaq_f32(sum, data0, kernel0);
                sum = vmlaq_f32(sum, data1, kernel1);
                sum = vmlaq_f32(sum, data2, kernel2);
                sum = vmlaq_f32(sum, data3, kernel3);
            }
            // Store the result back to output
            vst1q_f32(&output[y * width + x], sum);
        }
    }

    // Copy the output back to the image data
    for (int i = 0; i < img.data.size(); ++i) {
        img.data[i] = output[i];
    }

    delete[] output;
}

int main() {
    // 示例图像
    int width = 512;
    int height = 512;
    Image img(width, height);

    // 生成随机图像数据
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            img.at(x, y) = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // NEON 指令集优化的高斯滤波器
    float sigma = 1.0f;
    neonGaussianBlur(img, sigma);

    return 0;
}
