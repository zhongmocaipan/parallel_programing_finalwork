#include <immintrin.h>
#include <iostream>
#include <vector>
#include <cmath>

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

// 高斯滤波器 - AVX 实现
void gaussianBlurAVX(Image& img, float sigma) {
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

    Image temp(img.width, img.height);

    // 水平高斯滤波
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; x += 8) {
            __m256 value = _mm256_setzero_ps();
            for (int k = -radius; k <= radius; ++k) {
                int ix = std::clamp(x + k, 0, img.width - 1);
                __m256 pixel = _mm256_loadu_ps(&img.at(ix, y));
                __m256 weight = _mm256_set1_ps(kernel[k + radius]);
                value = _mm256_fmadd_ps(pixel, weight, value);
            }
            _mm256_storeu_ps(&temp.at(x, y), value);
        }
    }

    // 垂直高斯滤波
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; x += 8) {
            __m256 value = _mm256_setzero_ps();
            for (int k = -radius; k <= radius; ++k) {
                int iy = std::clamp(y + k, 0, img.height - 1);
                __m256 pixel = _mm256_loadu_ps(&temp.at(x, iy));
                __m256 weight = _mm256_set1_ps(kernel[k + radius]);
                value = _mm256_fmadd_ps(pixel, weight, value);
            }
            _mm256_storeu_ps(&img.at(x, y), value);
        }
    }
}

// 高斯差分
Image differenceOfGaussiansAVX(const Image& img, float sigma1, float sigma2) {
    Image blurred1 = img;
    Image blurred2 = img;
    gaussianBlurAVX(blurred1, sigma1);
    gaussianBlurAVX(blurred2, sigma2);

    Image dog(img.width, img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; x += 8) {
            __m256 b1 = _mm256_loadu_ps(&blurred1.at(x, y));
            __m256 b2 = _mm256_loadu_ps(&blurred2.at(x, y));
            __m256 diff = _mm256_sub_ps(b1, b2);
            _mm256_storeu_ps(&dog.at(x, y), diff);
        }
    }
    return dog;
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

    // 构建DOG尺度空间
    float sigma1 = 1.0f;
    float sigma2 = 2.0f;
    Image dog = differenceOfGaussiansAVX(img, sigma1, sigma2);

    // 检测特征点
    std::vector<std::pair<int, int>> keypoints = detectKeypoints(dog);

    // 输出特征点数量
    std::cout << "Detected " << keypoints.size() << " keypoints.\n";

    return 0;
}