#include<iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

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

// CUDA 内核：水平高斯滤波
__global__ void horizontalBlurKernel(float* input, float* output, int width, int height, float* kernel, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float value = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int ix = min(max(x + k, 0), width - 1);
            value += input[y * width + ix] * kernel[k + radius];
        }
        output[y * width + x] = value;
    }
}

// CUDA 内核：垂直高斯滤波
__global__ void verticalBlurKernel(float* input, float* output, int width, int height, float* kernel, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float value = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int iy = min(max(y + k, 0), height - 1);
            value += input[iy * width + x] * kernel[k + radius];
        }
        output[y * width + x] = value;
    }
}

// 高斯滤波器 - CUDA 实现
void gaussianBlurCUDA(Image& img, float sigma) {
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

    float *d_input, *d_temp, *d_output, *d_kernel;
    size_t imgSize = img.width * img.height * sizeof(float);
    size_t kernelSize = kernel.size() * sizeof(float);

    // 分配 GPU 内存
    cudaMalloc((void)&d_input, imgSize);
    cudaMalloc((void)&d_temp, imgSize);
    cudaMalloc((void)&d_output, imgSize);
    cudaMalloc((void)&d_kernel, kernelSize);

    // 复制数据到 GPU
    cudaMemcpy(d_input, img.data.data(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice);

    // 定义 CUDA 网格和块
    dim3 blockSize(16, 16);
    dim3 gridSize((img.width + blockSize.x - 1) / blockSize.x, (img.height + blockSize.y - 1) / blockSize.y);

    // 执行水平高斯滤波
    horizontalBlurKernel<<<gridSize, blockSize>>>(d_input, d_temp, img.width, img.height, d_kernel, radius);
    cudaDeviceSynchronize();

    // 执行垂直高斯滤波
    verticalBlurKernel<<<gridSize, blockSize>>>(d_temp, d_output, img.width, img.height, d_kernel, radius);
    cudaDeviceSynchronize();

    // 复制结果回主机
    cudaMemcpy(img.data.data(), d_output, imgSize, cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// 高斯差分
Image differenceOfGaussiansCUDA(const Image& img, float sigma1, float sigma2) {
    Image blurred1 = img;
    Image blurred2 = img;
    gaussianBlurCUDA(blurred1, sigma1);
    gaussianBlurCUDA(blurred2, sigma2);

    Image dog(img.width, img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            dog.at(x, y) = blurred1.at(x, y) - blurred2.at(x, y);
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
    Image dog = differenceOfGaussiansCUDA(img, sigma1, sigma2);

    // 检测特征点
    std::vector<std::pair<int, int>> keypoints = detectKeypoints(dog);

    // 输出特征点数量
    std::cout << "Detected " << keypoints.size() << " keypoints.\n";

    return 0;
}