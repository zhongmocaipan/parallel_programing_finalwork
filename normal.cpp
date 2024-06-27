#include <vector>
#include <cmath>
#include <algorithm>

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

// 高斯滤波器
void gaussianBlur(Image& img, float sigma) {
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
        for (int x = 0; x < img.width; ++x) {
            float value = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int ix = std::clamp(x + k, 0, img.width - 1);
                value += img.at(ix, y) * kernel[k + radius];
            }
            temp.at(x, y) = value;
        }
    }

    // 垂直高斯滤波
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float value = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int iy = std::clamp(y + k, 0, img.height - 1);
                value += temp.at(x, iy) * kernel[k + radius];
            }
            img.at(x, y) = value;
        }
    }
}

// 高斯差分
Image differenceOfGaussians(const Image& img, float sigma1, float sigma2) {
    Image blurred1 = img;
    Image blurred2 = img;
    gaussianBlur(blurred1, sigma1);
    gaussianBlur(blurred2, sigma2);

    Image dog(img.width, img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            dog.at(x, y) = blurred1.at(x, y) - blurred2.at(x, y);
        }
    }
    return dog;
}

// 特征点检测（局部极大值检测）
std::vector<std::pair<int, int>> detectKeypoints(const Image& dog) {
    std::vector<std::pair<int, int>> keypoints;

    for (int y = 1; y < dog.height - 1; ++y) {
        for (int x = 1; x < dog.width - 1; ++x) {
            float value = dog.at(x, y);
            bool isMax = true;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (dog.at(x + dx, y + dy) >= value) {
                        isMax = false;
                        break;
                    }
                }
                if (!isMax) break;
            }
            if (isMax) keypoints.emplace_back(x, y);
        }
    }

    return keypoints;
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
    Image dog = differenceOfGaussians(img, sigma1, sigma2);

    // 检测特征点
    std::vector<std::pair<int, int>> keypoints = detectKeypoints(dog);

    // 输出特征点数量
    std::cout << "Detected " << keypoints.size() << " keypoints.\n";

    return 0;
}