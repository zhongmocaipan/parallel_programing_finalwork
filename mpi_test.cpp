include <mpi.h>
include <iostream>
include <vector>
include <cmath>
include <algorithm>

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

// 高斯滤波器 - 单进程实现
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

int main(int argc, charargv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 512, height = 512;
    Image img(width, height);

    // 生成随机图像数据
    if (rank == 0) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                img.at(x, y) = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }

    // 广播图像大小
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 每个进程处理的行数
    int rowsPerProcess = height / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? height : startRow + rowsPerProcess;

    // 分配每个进程处理的图像部分
    Image localImg(width, endRow - startRow);
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int startRowP = p * rowsPerProcess;
            int endRowP = (p == size - 1) ? height : startRowP + rowsPerProcess;
            MPI_Send(&img.data[startRowP * width], (endRowP - startRowP) * width, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
        localImg = img;
    } else {
        MPI_Recv(&localImg.data[0], (endRow - startRow) * width, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 对本地图像部分应用高斯滤波
    gaussianBlur(localImg, 1.0);

    // 收集所有进程的结果
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int startRowP = p * rowsPerProcess;
            int endRowP = (p == size - 1) ? height : startRowP + rowsPerProcess;
            MPI_Recv(&img.data[startRowP * width], (endRowP - startRowP) * width, MPI_FLOAT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(&localImg.data[0], (endRow - startRow) * width, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // 构建DOG尺度空间
        float sigma1 = 1.0f;
        float sigma2 = 2.0f;
        Image dog = differenceOfGaussians(img, sigma1, sigma2);

        // 检测特征点
        std::vector<std::pair<int, int>> keypoints = detectKeypoints(dog);

        // 输出特征点数量
        std::cout << "Detected " << keypoints.size() << " keypoints.\n";
    }

    MPI_Finalize();
    return 0;
}