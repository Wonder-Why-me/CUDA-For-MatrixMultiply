#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// 检查cuda函数是否正确执行
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

typedef struct
{
    int rows;
    int cols;
    float *h_data;
    float *d_data;
} matrix;

// 创建矩阵同时分配内存
matrix *createMatrix(int rows, int cols)
{
    matrix *mat = (matrix *)malloc(sizeof(matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->h_data = (float *)malloc(sizeof(float) * rows * cols);
    if (!mat->h_data)
    {
        free(mat);
        return NULL;
    }
    // 分配gpu内存
    CUDA_CHECK(cudaMalloc(&mat->d_data, sizeof(float) * rows * cols));
    return mat;
}
void free_matrix(matrix *mat)
{
    if (mat)
    {
        if (mat->h_data)
        {
            free(mat->h_data);
        }
        if (mat->d_data)
        {
            cudaFree(mat->d_data);
        }
        free(mat);
    }
}
// 随机初始化[0,1]
void initMatrix(matrix *mat)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->h_data[i] = (float)rand() / RAND_MAX;
    }
}
// 将数据从cpu复制到gpu
void h2d(matrix *mat)
{
    CUDA_CHECK(cudaMemcpy(mat->d_data, mat->h_data, sizeof(float) * mat->rows * mat->cols, cudaMemcpyHostToDevice));
}

// 将数据从gpu传回cpu
void d2h(matrix *mat)
{
    CUDA_CHECK(cudaMemcpy(mat->h_data, mat->d_data, sizeof(float) * mat->rows * mat->cols, cudaMemcpyDeviceToHost));
}

// 矩阵置0
void set_zero(matrix *mat)
{
    memset(mat->h_data, 0, sizeof(float) * mat->cols * mat->rows);
}

// 验证两个矩阵是否相等
int verify(matrix *C1, matrix *C2, float tolerance)
{

    if (C1->rows != C2->rows || C1->cols != C2->cols)
    {
        printf("维度不匹配!\n");
        return 0;
    }

    int size = C1->rows * C1->cols;
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < size; i++)
    {
        float diff = fabsf(C1->h_data[i] - C2->h_data[i]);
        if (diff > max_diff)
            max_diff = diff;
        if (diff > tolerance)
        {
            errors++;
            if (errors <= 5)
            { // 只打印前5个错误
                printf("  位置[%d]: %f vs %f, diff=%f\n",
                       i, C1->h_data[i], C2->h_data[i], diff);
            }
        }
    }

    if (errors > 0)
    {
        printf("验证失败: %d/%d 个元素超出误差 (max diff=%f)\n",
               errors, size, max_diff);
        return 0;
    }

    printf("验证通过 (max diff=%f)\n", max_diff);
    return 1;
}
// cpu矩阵乘法，用于验证
void matmul_cpu(const matrix *A, const matrix *B, matrix *C)
{
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;

    set_zero(C);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (int t = 0; t < k; t++)
            {
                sum += A->h_data[i * k + t] * B->h_data[t * n + j];
            }
            C->h_data[i * n + j] = sum;
        }
    }
}

__global__ void matmul_gpu(const float *A, const float *B, float *C, int m, int k, int n)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x; // 全局列索引
    int row = blockDim.y * blockIdx.y + threadIdx.y; // 全局行索引
    if (row < m && col < n)
    {
        float val = 0.0f;
        for (int t = 0; t < k; t++)
        {
            val += A[row * k + t] * B[t * n + col];
        }
        C[n * row + col] = val;
    }
}
typedef struct
{
    int m, k, n;       // 矩阵维度
    double cpu_time;   // CPU时间
    double gpu_time;   // GPU时间
    double cpu_gflops; // CPU GFLOPS
    double gpu_gflops; // GPU GFLOPS
    double speedup;    // 加速比
    int correct;       // 是否正确
} BenchmarkResult;

BenchmarkResult run_benchmark(int m, int k, int n, int num_runs)
{
    // 配置网格和块，一个线程算C的一个数
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, // 向上取整
              (m + block.y - 1) / block.y);
    BenchmarkResult result = {m, k, n, 0, 0, 0, 0, 0, 0};
    double flops = 2.0 * m * n * k; // 总浮点运算次数

    printf("\n--------------------------------------------------\n");
    printf("测试维度: %4d x %4d  *  %4d x %4d\n", m, k, k, n);
    printf("总运算量: %.2e FLOP\n", flops);

    // 创建矩阵
    matrix *A = createMatrix(m, k);
    matrix *B = createMatrix(k, n);
    // matrix *C_cpu = createMatrix(m, n);
    matrix *C_gpu = createMatrix(m, n);
    matrix *C_ref = createMatrix(m, n);

    if (!A || !B || !C_gpu || !C_ref)
    {
        printf("内存分配失败!\n");
        return result;
    }

    // 初始化数据
    srand(42); // 固定种子，保证可重复性
    initMatrix(A);
    initMatrix(B);

    // ---------- CPU基准 ----------
    printf("\n[CPU] 正在计算...\n");
    clock_t cpu_start = clock();
    matmul_cpu(A, B, C_ref);
    clock_t cpu_end = clock();

    result.cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    result.cpu_gflops = flops / result.cpu_time / 1e9;

    printf("  CPU时间: %.3f 秒\n", result.cpu_time);
    printf("  CPU性能: %.2f GFLOPS\n", result.cpu_gflops);

    // ---------- GPU数据传输 ----------
    printf("\n[GPU] 传输数据到设备...\n");
    h2d(A);
    h2d(B);
    // 预热运行，不计时
    matmul_gpu<<<grid, block>>>(A->d_data, B->d_data, C_gpu->d_data, m, k, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaDeviceSynchronize();

    // ---------- GPU性能测试 ----------
    printf("[GPU] 运行 %d 次取平均...\n", num_runs);

    double total_gpu_time = 0.0;
    for (int run = 0; run < num_runs; run++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // 开始计时
        cudaEventRecord(start, 0);
        matmul_gpu<<<grid, block>>>(A->d_data, B->d_data, C_gpu->d_data, m, k, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // 计算耗时（毫秒）
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_gpu_time += milliseconds / 1000.0; // 转换为秒

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    result.gpu_time = total_gpu_time / num_runs;
    result.gpu_gflops = flops / result.gpu_time / 1e9;
    result.speedup = result.cpu_time / result.gpu_time;

    printf("  平均GPU时间: %.6f 秒\n", result.gpu_time);
    printf("  GPU性能: %.2f GFLOPS\n", result.gpu_gflops);
    printf("  加速比: %.2f x\n", result.speedup);

    // ---------- 结果验证 ----------
    d2h(C_gpu);

    printf("\n[验证] ");
    result.correct = verify(C_ref, C_gpu, 1e-4);
    free_matrix(A);
    free_matrix(B);
    // free_matrix(C_cpu);
    free_matrix(C_gpu);
    free_matrix(C_ref);

    return result;
}

// 打印结果表格
void print_results_table(BenchmarkResult *results, int count)
{
    printf("\n\n");
    printf("================================================================================\n");
    printf("维度                | CPU时间(s) | CPU GFLOPS | GPU时间(s) | GPU GFLOPS | 加速比 | 正确性\n");
    printf("================================================================================\n");

    for (int i = 0; i < count; i++)
    {
        BenchmarkResult *r = &results[i];
        printf("%4d x %4d x %4d | %10.3f | %10.2f | %10.6f | %10.2f | %6.2fx |   %s\n",
               r->m, r->k, r->n,
               r->cpu_time, r->cpu_gflops,
               r->gpu_time, r->gpu_gflops,
               r->speedup,
               r->correct ? "✓" : "✗");
    }
    printf("================================================================================\n");
}

int main()
{
    printf("========================================\n");
    printf("   Naive 通用矩阵乘法基准程序\n");
    printf("========================================\n\n");
    // 检查CUDA设备
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("错误：没有找到CUDA设备！\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU信息:\n");
    printf("  设备名称: %s\n", prop.name);
    printf("  计算能力: %d.%d\n", prop.major, prop.minor);
    printf("  显存大小: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  CUDA核心数: %d\n", prop.multiProcessorCount * 128); // 根据prop.major/prop.minor查表：prop.major=8，cores_per_sm = 128
    printf("  最大线程数/块: %d\n", prop.maxThreadsPerBlock);

    printf("\n========================================\n\n");

    // 定义测试用例：各种矩阵形状
    // 格式: {m, k, n}
    int test_cases[][3] = {

        // 中等方阵
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},

        // 长方形矩阵
        {256, 1024, 512},
        {512, 2048, 256},

        // 瘦高矩阵
        {2048, 64, 1024},
        {4096, 128, 256},

        // 大矩阵（需要足够显存）
        // {2048, 2048, 2048},
        // {4096, 4096, 4096},
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    BenchmarkResult *results = (BenchmarkResult *)malloc(num_tests * sizeof(BenchmarkResult));

    // 运行所有测试
    int runs_per_test = 5; // 每个测试运行5次取平均

    for (int i = 0; i < num_tests; i++)
    {
        int m = test_cases[i][0];
        int k = test_cases[i][1];
        int n = test_cases[i][2];

        results[i] = run_benchmark(m, k, n, runs_per_test);
    }

    // 打印结果汇总
    print_results_table(results, num_tests);

    free(results);

    printf("\n测试完成！\n");
    return 0;
}