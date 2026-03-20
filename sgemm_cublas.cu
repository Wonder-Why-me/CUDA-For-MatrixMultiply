#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ==================== 错误检查宏 ====================

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

#define CUBLAS_CHECK(call)                                \
    do                                                    \
    {                                                     \
        cublasStatus_t status = call;                     \
        if (status != CUBLAS_STATUS_SUCCESS)              \
        {                                                 \
            fprintf(stderr, "[CUBLAS ERROR] %s:%d: %d\n", \
                    __FILE__, __LINE__, status);          \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

// ==================== 矩阵结构体 ====================

typedef struct
{
    int rows;      // 行数
    int cols;      // 列数
    float *h_data; // CPU端数据 (host)
    float *d_data; // GPU端数据 (device)
} Matrix;

// ==================== 矩阵操作函数 ====================

// 创建矩阵（同时分配CPU和GPU内存）
Matrix *create_matrix(int rows, int cols)
{
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    if (!mat)
        return NULL;

    mat->rows = rows;
    mat->cols = cols;

    // 分配CPU内存
    mat->h_data = (float *)malloc(rows * cols * sizeof(float));
    if (!mat->h_data)
    {
        free(mat);
        return NULL;
    }

    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&mat->d_data, rows * cols * sizeof(float)));

    return mat;
}

// 释放矩阵
void free_matrix(Matrix *mat)
{
    if (mat)
    {
        if (mat->h_data)
            free(mat->h_data);
        if (mat->d_data)
            cudaFree(mat->d_data);
        free(mat);
    }
}

// 随机初始化矩阵（范围[0,1]）
void init_random(Matrix *mat)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->h_data[i] = (float)rand() / RAND_MAX;
    }
}

// 矩阵置零
void set_zero(Matrix *mat)
{
    memset(mat->h_data, 0, mat->rows * mat->cols * sizeof(float));
}

// 将数据从CPU复制到GPU
void h2d_copy(Matrix *mat)
{
    CUDA_CHECK(cudaMemcpy(mat->d_data, mat->h_data,
                          mat->rows * mat->cols * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// 将数据从GPU复制到CPU
void d2h_copy(Matrix *mat)
{
    CUDA_CHECK(cudaMemcpy(mat->h_data, mat->d_data,
                          mat->rows * mat->cols * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// 打印矩阵（小矩阵用）
void print_matrix(const Matrix *mat, const char *name)
{
    printf("\n%s (%d x %d):\n", name, mat->rows, mat->cols);
    for (int i = 0; i < mat->rows && i < 6; i++)
    {
        for (int j = 0; j < mat->cols && j < 6; j++)
        {
            printf("%8.3f ", mat->h_data[i * mat->cols + j]);
        }
        if (mat->cols > 6)
            printf(" ...");
        printf("\n");
    }
    if (mat->rows > 6)
        printf(" ...\n");
}

// 验证两个矩阵是否相等
int verify_matrix(const Matrix *C1, const Matrix *C2, float tolerance)
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

// ==================== CPU矩阵乘法（用于验证）====================

void matmul_cpu(const Matrix *A, const Matrix *B, Matrix *C)
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

// ==================== cuBLAS矩阵乘法 ====================

// cuBLAS矩阵乘法（行主序）
void matmul_cublas(const Matrix *A, const Matrix *B, Matrix *C,
                   cublasHandle_t handle)
{
    int m = A->rows; // A的行数
    int k = A->cols; // A的列数 = B的行数
    int n = B->cols; // B的列数

    float alpha = 1.0f;
    float beta = 0.0f;

    // 重要：cuBLAS使用列主序，但我们的数据是行主序
    // 对于行主序的 C = A * B，cuBLAS调用等价于：
    // C^T = B^T * A^T，所以行列要交换
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, // A不转置
                             CUBLAS_OP_N, // B不转置
                             n,           // 结果矩阵的列数（转置后的行数）
                             m,           // 结果矩阵的行数（转置后的列数）
                             k,           // 内积维度
                             &alpha,      // alpha
                             B->d_data,   // B在GPU上
                             n,           // B的leading dimension
                             A->d_data,   // A在GPU上
                             k,           // A的leading dimension
                             &beta,       // beta
                             C->d_data,   // C在GPU上
                             n));         // C的leading dimension
}

// ==================== 性能测试函数 ====================

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

// 运行单次测试
BenchmarkResult run_benchmark(int m, int k, int n, int num_runs)
{
    BenchmarkResult result = {m, k, n, 0, 0, 0, 0, 0, 0};
    double flops = 2.0 * m * n * k; // 总浮点运算次数

    printf("\n--------------------------------------------------\n");
    printf("测试维度: %4d x %4d  *  %4d x %4d\n", m, k, k, n);
    printf("总运算量: %.2e FLOP\n", flops);

    // 创建矩阵
    Matrix *A = create_matrix(m, k);
    Matrix *B = create_matrix(k, n);
    Matrix *C_cpu = create_matrix(m, n);
    Matrix *C_gpu = create_matrix(m, n);
    Matrix *C_ref = create_matrix(m, n);

    if (!A || !B || !C_cpu || !C_gpu || !C_ref)
    {
        printf("内存分配失败!\n");
        return result;
    }

    // 初始化数据
    srand(42); // 固定种子，保证可重复性
    init_random(A);
    init_random(B);

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
    h2d_copy(A);
    h2d_copy(B);

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 设置CUDA设备为固定时钟（可选）
    cudaDeviceSynchronize();

    // 预热运行（不计时）
    matmul_cublas(A, B, C_gpu, handle);
    cudaDeviceSynchronize();

    // ---------- GPU性能测试 ----------
    printf("[GPU] 运行 %d 次取平均...\n", num_runs);

    double total_gpu_time = 0.0;

    for (int run = 0; run < num_runs; run++)
    {
        // 创建CUDA事件用于精确计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 开始计时
        cudaEventRecord(start, 0);

        // 执行cuBLAS矩阵乘法
        matmul_cublas(A, B, C_gpu, handle);

        // 结束计时
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
    d2h_copy(C_gpu);

    printf("\n[验证] ");
    result.correct = verify_matrix(C_ref, C_gpu, 1e-4);

    // 小矩阵打印结果
    if (m <= 5 && n <= 5 && k <= 5)
    {
        print_matrix(A, "矩阵 A");
        print_matrix(B, "矩阵 B");
        print_matrix(C_ref, "结果 C (CPU)");
        print_matrix(C_gpu, "结果 C (GPU)");
    }

    // 清理
    cublasDestroy(handle);
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_cpu);
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

// ==================== 主函数 ====================

int main()
{
    printf("========================================\n");
    printf("   cuBLAS 通用矩阵乘法基准程序\n");
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
        // 小矩阵（用于验证）
        {4, 5, 3},
        {16, 16, 16},

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