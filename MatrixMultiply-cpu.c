#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 可选优化：如果编译器支持AVX，定义宏 USE_AVX
#ifdef __AVX__
#include <immintrin.h>
#define USE_AVX 1
#endif

// 可选优化：如果编译器支持OpenMP，定义宏 USE_OMP
#ifdef _OPENMP
#include <omp.h>
#define USE_OMP 1
#endif

// 矩阵结构体
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

// 创建矩阵
Matrix* create_matrix(int rows, int cols) {
    Matrix *mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)malloc(rows * cols * sizeof(float));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    return mat;
}

// 释放矩阵
void free_matrix(Matrix *mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

// 随机初始化
void init_random(Matrix *mat) {
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = (float)rand() / RAND_MAX; // [0,1]
    }
}

// 置零
void set_zero(Matrix *mat) {
    memset(mat->data, 0, mat->rows * mat->cols * sizeof(float));
}

// 打印矩阵（小矩阵用）
void print_matrix(const Matrix *mat, const char *name) {
    printf("%s (%d x %d):\n", name, mat->rows, mat->cols);
    for (int i = 0; i < mat->rows && i < 6; i++) {
        for (int j = 0; j < mat->cols && j < 6; j++) {
            printf("%8.3f ", mat->data[i * mat->cols + j]);
        }
        if (mat->cols > 6) printf("...");
        printf("\n");
    }
    if (mat->rows > 6) printf("...\n");
    printf("\n");
}

// 验证两个矩阵是否相等（容忍误差）
int verify(const Matrix *C1, const Matrix *C2, float tol) {
    if (C1->rows != C2->rows || C1->cols != C2->cols) return 0;
    int n = C1->rows * C1->cols;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(C1->data[i] - C2->data[i]);
        if (diff > tol) {
            printf("验证失败: 位置 %d, %f vs %f, diff=%f\n", i, C1->data[i], C2->data[i], diff);
            return 0;
        }
    }
    return 1;
}

// ---------- 实现1：原始三重循环 (i-j-k) ----------
void matmul_basic(const Matrix *A, const Matrix *B, Matrix *C) {
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;
    set_zero(C);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int t = 0; t < k; t++) {
                sum += A->data[i * k + t] * B->data[t * n + j];
            }
            C->data[i * n + j] = sum;
        }
    }
}

// ---------- 实现2：循环重排 i-k-j (缓存优化) ----------
void matmul_ikj(const Matrix *A, const Matrix *B, Matrix *C) {
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;
    set_zero(C);
    for (int i = 0; i < m; i++) {
        for (int t = 0; t < k; t++) {
            float a = A->data[i * k + t];
            for (int j = 0; j < n; j++) {
                C->data[i * n + j] += a * B->data[t * n + j];
            }
        }
    }
}

// ---------- 实现3：分块乘法 ----------
void matmul_block(const Matrix *A, const Matrix *B, Matrix *C, int bs) {
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;
    set_zero(C);
    for (int i0 = 0; i0 < m; i0 += bs) {
        for (int t0 = 0; t0 < k; t0 += bs) {
            for (int j0 = 0; j0 < n; j0 += bs) {
                int i_end = (i0 + bs < m) ? i0 + bs : m;
                int t_end = (t0 + bs < k) ? t0 + bs : k;
                int j_end = (j0 + bs < n) ? j0 + bs : n;
                for (int i = i0; i < i_end; i++) {
                    for (int t = t0; t < t_end; t++) {
                        float a = A->data[i * k + t];
                        for (int j = j0; j < j_end; j++) {
                            C->data[i * n + j] += a * B->data[t * n + j];
                        }
                    }
                }
            }
        }
    }
}

// ---------- 实现4：AVX向量化 (需要AVX支持) ----------
#ifdef USE_AVX
void matmul_avx(const Matrix *A, const Matrix *B, Matrix *C) {
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;
    set_zero(C);
    for (int i = 0; i < m; i++) {
        for (int t = 0; t < k; t++) {
            __m256 a_vec = _mm256_set1_ps(A->data[i * k + t]);
            int j = 0;
            // 处理8的倍数
            for (; j <= n - 8; j += 8) {
                __m256 b = _mm256_loadu_ps(&B->data[t * n + j]);
                __m256 c = _mm256_loadu_ps(&C->data[i * n + j]);
                c = _mm256_fmadd_ps(a_vec, b, c);
                _mm256_storeu_ps(&C->data[i * n + j], c);
            }
            // 处理剩余元素
            for (; j < n; j++) {
                C->data[i * n + j] += A->data[i * k + t] * B->data[t * n + j];
            }
        }
    }
}
#endif

// ---------- 实现5：OpenMP并行 (需要OpenMP支持) ----------
#ifdef USE_OMP
void matmul_omp(const Matrix *A, const Matrix *B, Matrix *C) {
    int m = A->rows;
    int k = A->cols;
    int n = B->cols;
    set_zero(C);
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int t = 0; t < k; t++) {
            float a = A->data[i * k + t];
            for (int j = 0; j < n; j++) {
                C->data[i * n + j] += a * B->data[t * n + j];
            }
        }
    }
}
#endif

// ---------- 性能测试 ----------
void test_one(const char *name, void (*func)(const Matrix*, const Matrix*, Matrix*),
              const Matrix *A, const Matrix *B, Matrix *C, const Matrix *C_ref) {
    set_zero(C);
    clock_t start = clock();
    func(A, B, C);
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    double flops = 2.0 * A->rows * A->cols * B->cols; // 乘加各一次
    double gflops = flops / time / 1e9;

    printf("%-20s: time=%.3f s, GFLOPS=%.2f, speedup=%.2f", 
           name, time, gflops, time ? (1.0/time) : 0);
    if (verify(C_ref, C, 1e-4))
        printf(" ✓\n");
    else
        printf(" ✗\n");
}

// 针对需要额外参数的函数（如分块）单独测试
void test_block(const char *name, const Matrix *A, const Matrix *B, Matrix *C, const Matrix *C_ref, int bs) {
    set_zero(C);
    clock_t start = clock();
    matmul_block(A, B, C, bs);
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    double flops = 2.0 * A->rows * A->cols * B->cols;
    double gflops = flops / time / 1e9;
    printf("%-20s: time=%.3f s, GFLOPS=%.2f, speedup=%.2f", 
           name, time, gflops, time ? (1.0/time) : 0);
    if (verify(C_ref, C, 1e-4))
        printf(" ✓\n");
    else
        printf(" ✗\n");
}

int main() {
    srand(time(NULL));

    printf("===== 通用单精度矩阵乘法性能测试 =====\n\n");
#ifdef USE_AVX
    printf("AVX: 启用\n");
#else
    printf("AVX: 未启用\n");
#endif
#ifdef USE_OMP
    printf("OpenMP: 启用 (最大线程 %d)\n", omp_get_max_threads());
#else
    printf("OpenMP: 未启用\n");
#endif

    // 定义测试用例： (m, k, n)
    int tests[][3] = {
        {512, 512, 512},     // 方阵
        {256, 1024, 128},    // 长方形
        {1024, 128, 256},    // 瘦高
        {4, 5, 3}            // 小矩阵验证
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < num_tests; t++) {
        int m = tests[t][0];
        int k = tests[t][1];
        int n = tests[t][2];

        printf("\n---------- 测试 %d x %d  *  %d x %d ----------\n", m, k, k, n);

        Matrix *A = create_matrix(m, k);
        Matrix *B = create_matrix(k, n);
        Matrix *C_ref = create_matrix(m, n); // 基准结果
        Matrix *C_test = create_matrix(m, n);

        init_random(A);
        init_random(B);

        // 计算基准（使用原始三重循环）
        matmul_basic(A, B, C_ref);
        if (m <= 6 && k <= 6 && n <= 6) {
            print_matrix(A, "A");
            print_matrix(B, "B");
            print_matrix(C_ref, "C_ref");
        }

        // 测试各个优化版本
        test_one("basic", matmul_basic, A, B, C_test, C_ref); // 基准自我验证
        test_one("ikj", matmul_ikj, A, B, C_test, C_ref);
        test_block("block(64)", A, B, C_test, C_ref, 64);

#ifdef USE_AVX
        test_one("avx", matmul_avx, A, B, C_test, C_ref);
#endif
#ifdef USE_OMP
        test_one("omp", matmul_omp, A, B, C_test, C_ref);
#endif

        free_matrix(A);
        free_matrix(B);
        free_matrix(C_ref);
        free_matrix(C_test);
    }

    return 0;
}