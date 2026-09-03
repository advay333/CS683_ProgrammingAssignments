#include <stdio.h>

int main() {
    __builtin_cpu_init();

    printf("AVX-512F : %s\n", __builtin_cpu_supports("avx512f") ? "Yes" : "No");
    printf("AVX2     : %s\n", __builtin_cpu_supports("avx2") ? "Yes" : "No");
    printf("AVX      : %s\n", __builtin_cpu_supports("avx") ? "Yes" : "No");
    printf("FMA      : %s\n", __builtin_cpu_supports("fma") ? "Yes" : "No");
    printf("SSE4.2   : %s\n", __builtin_cpu_supports("sse4.2") ? "Yes" : "No");

    return 0;
}