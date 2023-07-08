#include "onnx_extended_test_common.h"
#include "cpu/c_op_conv_common.h"
#include "cpu/c_op_conv.h"

using namespace onnx_c_ops;

void testAssertTrue() {
    ASSERT_THROW(true);
}

void test_gemm() {
    float pa[4] = { 1, 2, 3, 4 };
    float pb[4] = { 10, 20, 30, 40 };
    float pc[4] = { -0.1, -0.2, -0.3, -0.4 };
    float expected[4] = { 69.9, 99.8, 149.7, 219.6 };
    gemm(false, false, 2, 2, 2, 1.0f, pa, pb, 1.0f, pc);
    ASSERT_EQUAL_VECTOR(4, expected, pc);

    float pc2[4] = { -0.1, -0.2, -0.3, -0.4 };
    float expected2[4] = { 70.0, 100.0, 150.0, 220.0 };
    gemm(false, false, 2, 2, 2, 1.0f, pa, pb, 0.0f, pc2);
    ASSERT_EQUAL_VECTOR(4, expected2, pc2);

    float pc3[4] = { -0.1, -0.2, -0.3, -0.4 };
    float expected3[4] = { 139.9, 199.8, 299.7, 439.6 };
    gemm(false, false, 2, 2, 2, 2.0f, pa, pb, 1.0f, pc3);
    ASSERT_EQUAL_VECTOR(4, expected3, pc3);

    float paA[4] = { 1, 2, 3, 4 };
    float pbA[4] = { 1, 0, 0, 1 };
    float pcA[4] = { 0, 0, 0, 0 };
    float expectedA[4] = { 1, 3, 2, 4 };
    gemm(true, false, 2, 2, 2, 1.0f, paA, pbA, 1.0f, pcA);
    ASSERT_EQUAL_VECTOR(4, expectedA, pcA);

    float paB[4] = { 1, 0, 0, 1 };
    float pbB[4] = { 1, 2, 3, 4 };
    float pcB[4] = { 0, 0, 0, 0 };
    float expectedB[4] = { 1, 2, 3, 4 };
    gemm(true, false, 2, 2, 2, 1.0f, paB, pbB, 1.0f, pcB);
    ASSERT_EQUAL_VECTOR(4, expectedB, pcB);

    float paC[4] = { 1, 1, 0, 1 };
    float pbC[4] = { 1, 1, 0, 0 };
    float pcC[4] = { 0, 0, 0, 0 };
    float expectedC[4] = { 10, 10, 10, 10 };
    gemm(true, false, 2, 2, 2, 10.0f, paC, pbC, 1.0f, pcC);
    ASSERT_EQUAL_VECTOR(4, expectedC, pcC);

    float pc6[4] = { -0.1, -0.2, -0.3, -0.4 };
    float expected6[4] = { 69.9, 149.8, 99.7, 219.6 };
    gemm(true, true, 2, 2, 2, 1.0f, pa, pb, 1.0f, pc6);
    ASSERT_EQUAL_VECTOR(4, expected6, pc6);

    float pc4[4] = { -0.1, -0.2, -0.3, -0.4 };
    float expected4[4] = { 99.9, 139.8, 139.7, 199.6 };
    gemm(true, false, 2, 2, 2, 1.0f, pa, pb, 1.0f, pc4);
    ASSERT_ALMOST_VECTOR(4, expected4, pc4, 1e-5f);

    float pc5[4] = { -0.1, -0.2, -0.3, -0.4 };
    float expected5[4] = { 49.9, 109.8, 109.7, 249.6 };
    gemm(false, true, 2, 2, 2, 1.0f, pa, pb, 1.0f, pc5);
    ASSERT_EQUAL_VECTOR(4, expected5, pc5);
}

int main(int, char**) {
    testAssertTrue();
    test_gemm();
}
