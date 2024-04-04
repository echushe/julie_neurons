
#include "matrix_cpu_cases.hpp"
#include "slmatrix_cases.hpp"
#include "fully_connected_cases.hpp"
#include "matrix_cpu_adv_cases.hpp"
#include "conv2d_cases.hpp"
#ifdef WITH_ONEDNN
#include "conv2d_onednn_cases.hpp"
#endif

#ifdef WITH_CUDA
#include "matrix_cuda_cases.hpp"
#include "matrix_cuda_adv_cases.hpp"
#include "conv2d_cuda_cases.hpp"
#ifdef WITH_CUDNN
#include "conv2d_cudnn_cases.hpp"
#endif
#endif

int main()
{
    std::cout << "============= test program begins ============\n";

    test::test_of_Matrix_CPU_operations();
    test::test_of_Matrix_CPU_adv();
    test::test_of_fc_model();

    test::run_conv2d_cases();
#ifdef WITH_ONEDNN
    test::run_conv2d_onednn_cases();
#endif


#ifdef WITH_CUDA
    test::test_of_Matrix_CUDA_operations();
    test::test_of_Matrix_CUDA_adv();
    test::run_conv2d_cuda_cases();
#ifdef WITH_CUDNN
    test::run_conv2d_cudnn_cases();
#endif
#endif

    //test::test_of_SLMatrix_operations();


    return 0;
}