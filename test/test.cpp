
#include "dmatrix_cases.hpp"
#include "slmatrix_cases.hpp"
#include "fully_connected_cases.hpp"
#include "dmatrix_adv_cases.hpp"
#include "conv2d_cases.hpp"

int main()
{
    std::cout << "============= test program begins ============\n";

    test::test_of_dmatrix_operations();
    
    test::test_of_SLMatrix_operations();

    test::test_of_fc_model();

    test::test_of_dmatrix_adv();
    test::run_conv2d_cases();


    /*
    std::vector<std::vector<double>> matrix {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    for (auto & row : matrix)
    {
        for (auto & col : row)
        {
            std::cout << col << "\t";
        }

        std::cout << "\n";
    }
    */

    // int *a = new int;

    return 0;
}