#pragma once

#include "Matrix_CPU.hpp"
#include "iMatrix.hpp"
#include <vector>


namespace dataset
{
    typedef long long int lint;

    /*
    This is an abstract interface of Dataset to read inputs and labels
    */
    class Dataset
    {
    public:
        virtual void get_samples_and_labels(
            std::vector<std::shared_ptr<julie::la::iMatrix<float>>> & inputs,
            std::vector<std::shared_ptr<julie::la::iMatrix<float>>> & labels,
            lint limit = 0) const = 0;
    };
}
