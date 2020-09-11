#pragma once

#include "DMatrix.hpp"
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
            std::vector<julie::la::DMatrix<double>> & inputs, std::vector<julie::la::DMatrix<double>> & labels, lint limit = 0) const = 0;
    };
}
