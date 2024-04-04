#pragma once
#include "SLMatrix.hpp"
#include <vector>
#include <random>
#include <map>
#include <type_traits>

namespace test
{
    template <typename DT>
    class RandSMatrix
    {
    public:

        static julie::la::cpu::SLMatrix<DT> generate_random_SLMatrix(float non_zero_rate, const julie::la::Shape & sh);

        static julie::la::cpu::SLMatrix<DT> generate_random_SLMatrix(float non_zero_rate, const julie::la::Shape & sh, DT min, DT max);


    public:
        static std::default_random_engine generator;
    };
} // namespace test

template <typename DT>
std::default_random_engine test::RandSMatrix<DT>::generator;


template <typename DT>
julie::la::cpu::SLMatrix<DT> test::RandSMatrix<DT>::generate_random_SLMatrix(float non_zero_rate, const julie::la::Shape & sh, DT min, DT max)
{
    if (non_zero_rate > 1.0)
    {
        non_zero_rate = 1.0;
    }
    else if (non_zero_rate < 0.0)
    {
        non_zero_rate = 0.0;
    }

    if (sh.dim() != 2)
    {
        throw std::invalid_argument("Shape of lower or higher dimensions than 2-dimension is not allowed");
    }

    lint size = sh.size();
    lint n_non_zero_items = static_cast<lint>(size * non_zero_rate);

    std::map<lint, julie::la::cpu::SLMatrixTuple<DT>> key_values;

    std::uniform_int_distribution<lint> index_distribution(0, size - 1);
    std::uniform_real_distribution<float> value_distribution(min, max);

    while (key_values.size() < n_non_zero_items)
    {
        lint index = index_distribution(generator);
        DT value = static_cast<DT>(value_distribution(generator));

        if (!key_values.count(index))
        {
            auto pos = julie::la::Coordinate{index, sh};
            julie::la::cpu::SLMatrixTuple<DT> new_tuple {pos[0], pos[1], value};
            key_values.insert({index, new_tuple});
        }
    }

    // Insert generated items into the SLMatrix

    julie::la::cpu::SLMatrix<DT> rand_mat {sh};

    julie::la::cpu::SLMatrixTuple<DT> ** row_ref = new julie::la::cpu::SLMatrixTuple<DT>*[sh[0]] {nullptr};
    julie::la::cpu::SLMatrixTuple<DT> ** col_ref = new julie::la::cpu::SLMatrixTuple<DT>*[sh[1]] {nullptr};

    for (auto & key_value : key_values)
    {
        julie::la::cpu::SLMatrixTuple<DT> *t_ptr = new julie::la::cpu::SLMatrixTuple<DT> {key_value.second};
        julie::la::cpu::SLMatrix<DT>::new_item(rand_mat, row_ref, col_ref, t_ptr);
    }

    delete []row_ref;
    delete []col_ref;

    return rand_mat;

}

template <typename DT>
julie::la::cpu::SLMatrix<DT> test::RandSMatrix<DT>::generate_random_SLMatrix(float non_zero_rate, const julie::la::Shape & sh)
{
    return test::RandSMatrix<DT>::generate_random_SLMatrix(non_zero_rate, sh, -10, 10);
}