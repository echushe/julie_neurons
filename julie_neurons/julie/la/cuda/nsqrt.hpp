/***************************************************************************
 * 
 *   An approximation of square root in integer mode
 *   
 *   This piece of code was written by Craig McQueen on stackoverflow forum
 *   The stackoverflow forum link:
 *   https://stackoverflow.com/questions/1100090/looking-for-an-efficient-integer-square-root-algorithm-for-arm-thumb2
 * 
*/


#pragma once
#include <stdint.h>
/**
 * \brief    Fast Square root algorithm
 *
 * Fractional parts of the answer are discarded. That is:
 *      - SquareRoot(3) --> 1
 *      - SquareRoot(4) --> 2
 *      - SquareRoot(5) --> 2
 *      - SquareRoot(8) --> 2
 *      - SquareRoot(9) --> 3
 *
 * \param[in] a_nInput - unsigned integer for which to find the square root
 *
 * \return Integer square root of the input value.
 */

int64_t nsqrt(int64_t a_nInput);
