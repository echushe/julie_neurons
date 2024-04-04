/******************************************************************************
 *             Copyright 2020 DeepFrame AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#pragma once
#include <string>
#include <random>
#include <exception>
#include <chrono>

namespace julie
{
    // Calculations of matrices can be able to run on CPU, nvidia GPU or any other compatible platform.
    // Matrix types:
    //     CPU:  A matrix of this type is supposed to run on a CPU
    //     CUDA: A matrix of this type is supposed to run on an nvidia GPU via CUDA APIs
    //     CL:   A matrix of this type is supposed to run via openCL APIs (under development)
    enum MatrixType
    {
        UNKNOWN = 0,
        CPU = 1,
        CUDA = 2,
        CL = 3
    };

    // Get time in milliseconds
    static int64_t get_time_in_milliseconds()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::system_clock::now().time_since_epoch()).count();
    }

    /*
    Definitions of most frequently used exceptions
    */

    const std::string invalid_shape_dim("Shape: Shape dimensions should not be less than 1; ");
    const std::string invalid_shape_val("Shape: Each dimension of Shape should not be less than 1; ");
    const std::string invalid_coord_dim("Coordinate: coodinate dimensions should not be less than 1; ");
    const std::string invalid_coord_val("Coordinate: Each dimension of coordinate should not be less than 0; ");
    const std::string invalid_coordinate("Coordinate: invalid coordinate dimentsions or values; ");
    const std::string invalid_shape("Matrix: The two matrices should have the same shape; ");
    const std::string incompatible_shape("Matrix: The two matrices cannot be multiplied due to inconsistent shapes; ");
    const std::string incompatible_size("Matrix: The two matrices should have the same amount of elements; ");

    static std::default_random_engine global_rand_engine;

} // namespace julie
