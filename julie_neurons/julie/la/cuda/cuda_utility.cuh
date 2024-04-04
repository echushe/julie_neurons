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
#include <iostream>
#include <sstream>
#include <stdexcept>

// Error handling macro
#define CUDA_CHECK(expression)                                                                                  \
if((expression) != cudaSuccess)                                                                                 \
{                                                                                                               \
    cudaError_t err = cudaGetLastError();                                                                       \
    std::ostringstream o_stream;                                                                                \
    o_stream << "CUDA error calling \""#expression"\", code is " << err << "; from function: " << std::endl;    \
    throw std::runtime_error(o_stream.str() + std::string{__FUNCTION__});                                       \
}

#define CUDNN_CHECK(expression)                                     \
{                                                                   \
    cudnnStatus_t status = (expression);                            \
    if (status != CUDNN_STATUS_SUCCESS)                             \
    {                                                               \
        std::cerr << "Error on line " << __LINE__ << ": "           \
                    << cudnnGetErrorString(status) << std::endl;    \
        std::exit(EXIT_FAILURE);                                    \
    }                                                               \
}
