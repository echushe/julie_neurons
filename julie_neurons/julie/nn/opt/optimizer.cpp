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

#include "optimizer.hpp"

namespace julie
{
namespace nn
{
namespace opt
{

Optimizer::Optimizer (const std::vector<std::shared_ptr<op::Variable>> & params)
    : m_params {params}
{}

Optimizer::Optimizer (const op::Graph & graph)
    : m_params {graph.all_trainable_variables()}
{}

Optimizer::Optimizer (const Optimizer & other)
    : m_params {other.m_params}
{}

Optimizer::Optimizer (Optimizer && other)
    : m_params {std::move(other.m_params)}
{}

Optimizer & Optimizer::operator = (const Optimizer & other)
{
    this->m_params = other.m_params;
    return *this;
}

Optimizer & Optimizer::operator = (Optimizer && other)
{
    this->m_params = std::move(other.m_params);
    return *this;
}

} // namespace opt
} // namespace nn
} // namespace julie

