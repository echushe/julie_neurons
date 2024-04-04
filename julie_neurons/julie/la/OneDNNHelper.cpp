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

#include "OneDNNHelper.hpp"

namespace julie
{
namespace la
{

std::shared_ptr<dnnl::engine> OneDNNEngineHelper::the_dnnl_engine = nullptr;

std::shared_ptr<dnnl::engine> OneDNNEngineHelper::get_dnnl_engine()
{
    if (!the_dnnl_engine)
    {
        //std::cout << "Create dnnl::engine !!!!" << std::endl;
        the_dnnl_engine = std::make_shared<dnnl::engine> (dnnl::engine::kind::cpu, 0);
    }

    return the_dnnl_engine;
}

}
}