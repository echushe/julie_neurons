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

#include "function.hpp"

namespace julie
{

namespace op
{

int Function::ID_COUNT = 0;
std::mutex Function::MUTEX;

int Function::new_id()
{
    int new_id;
    MUTEX.lock();
    new_id = ID_COUNT;
    ++ID_COUNT;
    MUTEX.unlock();

    return new_id;
}

Function::Function(const std::string & type, bool is_loss_function)
    :
    m_id {new_id()},
    m_type {type},
    m_is_loss_function {is_loss_function},
    m_forward_visit_started {false},
    m_forward_visited {false},
    m_backward_visit_started {false},
    m_backward_visited {false},
    m_inputs {},
    m_output {nullptr}
{}

Function::Function(const Function & other)
    :
    m_id {new_id()},
    m_type {other.m_type},
    m_is_loss_function {other.m_is_loss_function},
    m_forward_visit_started {other.m_forward_visit_started},
    m_forward_visited {other.m_forward_visited},
    m_backward_visit_started {other.m_backward_visit_started},
    m_backward_visited {other.m_backward_visited},
    m_inputs {},
    m_output {nullptr}
{}

Function::Function(Function && other)
    :
    m_id {new_id()},
    m_type {std::move(other.m_type)},
    m_is_loss_function {other.m_is_loss_function},
    m_forward_visit_started {other.m_forward_visit_started},
    m_forward_visited {other.m_forward_visited},
    m_backward_visit_started {other.m_backward_visit_started},
    m_backward_visited {other.m_backward_visited},
    m_inputs {},
    m_output {nullptr}
{}
    
Function & Function::operator = (const Function & other)
{
    this->m_type = other.m_type;
    this->m_is_loss_function = other.m_is_loss_function;
    this->m_forward_visit_started = other.m_forward_visit_started;
    this->m_forward_visited = other.m_forward_visited;
    this->m_backward_visit_started = other.m_backward_visit_started;
    this->m_backward_visited = other.m_backward_visited;

    return *this;
}

Function & Function::operator = (Function && other)
{
    this->m_type = std::move(other.m_type);
    this->m_is_loss_function = other.m_is_loss_function;
    this->m_forward_visit_started = other.m_forward_visit_started;
    this->m_forward_visited = other.m_forward_visited;
    this->m_backward_visit_started = other.m_backward_visit_started;
    this->m_backward_visited = other.m_backward_visited;
     
    return *this;
}

std::vector<std::shared_ptr<Variable>> Function::get_inputs() const
{
    return this->m_inputs;
}

std::shared_ptr<Variable> Function::get_output() const
{
    return this->m_output;
}

void Function::set_inputs(const std::shared_ptr<Function> & self, const std::vector<std::shared_ptr<Variable>> & inputs)
{
    if (self.get() != this)
    {
        throw std::invalid_argument(
            std::string("Function::set_inputs: Self argument should be the same instance as \"this\" pointer in ") + 
            std::string(__FUNCTION__));
    }

    this->m_inputs = inputs;

    for (auto & input : inputs)
    {
        input->add_receiver(self);
    }

    if (this->m_output)
    {
        this->m_output->set_provider(self);
    }
}

void Function::clear_inputs(const std::shared_ptr<Function> & self)
{
    if (self.get() != this)
    {
        throw std::invalid_argument(
            std::string("Function::clear_inputs: Self argument should be the same instance as \"this\" pointer in ") + 
            std::string(__FUNCTION__));
    }

    for (auto &input : this->m_inputs)
    {
        input->remove_receiver(self);
    }

    this->m_inputs.clear();
}

int Function::id() const
{
    return this->m_id;
}

std::string Function::type() const
{
    return this->m_type;
}

bool Function::is_loss_function() const
{
    return this->m_is_loss_function;
}

// Set this function to loss function
Function & Function::to_loss_function()
{
    this->m_is_loss_function = true;
    return *this;
}

// Set this function to non loss function
Function & Function::to_non_loss_function()
{
    this->m_is_loss_function = false;
    return *this;
}

void Function::clear_forward_visit()
{
    // this->clear_cache();
    this->m_forward_visit_started = false;
    this->m_forward_visited = false;
}

void Function::clear_backward_visit()
{
    // this->clear_cache();
    this->m_backward_visit_started = false;
    this->m_backward_visited = false;
}

void Function::forward_visit_start()
{
    this->m_forward_visit_started = true;
    this->m_forward_visited = false;
}

void Function::forward_visit_finish()
{
    this->m_forward_visit_started = false;
    this->m_forward_visited = true;
}

void Function::backward_visit_start()
{
    this->m_backward_visit_started = true;
    this->m_backward_visited = false;
}

void Function::backward_visit_finish()
{
    this->m_backward_visit_started = false;
    this->m_backward_visited = true;
}

bool Function::forward_visit_started() const
{
    return this->m_forward_visit_started;
}

bool Function::forward_visited() const
{
    return this->m_forward_visited;
}

bool Function::backward_visit_started() const
{
    return this->m_backward_visit_started;
}

bool Function::backward_visited() const
{
    return this->m_backward_visited;
}

std::ostream & operator << (std::ostream & os, const Function & func)
{
    os << "func: " << func.id() << " type: " <<  func.type();
    return os;
}

} // namespace op
} // namespace julie