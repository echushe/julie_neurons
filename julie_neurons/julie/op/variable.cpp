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

#include "variable.hpp"

namespace julie
{

namespace op
{

int Variable::ID_COUNT = 0;
std::mutex Variable::MUTEX;

int Variable::new_id()
{
    int new_id;
    MUTEX.lock();
    new_id = ID_COUNT;
    ++ID_COUNT;
    MUTEX.unlock();

    return new_id;
}

Variable::Variable ()
    :
    m_id {new_id()},
    m_trainable {false},
    m_forward_visit_started {false},
    m_forward_visited {false},
    m_backward_visit_started {false},
    m_backward_visited {false},
    m_provider {nullptr},
    m_receivers {}
{}

Variable::Variable (const Variable & other)
    :
    m_id {new_id()},
    m_trainable {other.m_trainable},
    m_forward_visit_started {other.m_forward_visit_started},
    m_forward_visited {other.m_forward_visited},
    m_backward_visit_started {other.m_backward_visit_started},
    m_backward_visited {other.m_backward_visited},
    m_provider {nullptr},
    m_receivers {}
{}

Variable::Variable (Variable && other)
    :
    m_id {new_id()},
    m_trainable {other.m_trainable},
    m_forward_visit_started {other.m_forward_visit_started},
    m_forward_visited {other.m_forward_visited},
    m_backward_visit_started {other.m_backward_visit_started},
    m_backward_visited {other.m_backward_visited},
    m_provider {nullptr},
    m_receivers {}
{}

Variable & Variable::operator = (const Variable & other)
{
    this->m_trainable = other.m_trainable;
    this->m_forward_visit_started = other.m_forward_visit_started;
    this->m_forward_visited = other.m_forward_visited;
    this->m_backward_visit_started = other.m_backward_visit_started;
    this->m_backward_visited = other.m_backward_visited;

    return *this;
}


Variable & Variable::operator = (Variable && other)
{
    this->m_trainable = other.m_trainable;
    this->m_forward_visit_started = other.m_forward_visit_started;
    this->m_forward_visited = other.m_forward_visited;
    this->m_backward_visit_started = other.m_backward_visit_started;
    this->m_backward_visited = other.m_backward_visited;

    return *this;
}

bool Variable::trainable() const
{
    return this->m_trainable;
}

void Variable::trainable(bool trainable)
{
    this->m_trainable = trainable;
}

std::shared_ptr<Function> Variable::get_provider() const
{
    return this->m_provider;
}

std::vector<std::shared_ptr<Function>> Variable::get_receivers() const
{
    return this->m_receivers;
}

void Variable::set_provider(const std::shared_ptr<Function> & provider)
{
    this->m_provider = provider;
}

void Variable::add_receiver(const std::shared_ptr<Function> & receiver)
{
    this->m_receivers.push_back(receiver);
}

bool Variable::remove_receiver(const std::shared_ptr<Function> & receiver)
{
    int idx_to_remove = -1;
    for (size_t i = 0; i < this->m_receivers.size(); ++i)
    {
        if (this->m_receivers[i] == receiver)
        {
            idx_to_remove = i;
            break;
        }
    }

    if (idx_to_remove == -1)
    {
        return false;
    }

    this->m_receivers.erase(this->m_receivers.begin() + idx_to_remove);
    return true;
}

void Variable::clear_provider()
{
    this->m_provider.reset();
}

int Variable::id() const
{
    return this->m_id;
}

void Variable::clear_forward_visit()
{
    this->m_forward_visit_started = false;
    this->m_forward_visited = false;
}

void Variable::clear_backward_visit()
{
    this->set_grad_to_zero();
    this->m_backward_visit_started = false;
    this->m_backward_visited = false;
}

bool Variable::forward_visit_started () const
{
    return this->m_forward_visit_started;
}

bool Variable::forward_visited() const
{
    return this->m_forward_visited;
}

bool Variable::backward_visit_started () const
{
    return this->m_backward_visit_started;
}

bool Variable::backward_visited() const
{
    return this->m_backward_visited;
}

void Variable::forward_visit_start()
{
    this->m_forward_visit_started = true;
    this->m_forward_visited = false;
}

void Variable::backward_visit_start()
{
    this->m_backward_visit_started = true;
    this->m_backward_visited = false;
}

std::ostream & operator << (std::ostream & os, const Variable & var)
{
    os << "var: " << var.id() << " type: " << var.data_type();

    return os;
}

} // namespace op    
} // namespace julie
