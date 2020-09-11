#pragma once
#include "function.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>

namespace julie
{

namespace op
{

class Graph
{
public:

    Graph () {}
    Graph (const Graph & other) = delete;
    Graph (Graph && other) = delete;
    Graph & operator = (const Graph & other) = delete;
    Graph & operator = (Graph && other) = delete; 

    void new_function(const std::shared_ptr<Function> & func);

    // Determine back propagation routes from the loss function to this trainable variable
    void pave_backward_route(const std::shared_ptr<Variable> & the_trainable);

    // Destroy all back propagation routes (remove all need_grad flags)
    void destroy_backward_route();


    void func_forward(const std::shared_ptr<Function> & func);

    void func_backward(const std::shared_ptr<Function> & func);

    // Clear all variable values except trainable weights
    void clear_forwards();

    // Clear all variable gradients
    void clear_backwards();

public:
    ~Graph() {};

private:

    void func_pave_backward_route(Function *func);

    void func_forward(Function *func);

    void func_backward(Function *func);

private:

    std::unordered_set<std::shared_ptr<Variable>> m_variables;
    std::unordered_set<std::shared_ptr<Function>> m_functions;
};


} // namespace op    
} // namespace julie
