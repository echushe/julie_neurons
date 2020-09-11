
#pragma once
#include <memory>
#include <vector>

namespace julie
{

namespace op
{

class Function;

class Variable
{
public:
    enum VariableType
    {
        TENSOR = 0,
        SCALAR = 1
    };

public:

    Variable ();
    Variable (const Variable & other);
    Variable (Variable && other);

    Variable & operator = (const Variable & other);
    Variable & operator = (Variable && other);

    bool trainable() const;
    void trainable(bool trainable);

    bool needs_grad() const;
    void needs_grad(bool grad);

    Function* get_provider() const;
    std::vector<Function*> get_receivers() const;

    void set_provider(Function *provider);
    void add_receiver(Function *reveiver);

/*
public:
    virtual ~Variable() {};
*/

public:
    // Virtual functions
    virtual void clear_grad() = 0;
    virtual void clear_val() = 0;

    virtual bool has_grad() const = 0;
    virtual bool has_val() const = 0;

    virtual VariableType data_type() const = 0;

protected:
    bool m_trainable;
    bool m_needs_grad;

    Function *m_provider;
    std::vector<Function*> m_receivers;
};

} // namespace op
} // namespace julie
