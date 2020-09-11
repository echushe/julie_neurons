#pragma once

#include <exception>
#include <stdexcept>
#include <chrono>

typedef long long lint;
namespace test
{
    static lint get_time_in_milliseconds()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::system_clock::now().time_since_epoch()).count();
    }

    static void ASSERT(bool expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }

    static void ASSERT(char expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }

    static void ASSERT(unsigned char expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }

    static void ASSERT(int expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }

    static void ASSERT(unsigned int expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }

    static void ASSERT(long expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }

    static void ASSERT(unsigned long expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }

    static void ASSERT(long long expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }

    static void ASSERT(unsigned long long expr)
    {
        if (!expr)
        {
            throw std::invalid_argument(std::string("ASSERT FAILURE"));
        }
    }
}