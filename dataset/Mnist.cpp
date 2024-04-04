#include "Mnist.hpp"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <memory>

/*!
* \brief Extract the MNIST header from the given buffer
* \param buffer The current buffer
* \param position The current reading positoin
* \return The value of the mnist header
*/
int64_t dataset::Mnist::read_header(const std::unique_ptr<char[]>& buffer, size_t position) const
{
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position);
    auto decode = (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);

    return static_cast<lint>(decode);
}

/*!
* \brief Read a MNIST file inside a raw buffer
* \param path The path to the image file
* \return The buffer of byte on success, a nullptr-unique_ptr otherwise
*/
inline std::unique_ptr<char[]> dataset::Mnist::read_mnist_file(const std::string & path, uint32_t key) const
{
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (!file)
    {
        std::cout << "Error opening file " << path << std::endl;
        throw std::invalid_argument(std::string("Path or file not found"));
    }

    auto size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[size]);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), size);
    file.close();

    auto magic = read_header(buffer, 0);

    // std::cout << key << std::endl;
    // std::cout << magic << std::endl;

    if (magic != key)
    {
        std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        return {};
    }

    auto count = this->read_header(buffer, 1);

    if (magic == 0x803)
    {
        auto rows = read_header(buffer, 2);
        auto columns = read_header(buffer, 3);

        if (size < count * rows * columns + 16) {
            std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            return {};
        }
    }
    else if (magic == 0x801)
    {
        if (size < count + 8)
        {
            std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            return {};
        }
    }

    return buffer;
}


void dataset::Mnist::read_mnist_image_file(std::vector<std::shared_ptr<julie::la::iMatrix<float>>> & images, const std::string& path, lint limit) const
{
    auto buffer = this->read_mnist_file(path, 0x803);

    if (buffer)
    {
        auto count = this->read_header(buffer, 1);
        auto rows = this->read_header(buffer, 2);
        auto columns = this->read_header(buffer, 3);

        if (limit > 0 && count > limit)
        {
            count = limit;
        }

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        uint8_t* image_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 16);
        lint image_len = rows * columns;

        uint8_t* image_buffer_p = image_buffer;

        for (lint i = 0; i < count; ++i)
        {
            julie::la::cpu::Matrix_CPU<float> new_image{ julie::la::Shape{ rows, columns } };

            for (lint j = 0; j < image_len; ++j)
            {
                new_image.m_data[j] = image_buffer_p [j];
            }

            images.push_back(std::make_shared<julie::la::iMatrix<float>>(std::move(new_image)));

            image_buffer_p += image_len;
        }
    }
}


void dataset::Mnist::read_mnist_label_file(std::vector<std::shared_ptr<julie::la::iMatrix<float>>> & labels, const std::string& path, lint limit) const
{
    auto buffer = read_mnist_file(path, 0x801);

    if (buffer)
    {
        auto count = this->read_header(buffer, 1);

        //Skip the header
        //Cast to unsigned char is necessary cause signedness of char is
        //platform-specific
        uint8_t* label_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 8);

        if (limit > 0 && count > limit)
        {
            count = limit;
        }

        uint8_t max_val = 0;
        for (lint i = 0; i < count; ++i)
        {
            uint8_t label_val = *(label_buffer + i);
            if (label_val > max_val)
            {
                max_val = label_val;
            }
        }

        for (lint i = 0; i < count; ++i)
        {
            julie::la::cpu::Matrix_CPU<float> label{ julie::la::Shape{ max_val + 1 } };
            uint8_t label_val = *(label_buffer + i);

            for (uint8_t j = 0; j <= max_val; ++j)
            {
                if (j == label_val)
                {
                    // label[{j}] = 1;
                    label.m_data[j] = 1;
                }
                else
                {
                    // label[{j}] = 0;
                    label.m_data[j] =  0;
                }
            }

            labels.push_back(std::make_shared<julie::la::iMatrix<float>>(std::move(label)));
        }
    }
}


dataset::Mnist::Mnist(
    const std::string & sample_file,
    const std::string & label_file)
    :
    m_sample_file{ sample_file },
    m_label_file{ label_file }
{}


void dataset::Mnist::get_samples_and_labels(
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> & inputs,
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> & labels,
    lint limit) const
{
    this->read_mnist_image_file(inputs, this->m_sample_file, limit);
    this->read_mnist_label_file(labels, this->m_label_file, limit);
}


