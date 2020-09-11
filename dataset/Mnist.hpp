#pragma once
#include "Dataset.hpp"
#include <string>
#include <vector>



namespace dataset
{
    class Mnist : public Dataset
    {
    private:
        std::string m_sample_file;
        std::string m_label_file;

    private:
        /*!
        * \brief Extract the MNIST header from the given buffer
        * \param buffer The current buffer
        * \param position The current reading positoin
        * \return The value of the mnist header
        */
        int64_t read_header(const std::unique_ptr<char[]>& buffer, size_t position) const;

        /*!
        * \brief Read a MNIST file inside a raw buffer
        * \param path The path to the image file
        * \return The buffer of byte on success, a nullptr-unique_ptr otherwise
        */
        inline std::unique_ptr<char[]> read_mnist_file(const std::string & path, uint32_t key) const;

        void read_mnist_image_file(std::vector<julie::la::DMatrix<double>> & images, const std::string& path, lint limit = 0) const;

        void read_mnist_label_file(std::vector<julie::la::DMatrix<double>> & labels, const std::string& path, lint limit = 0) const;

    public:

        Mnist(
            const std::string & sample_file,
            const std::string & label_file);

        virtual void get_samples_and_labels(
            std::vector<julie::la::DMatrix<double>> & inputs, std::vector<julie::la::DMatrix<double>> & labels, lint limit = 0) const;
    };
}
