#include <iostream>
#include <string>
#include "conv2d.hpp"


int main(int argc, char **argv)
{
   if (argc > 1)
   {
      demo::conv2d_mnist(argv[1]);
   }
   else
   {
      std::cout << "Please run this program this way:\n";
      std::cout << "./conv_demo <Path_of_the_MNIST_dataset>" << std::endl;
   }
}