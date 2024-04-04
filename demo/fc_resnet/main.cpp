#include <iostream>
#include "fc.hpp"


int main(int argc, char **argv)
{
   if (argc > 1)
   {
      demo::fc_train_mnist(argv[1]);
   }
   else
   {
      std::cout << "Please run this program this way:\n";
      std::cout << "./fc_resnet_demo <Path_of_the_MNIST_dataset>" << std::endl;
   }
}