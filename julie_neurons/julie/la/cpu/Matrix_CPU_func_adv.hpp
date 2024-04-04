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

#pragma once
#include "Matrix_CPU.hpp"

namespace julie
{
namespace la
{
namespace cpu
{

/*********************************************************************************
 * Here are some advanced calculations of matrices via the iMatrix interface
 *********************************************************************************/

// Do padding for a 4-dimensional array like this:
// from (b, c, h, w) to (b, c, h + pad_h, w + pad_w)
// Arguments:
//     output: The output of padding
//     input:  The input
//     pad_h:  Size of padding along height dimension
//     pad_w:  Size of padding along weight dimension
// Returns: void
template <typename DT>
void pad_2d(Matrix_CPU<DT> & output, const Matrix_CPU<DT> & input, lint pad_h, lint pad_w);

// Do back-propagation for a 4-dimensional array like this:
// from (b, c, h + pad_h, w + pad_w) to (b, c, h, w)
// Arguments:
//     in_gradient: Gradient of padding input.
//     gradient:    Gradient of padding output.
//     pad_h:       Size of padding along height dimension
//     pad_w:       Size of padding along weight dimension
// Returns: void
template <typename DT>
void pad_2d_backward(Matrix_CPU<DT> & in_gradient, const Matrix_CPU<DT> & gradient, lint pad_h, lint pad_w);

// This is one unit of "Image to Rows" calculation. It generates one row from a rectangular area of input
// Arguments:
//     in_ch_size: Size of one channel of the input. It is input_height * input_width
//     in_w:       Input width
//     in_begin:   A pointer pointing to top-left corner of the area where the convolutional kernel will do dot product with
//     out_begin:  A pointer pointing to beginning of a new row of the output
//     in_ch:      Number of channels of the input
//     w_h:        Height of convolutional kernel
//     w_w:        Width of convolutional kernel
template <typename DT>
void __img2row_2d_row(lint in_ch_size, lint in_w, DT *in_begin, DT *out_begin, lint in_ch, lint w_h, lint w_w);

// "Image to Rows" is to convert an array of (b, c, h, w) into an array of (b, n_conv_outputs, c * w_h * w_w)
// in which n_conv_outputs == conv_output_h * conv_output_w
//                         == ((h - w_h) / stride_h + 1) * ((w - w_w) / stride_w + 1) 
// Arguments:
//     output:   Output of img2row
//     input:    Input
//     stride_h: Stride of convolution along height dimension
//     stride_w: Stride of convolution along width dimension
//     w_h:      Height of convolution kernel
//     w_w:      Width of convolution kernel
// Returns: void
template <typename DT>
void img2row_2d(Matrix_CPU<DT> & output, const Matrix_CPU<DT> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);

// This is one unit of "Image to Rows" backward calculation. It converts one row into a rectangular area.
// Arguments:
//     in_ch_size: Size of one channel of the input. It is input_height * input_width
//     in_w:       Input width
//     in_begin:   A pointer pointing to top-left corner of convolutional kernel area
//     out_begin:  A pointer pointing to beginning of a new row of gradient of convolutional output
//     in_ch:      Number of channels of convolutional input
//     w_h:        Height of convolutional kernel
//     w_w:        Width of convolutional kernel
template <typename DT>
void __img2row_2d_row_backward(lint in_ch_size, lint in_w, DT *in_begin, DT *out_begin, lint in_ch, lint w_h, lint w_w);

// This method is backward operation of "Image to Rows" converting the gradient of img2row output into
// the gradient of img2row input.
// Arguments:
//     in_gradient: Gradient of img2row input. Shape of in_gradient is the same as input of img2row.
//     in_shape:    Shape of img2row input
//     gradient:    Gradient of img2row output. Shape of gradient is the same as output of img2row
// Returns: void
template <typename DT>
void img2row_2d_backward(Matrix_CPU<DT> & in_gradient, const Shape & in_shape, const Matrix_CPU<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);

// "Image to Columns" is to convert an array of (b, c, h, w) into an array of (b, c * w_h * w_w, n_conv_outputs)
// where n_conv_outputs == conv_output_h * conv_output_w
//                      == ((h - w_h) / stride_h + 1) * ((w - w_w) / stride_w + 1)
// Arguments:
//     output:   Output of img2col
//     input:    Input
//     stride_h: Stride of convolution along height dimension
//     stride_w: Stride of convolution along width dimension
//     w_h:      Height of convolution kernel
//     w_w:      Width of convolution kernel
// Returns: void
template <typename DT>
void img2col_2d(Matrix_CPU<DT> & output, const Matrix_CPU<DT> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);

// This method is backward operation of "Image to Columns" converting the gradient of img2col output into
// the gradient of img2col input.
// Arguments:
//     in_gradient: Gradient of img2col input. Shape of in_gradient is the same as input of img2col.
//     in_shape:    Shape of img2col input
//     gradient:    Gradient of img2col output. Shape of gradient is the same as output of img2col
// Returns: void
template <typename DT>
void img2col_2d_backward(Matrix_CPU<DT> & in_gradient, const Shape & in_shape, const Matrix_CPU<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);

// Forward operation of max pooling. Suppose there is a input: (b, c, h, w), then shape of max pooling output will
// be: (b, c, out_h, out_w) where out_h = (h - k_h) / stride_h + 1, out_w = (w - k_w) / stride_w + 1
// Arguments:
//      output:   Output of max pooling, shape: (b, c, out_h, out_w)
//      diff:     Derivative of max pooling, shape: (b, c, out_h * k_h, out_w * k_w)
//      input:    input of max pooling, shape: (b, c, h, w)
//      stride_h: Stride of max pooling along height dimension
//      stride_w: Stride of max pooling along width dimension
//      k_h:      Height of max pooling kernel
//      k_w:      Width of max pooling kernel
// Returns: void
template <typename DT>
void maxpool_2d(Matrix_CPU<DT> &output, Matrix_CPU<DT> &diff, const Matrix_CPU<DT> &input,
                    lint stride_h, lint stride_w, lint k_h, lint k_w);

// Forward operation of average pooling. Suppose there is a input: (b, c, h, w), then shape of average pooling output will
// be: (b, c, out_h, out_w) where out_h = (h - k_h) / stride_h + 1, out_w = (w - k_w) / stride_w + 1
// Arguments:
//      output:   Output of average pooling, shape: (b, c, out_h, out_w)
//      diff:     Derivative of average pooling, shape: (b, c, out_h * k_h, out_w * k_w)
//      input:    input of average pooling, shape: (b, c, h, w)
//      stride_h: Stride of average pooling along height dimension
//      stride_w: Stride of average pooling along width dimension
//      k_h:      Height of average pooling kernel
//      k_w:      Width of average pooling kernel
// Returns: void
template <typename DT>
void avgpool_2d(Matrix_CPU<DT> &output, Matrix_CPU<DT> &diff, const Matrix_CPU<DT> &input,
                    lint stride_h, lint stride_w, lint k_h, lint k_w);

// Back propagation of pooling operations (max pooling, average pooling, etc)
// Arguments:
//     in_gradient:    Gradient of pooling input. Its shape is the same as pooling input: (b, c, h, w)
//     gradient_cache: Intermediate variable of back propagation. Its shape: (b, c, out_h * k_h, out_w * k_w)
//     in_shape:       Shape of pooling input: (b, c, h, w)
//     diff:           Derivative of pooling forward operation, it is of shape: (b, c, out_h * k_h, out_w * k_w)
//     gradient:       Gradient of pooling output. Its shape: (b, c, out_h, out_w) where
//                     out_h = (h - k_h) / stride_h + 1, out_w = (w - k_w) / stride_w + 1
//     stride_h:       Stride of pooling along height dimension
//     stride_w:       Stride of pooling along width dimension
//     k_h:            Height of pooling kernel
//     k_w:            Width of pooling kernel
// Returns: void
template <typename DT>
void pool_2d_backward(Matrix_CPU<DT> &in_gradient, Matrix_CPU<DT> &gradient_cache,
                        const Shape &in_shape, const Matrix_CPU<DT> &diff, const Matrix_CPU<DT> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w);

} // namespace cpu
} // namespace julie
} // namespace la