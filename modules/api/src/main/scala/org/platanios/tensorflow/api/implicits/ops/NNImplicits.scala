/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.implicits.ops

import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.ops.NN.{CNNDataFormat, ConvPaddingMode}
import org.platanios.tensorflow.api.ops.{NN, Output}
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault

trait NNImplicits {
  implicit def outputConvertibleToNNOps[T, OC](
      value: OC
  )(implicit f: OC => Output[T]): NNOps[T] = {
    new NNOps(f(value))
  }

  implicit class NNOps[T](val output: Output[T]) {
    protected implicit val evTTF: TF[T] = {
      TF.fromDataType(output.dataType)
    }

    //region Core Ops

    /** $OpDocNNAddBias
      *
      * @group NNOps
      * @param  bias          Bias tensor that must be one-dimensional (i.e., it must have rank 1).
      * @param  cNNDataFormat Data format of the input and output tensors. With the default format [[NWCFormat]], the
      *                       `bias` tensor will be added to the last dimension of the `value` tensor. Alternatively, the
      *                       format could be [[NCWFormat]], and the `bias` tensor would be added to the third-to-last
      *                       dimension.
      * @return Created op output.
      */
    def addBias(
        bias: Output[T],
        cNNDataFormat: CNNDataFormat = CNNDataFormat.default
    )(implicit ev: IsNumeric[T]): Output[T] = {
      NN.addBias(output, bias, cNNDataFormat)
    }

    /** $OpDocNNLinear
      *
      * @group NNOps
      * @param  weights Weights tensor.
      * @param  bias    Bias tensor.
      * @return Created op output.
      */
    def linear(
        weights: Output[T],
        bias: Output[T]
    )(implicit ev: IsNotQuantized[T]): Output[T] = {
      NN.linear(output, weights, bias)
    }

    /** $OpDocNNL2Normalize
      *
      * @group NNOps
      * @param  axes    Tensor containing the axes along which to normalize.
      * @param  epsilon Lower bound value for the norm. The created op will use `sqrt(epsilon)` as the divisor, if
      *                 `norm < sqrt(epsilon)`.
      * @return Created op output.
      */
    def l2Normalize[I: TF : IsIntOrLong](
        axes: Output[I],
        epsilon: Float = 1e-12f
    )(implicit ev: IsNotQuantized[T]): Output[T] = {
      NN.l2Normalize(output, axes, epsilon)
    }

    //endregion Core Ops

    //region Activation Ops

    /** $OpDocNNRelu
      *
      * @group NNOps
      * @param  alpha Slope of the negative section, also known as leakage parameter. If other than `0.0f`, the negative
      *               part will be equal to `alpha * x` instead of `0`. Defaults to `0`.
      * @return Created op output.
      */
    def relu(
        alpha: Float = 0.0f
    )(implicit ev: IsReal[T]): Output[T] = {
      NN.relu(output, alpha)
    }

    /** $OpDocNNRelu6
      *
      * @group NNOps
      * @return Created op output.
      */
    def relu6(implicit ev: IsReal[T]): Output[T] = {
      NN.relu6(output)
    }

    /** $OpDocNNCrelu
      *
      * @group NNOps
      * @return Created op output.
      */
    def crelu(implicit ev: IsReal[T]): Output[T] = {
      NN.crelu(output)
    }

    /** $OpDocNNElu
      *
      * @group NNOps
      * @return Created op output.
      */
    def elu(implicit ev: IsReal[T]): Output[T] = {
      NN.elu(output)
    }

    /** $OpDocNNSelu
      *
      * @group NNOps
      * @return Created op output.
      */
    def selu(implicit ev: IsReal[T]): Output[T] = {
      NN.selu(output)
    }

    /** $OpDocNNSoftplus
      *
      * @group NNOps
      * @return Created op output.
      */
    def softplus(implicit ev: IsDecimal[T]): Output[T] = {
      NN.softplus(output)
    }

    /** $OpDocNNSoftsign
      *
      * @group NNOps
      * @return Created op output.
      */
    def softsign(implicit ev: IsDecimal[T]): Output[T] = {
      NN.softsign(output)
    }

    /** $OpDocNNSoftmax
      *
      * @group NNOps
      * @param  axis Axis along which to perform the softmax. Defaults to `-1` denoting the last axis.
      * @return Created op output.
      */
    def softmax(axis: Int = -1)(implicit ev: IsDecimal[T]): Output[T] = {
      NN.softmax(output, axis)
    }

    /** $OpDocNNLogSoftmax
      *
      * @group NNOps
      * @param  axis Axis along which to perform the log-softmax. Defaults to `-1` denoting the last axis.
      * @return Created op output.
      */
    def logSoftmax(axis: Int = -1)(implicit ev: IsDecimal[T]): Output[T] = {
      NN.logSoftmax(output, axis)
    }

    //endregion Activation Ops

    /** $OpDocNNDropout
      *
      * @group NNOps
      * @param  keepProbability Probability (i.e., number in the interval `(0, 1]`) that each element is kept.
      * @param  scaleOutput     If `true`, the outputs will be divided by the keep probability.
      * @param  noiseShape      Rank-1 tensor representing the shape for the randomly generated keep/drop flags.
      * @param  seed            Optional random seed, used to generate a random seed pair for the random number
      *                         generator, when combined with the graph-level seed.
      * @return Created op output that has the same shape as `input`.
      */
    def dropout[I: IntDefault : IsIntOrLong : TF](
        keepProbability: Float,
        scaleOutput: Boolean = true,
        noiseShape: Output[I] = null,
        seed: Option[Int] = None
    )(implicit ev: IsHalfOrFloatOrDouble[T]): Output[T] = {
      NN.dropout(output, keepProbability, scaleOutput, noiseShape, seed)
    }

    /** $OpDocNNDropout
      *
      * @group NNOps
      * @param  keepProbability Probability (i.e., number in the interval `(0, 1]`) that each element is kept.
      * @param  scaleOutput     If `true`, the outputs will be divided by the keep probability.
      * @param  noiseShape      Rank-1 tensor representing the shape for the randomly generated keep/drop flags.
      * @param  seed            Optional random seed, used to generate a random seed pair for the random number
      *                         generator, when combined with the graph-level seed.
      * @return Created op output that has the same shape as `input`.
      */
    def dynamicDropout[I: IntDefault : TF : IsIntOrLong](
        keepProbability: Output[T],
        scaleOutput: Boolean = true,
        noiseShape: Output[I] = null,
        seed: Option[Int] = None
    )(implicit ev: IsHalfOrFloatOrDouble[T]): Output[T] = {
      NN.dynamicDropout(output, keepProbability, scaleOutput, noiseShape, seed)
    }

    /** $OpDocNNTopK
      *
      * @group NNOps
      * @param  k      Scalar tensor containing the number of top elements to look for along the last axis of `input`.
      * @param  sorted If `true`, the resulting `k` elements will be sorted by their values in descending order.
      * @return Tuple containing the created op outputs: (i) `values`: the `k` largest elements along each last
      *         dimensional slice, and (ii) `indices`: the indices of `values` within the last axis of `input`.
      */
    def topK(
        k: Output[Int],
        sorted: Boolean = true
    )(implicit ev: IsReal[T]): (Output[T], Output[Int]) = {
      NN.topK(output, k, sorted)
    }

    /** $OpDocNNInTopK
      *
      * @group NNOps
      * @param  targets Tensor containing the targets.
      * @param  k       Scalar tensor containing the number of top elements to look at.
      * @return Created op output.
      */
    def inTopK[I: TF : IsIntOrLong](
        targets: Output[I],
        k: Output[I]
    )(implicit ev: T =:= Float): Output[Boolean] = {
      NN.inTopK(output.asInstanceOf[Output[Float]], targets, k)
    }

    //region Convolution Ops

    /** $OpDocNNConv2D
      *
      * @param  filter        4-D tensor with shape `[filterHeight, filterWidth, inChannels, outChannels]`.
      * @param  stride1       Stride of the sliding window along the second dimension of this tensor.
      * @param  stride2       Stride of the sliding window along the third dimension of this tensor.
      * @param  padding       Padding mode to use.
      * @param  dataFormat    Format of the input and output data.
      * @param  dilations     The dilation factor for each dimension of input. If set to `k > 1`, there will be `k - 1`
      *                       skipped cells between each filter element on that dimension. The dimension order is
      *                       determined by the value of `dataFormat`. Dilations in the batch and depth dimensions must
      *                       be set to `1`.
      * @param  useCuDNNOnGPU Boolean value indicating whether or not to use CuDNN for the created op, if its placed on
      *                       a GPU, as opposed to the TensorFlow implementation.
      * @param  name          Name for the created op.
      * @return Created op output, which is a 4-D tensor whose dimension order depends on the value of `dataFormat`.
      */
    def conv2D(
        filter: Output[T],
        stride1: Long,
        stride2: Long,
        padding: ConvPaddingMode,
        dataFormat: CNNDataFormat = CNNDataFormat.default,
        // TODO: [OPS/NN] Enforce the batch and depth dilation constraint at compile time.
        dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
        useCuDNNOnGPU: Boolean = true,
        name: String = "Conv2D"
    )(implicit ev: IsDecimal[T]): Output[T] = {
      NN.conv2D(output, filter, stride1, stride2, padding, dataFormat, dilations, useCuDNNOnGPU, name)
    }

    //endregion Convolution Ops

    //region Pooling Ops

    /** $OpDocNNMaxPool
      *
      * @param  windowSize The size of the pooling window for each dimension of the input tensor.
      * @param  strides    Strides for the sliding window. Strides in the batch and depth dimensions must be set to `1`.
      * @param  padding    Padding mode to use.
      * @param  dataFormat Format of the input and output data.
      * @param  name       Name for the created op.
      * @return Created op output, which is a 4-D tensor whose dimension order depends on the value of `dataFormat`.
      */
    def maxPool(
        windowSize: Output[Int],
        // TODO: [OPS|NN] Enforce the batch and depth stride constraint at compile time.
        strides: Output[Int],
        padding: ConvPaddingMode,
        dataFormat: CNNDataFormat = CNNDataFormat.default,
        name: String = "MaxPool"
    )(implicit ev: IsNumeric[T]): Output[T] = {
      NN.maxPool(output, windowSize, strides, padding, dataFormat, name)
    }

    //endregion Pooling Ops

    //region Normalization Ops

    /** $OpDocNNLocalResponseNormalization
      *
      * @group NNOps
      * @param  depthRadius Half-width of the 1-D normalization window.
      * @param  bias        Offset (usually positive to avoid dividing by 0).
      * @param  alpha       Scale factor (usually positive).
      * @param  beta        Exponent.
      * @param  name        Name for the created op.
      * @return Created op output.
      */
    def lrn(
        depthRadius: Int = 5,
        bias: Float = 1.0f,
        alpha: Float = 1.0f,
        beta: Float = 0.5f,
        name: String = "LRN"
    )(implicit ev: IsTruncatedHalfOrHalfOrFloat[T]): Output[T] = {
      NN.localResponseNormalization(output, depthRadius, bias, alpha, beta, name)
    }

    /** $OpDocNNLocalResponseNormalization
      *
      * @group NNOps
      * @param  depthRadius Half-width of the 1-D normalization window.
      * @param  bias        Offset (usually positive to avoid dividing by 0).
      * @param  alpha       Scale factor (usually positive).
      * @param  beta        Exponent.
      * @param  name        Name for the created op.
      * @return Created op output.
      */
    def localResponseNormalization(
        depthRadius: Int = 5,
        bias: Float = 1.0f,
        alpha: Float = 1.0f,
        beta: Float = 0.5f,
        name: String = "LocalResponseNormalization"
    )(implicit ev: IsTruncatedHalfOrHalfOrFloat[T]): Output[T] = {
      NN.localResponseNormalization(output, depthRadius, bias, alpha, beta, name)
    }

    //endregion Normalization Ops
  }
}
