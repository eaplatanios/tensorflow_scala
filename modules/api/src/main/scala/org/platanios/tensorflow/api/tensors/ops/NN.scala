/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.tensors.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.NN.{CNNDataFormat, ConvPaddingMode, NCWFormat, NWCFormat}
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.generated.tensors.{NN => NativeTensorOpsNN}

import java.nio.charset.StandardCharsets

/** Contains functions for executing ops related to neural networks.
  *
  * @author Emmanouil Antonios Platanios
  */
trait NN {
  //region Core Ops

  /** $OpDocNNAddBias
    *
    * @group NNOps
    * @param  value         Value tensor.
    * @param  bias          Bias tensor that must be one-dimensional (i.e., it must have rank 1).
    * @param  cNNDataFormat Data format of the input and output tensors. With the default format [[NWCFormat]], the
    *                       `bias` tensor will be added to the last dimension of the `value` tensor. Alternatively, the
    *                       format could be [[NCWFormat]], and the `bias` tensor would be added to the third-to-last
    *                       dimension.
    * @return Result as a new tensor.
    */
  def addBias[T: IsNotQuantized](
      value: Tensor[T],
      bias: Tensor[T],
      cNNDataFormat: CNNDataFormat = CNNDataFormat.default
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.biasAdd(
      executionContext.value.nativeHandle, value.nativeHandle, bias.nativeHandle,
      cNNDataFormat.toString.getBytes(StandardCharsets.ISO_8859_1)))
  }

  /** $OpDocNNLinear
    *
    * @group NNOps
    * @param  x       Input tensor.
    * @param  weights Weights tensor.
    * @param  bias    Bias tensor.
    * @return Result as a new tensor.
    */
  def linear[T: IsNotQuantized](x: Tensor[T], weights: Tensor[T], bias: Tensor[T] = null): Tensor[T] = {
    val product = {
      if (x.rank > 2)
        Math.tensorDot(x, weights, Seq(x.rank - 1), Seq(0))
      else
        Math.matmul(x, weights)
    }
    if (bias != null)
      addBias(product, bias)
    else
      product
  }

  /** $OpDocNNL2Normalize
    *
    * @group NNOps
    * @param  x       Input tensor.
    * @param  axes    Tensor containing the axes along which to normalize.
    * @param  epsilon Lower bound value for the norm. The created op will use `sqrt(epsilon)` as the divisor, if
    *                 `norm < sqrt(epsilon)`.
    * @return Result as a new tensor.
    */
  def l2Normalize[T: IsFloat32OrFloat64](
      x: Tensor[T],
      axes: Tensor[Int],
      epsilon: Float = 1e-12f
  ): Tensor[T] = {
    val squareSum = Math.sum(Math.square(x), axes = axes, keepDims = true)
    val xInverseNorm = Math.rsqrt(Math.maximum(squareSum, epsilon.toTensor.cast(x.dataType)))
    Math.multiply(x, xInverseNorm)
  }

  //endregion Core Ops

  //region Activation Ops

  /** $OpDocNNRelu
    *
    * @group NNOps
    * @param  x     Input tensor.
    * @param  alpha Slope of the negative section, also known as leakage parameter. If other than `0.0f`, the negative
    *               part will be equal to `alpha * x` instead of `0`. Defaults to `0`.
    * @return Result as a new tensor.
    */
  def relu[T: IsReal](x: Tensor[T], alpha: Float = 0.0f): Tensor[T] = {
    def reluOp[TL[A] <: TensorLike[A]](i: TL[T])(implicit ev: TensorOps.Aux[TL, T]): TL[T] = {
      ev.applyUnary(i, t => {
        Tensor.fromNativeHandle[T](NativeTensorOpsNN.relu(executionContext.value.nativeHandle, t.nativeHandle))
      })
    }

    if (alpha == 0.0) {
      reluOp(x)
    } else {
      val positive = reluOp(x)
      val negative = reluOp(-x)
      positive - (alpha.toTensor.cast(negative.dataType) * negative)
    }
  }

  /** $OpDocNNRelu6
    *
    * @group NNOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def relu6[T: IsReal, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsNN.relu6(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocNNCrelu
    *
    * @group NNOps
    * @param  x    Input tensor.
    * @param  axis Along along which the output values are concatenated along.
    * @return Result as a new tensor.
    */
  def crelu[T: IsReal](x: Tensor[T], axis: Tensor[Int] = -1): Tensor[T] = {
    relu(Basic.concatenate(Seq(x, -x), axis = axis))
  }

  /** $OpDocNNElu
    *
    * @group NNOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def elu[T: IsDecimal, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsNN.elu(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocNNSelu
    *
    * @group NNOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def selu[T: IsDecimal, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsNN.selu(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocNNSoftplus
    *
    * @group NNOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def softplus[T: IsReal, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsNN.softplus(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocNNSoftsign
    *
    * @group NNOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def softsign[T: IsReal](input: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.softsign(executionContext.value.nativeHandle, input.nativeHandle))
  }

  //endregion Activation Ops

  /** Helper function for [[softmax]] and [[logSoftmax]] that reshapes and transposes the input logits into
    * two-dimensional tensors and then creates the corresponding native op. The output is transposed and reshaped
    * back. */
  private[this] def softmaxHelper[T: IsDecimal](
      logits: Tensor[T],
      opFunction: Long => Long,
      axis: Int = -1
  ): Tensor[T] = {
    // We need the original shape of the logits for shape inference.
    val shape = logits.shape
    val isLastAxis = axis == -1 || axis == shape.rank - 1
    if (shape.rank == 2 && isLastAxis) {
      Tensor.fromNativeHandle[T](opFunction(logits.nativeHandle))
    } else if (isLastAxis) {
      // If axis is the last axis, we simply reshape the logits to a matrix and apply the internal softmax.
      val inputShape = Basic.shape(logits, INT64)
      val flattenedLogits = NN.flattenOuterAxes(logits)
      val output = Tensor.fromNativeHandle[T](opFunction(flattenedLogits.nativeHandle))
      Basic.reshape(output, inputShape)
    } else {
      // If axis is not the last dimension, we have to do a reshape and transpose so that we can still perform softmax
      // on its last dimension.
      // We swap the logits' axis of axis and its last axis.
      val inputRank = Basic.rank(logits)
      val swappedLogits = NN.swapAxes(logits, axis, Math.subtract(inputRank, 1))
      val shapeAfterSwap = Basic.shape(swappedLogits, INT64)
      // We reshape the logits into a matrix.
      val flattenedLogits = NN.flattenOuterAxes(swappedLogits)
      // We perform the actual softmax on the last axis.
      var output = Tensor.fromNativeHandle[T](opFunction(flattenedLogits.nativeHandle))
      // We transform back the output tensor.
      output = Basic.reshape(output, shapeAfterSwap)
      output = NN.swapAxes(output, axis, Math.subtract(inputRank, 1))
      output
    }
  }

  /** $OpDocNNSoftmax
    *
    * @group NNOps
    * @param  logits Tensor containing the logits.
    * @param  axis   Axis along which to perform the softmax. Defaults to `-1` denoting the last axis.
    * @return Result as a new tensor.
    */
  def softmax[T: IsDecimal](logits: Tensor[T], axis: Int = -1): Tensor[T] = {
    softmaxHelper(logits, NativeTensorOpsNN.softmax(executionContext.value.nativeHandle, _), axis)
  }

  /** $OpDocNNLogSoftmax
    *
    * @group NNOps
    * @param  logits Tensor containing the logits.
    * @param  axis   Axis along which to perform the log-softmax. Defaults to `-1` denoting the last axis.
    * @return Result as a new tensor.
    */
  def logSoftmax[T: IsDecimal](logits: Tensor[T], axis: Int = -1): Tensor[T] = {
    softmaxHelper(logits, NativeTensorOpsNN.logSoftmax(executionContext.value.nativeHandle, _), axis)
  }

  //region Loss Ops

  /** $OpDocNNL2Loss
    *
    * @group NNOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def l2Loss[T: IsDecimal](input: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.l2Loss(executionContext.value.nativeHandle, input.nativeHandle))
  }

  private[this] def nativeCrossEntropyProxy(
      features: Long,
      labels: Long,
      function: (Long, Long, Long) => Array[Long]
  ): Array[Long] = {
    function(executionContext.value.nativeHandle, features, labels)
  }

  /** $OpDocNNSoftmaxCrossEntropy
    *
    * @group NNOps
    * @param  logits Tensor of shape `[D0, D1, ..., Dr-1, numClasses]`, containing unscaled log probabilities.
    * @param  labels Tensor of shape `[D0, D1, ..., Dr-1, numClasses]`, where each row must be a valid probability
    *                distribution.
    * @param  axis   The class axis, along which the softmax is computed. Defaults to `-1`, which is the last axis.
    * @return Result as a new tensor, with rank one less than that of `logits` and the same data type as `logits`,
    *         containing the softmax cross entropy loss.
    */
  def softmaxCrossEntropy[T: IsDecimal](
      logits: Tensor[T],
      labels: Tensor[T],
      axis: Int = -1
  ): Tensor[T] = {
    val inputRank = Basic.rank(logits)
    // We move axis to the end, if it's not already the last axis.
    val transposedLogits = NN.moveAxisToEnd(logits, axis, inputRank)
    val transposedLabels = NN.moveAxisToEnd(labels, axis, inputRank)
    val inputShape = Basic.shape(logits, INT64)
    // Flatten transposedLogits and transposedLabels into matrices.
    val flattenedLogits = NN.flattenOuterAxes(transposedLogits)
    val flattenedLabels = NN.flattenOuterAxes(transposedLabels)
    // Create the native op.
    // The second output tensor contains the gradients, which is used for the gradient computation.
    val output = Tensor.fromNativeHandle[T](nativeCrossEntropyProxy(
      flattenedLogits.nativeHandle, flattenedLabels.nativeHandle,
      NativeTensorOpsNN.softmaxCrossEntropyWithLogits).apply(0))
    // The output shape should be the input shape without the axis over which the cross entropy was computed.
    val outputShape = Basic.slice(
      inputShape,
      Tensor.fill(INT64, Shape(1))(0L),
      Basic.expandDims(Math.subtract(inputRank.toInt64, 1L), -1))
    Basic.reshape(output, outputShape)
  }

  /** $OpDocNNSparseSoftmaxCrossEntropy
    *
    * @group NNOps
    * @param  logits Tensor of shape `[D0, D1, ..., Dr-1, numClasses]` (where `r` is the rank of `labels` and of the
    *                result), containing unscaled log probabilities.
    * @param  labels Tensor of shape `[D0, D1, ..., Dr-1]` (where `r` is the rank of `labels` and of the result). Each
    *                entry in `labels` must be an index in `[0, numClasses)`. Other values will raise an exception when
    *                this op is run on a CPU, and return `NaN` values for the corresponding loss and gradient rows when
    *                this op is run on a GPU.
    * @param  axis   The class axis, along which the softmax is computed. Defaults to `-1`, which is the last axis.
    * @return Result as a new tensor, with the same shape as `labels` and the same data type as `logits`, containing the
    *         softmax cross entropy loss.
    */
  def sparseSoftmaxCrossEntropy[T: IsDecimal, I: IsInt32OrInt64](
      logits: Tensor[T],
      labels: Tensor[I],
      axis: Int = -1
  ): Tensor[T] = {
    // Check if no reshapes are required.
    if (logits.rank == 2) {
      // Create the native op.
      // The second output tensor contains the gradients, which is used for the gradient computation.
      Tensor.fromNativeHandle[T](nativeCrossEntropyProxy(
        logits.nativeHandle, labels.nativeHandle,
        NativeTensorOpsNN.sparseSoftmaxCrossEntropyWithLogits).apply(0))
    } else {
      // Reshape logits to rank 2 and labels to rank 1.
      val flattenedLogits = NN.flattenOuterAxes(logits)
      val flattenedLabels = Basic.reshape(labels, -1)
      // Create the native op.
      // The second output tensor contains the gradients, which is used for the gradient computation.
      val output = Tensor.fromNativeHandle[T](nativeCrossEntropyProxy(
        flattenedLogits.nativeHandle, flattenedLabels.nativeHandle,
        NativeTensorOpsNN.sparseSoftmaxCrossEntropyWithLogits).apply(0))
      Basic.reshape(output, Basic.shape(labels, INT64))
    }
  }

  /** $OpDocNNSigmoidCrossEntropy
    *
    * @group NNOps
    * @param  logits  Tensor of shape `[D0, D1, ..., Dr-1, numClasses]`, containing unscaled log probabilities.
    * @param  labels  Tensor of shape `[D0, D1, ..., Dr-1, numClasses]]`, where each row must be a valid probability
    *                 distribution.
    * @param  weights Optionally, a coefficient to use for the positive examples.
    * @return Result as a new tensor, with rank one less than that of `logits` and the same data type as `logits`,
    *         containing the sigmoid cross entropy loss.
    */
  def sigmoidCrossEntropy[T: IsDecimal](
      logits: Tensor[T],
      labels: Tensor[T],
      weights: Tensor[T] = null
  ): Tensor[T] = {
    if (weights == null) {
      // The logistic loss formula from above is:
      //   x - x * z + log(1 + exp(-x))
      // For x < 0, a more numerically stable formula is:
      //   -x * z + log(1 + exp(x))
      // Note that these two expressions can be combined into the following single expression:
      //   max(x, 0) - x * z + log(1 + exp(-abs(x)))
      // To allow computing gradients at zero, we define custom versions of the max and the abs functions.
      val zeros = Tensor.zerosLike(logits)
      val condition = Math.greaterEqual(logits, zeros)
      val reluLogits = Math.select(condition, logits, zeros)
      val negativeAbsLogits = Math.select(condition, -logits, logits)
      Math.add(reluLogits - (logits * labels), Math.log1p(Math.exp(negativeAbsLogits)))
    } else {
      // The logistic loss formula from above is:
      //   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
      // For x < 0, a more numerically stable formula is:
      //   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
      // To avoid branching, we use the following single expression:
      //   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
      val one = 1.toTensor.cast(weights.dataType)
      val logWeights = ((weights - one) * labels) + one
      Math.addN(Seq[Tensor[T]](
        (one - labels) * logits,
        logWeights * Math.log1p(Math.exp(-Math.abs(logits))),
        relu(-logits)))
    }
  }

  /** $OpDocNNLogPoissonLoss
    *
    * @group NNOps
    * @param  logPredictions  Tensor containing the log-predictions.
    * @param  targets         Tensor with the same shape as `logPredictions`, containing the target values.
    * @param  computeFullLoss If `true`, Stirling's Approximation is used to approximate the full loss. Defaults to
    *                         `false`, meaning that the constant term is ignored.
    * @return Result as a new tensor.
    */
  def logPoissonLoss[T: IsDecimal](
      logPredictions: Tensor[T],
      targets: Tensor[T],
      computeFullLoss: Boolean = false
  ): Tensor[T] = {
    val output = Math.exp(logPredictions) - (logPredictions * targets)
    if (computeFullLoss) {
      // Need to create constant tensors here so that their data types can be matched to that of the targets.
      val pointFive = 0.5.toTensor.cast(targets.dataType)
      val twoPi = (2 * math.Pi).toTensor.cast(targets.dataType)
      val stirlingApproximation = (targets * Math.log(targets)) - targets + (pointFive * Math.log(twoPi * targets))
      val zeros = Tensor.zerosLike(targets)
      val ones = Tensor.onesLike(targets)
      val condition = Math.logicalAnd(Math.greaterEqual(targets, zeros), Math.lessEqual(targets, ones))
      output + Math.select(condition, zeros, stirlingApproximation)
    } else {
      output
    }
  }

  /** $OpDocNNSequenceLoss
    *
    * @group NNOps
    * @param  logits                 Tensor of shape `[batchSize, sequenceLength, numClasses]` containing unscaled log
    *                                probabilities.
    * @param  labels                 Tensor of shape `[batchSize, sequenceLength]` containing the true label at each
    *                                time step.
    * @param  weights                Optionally, a tensor of shape `[batchSize, sequenceLength]` containing weights to
    *                                use for each prediction. When using `weights` as masking, set all valid time steps
    *                                to 1 and all padded time steps to 0 (e.g., a mask returned by `tf.sequenceMask`).
    * @param  averageAcrossTimeSteps If `true`, the loss is summed across the sequence dimension and divided by the
    *                                total label weight across all time steps.
    * @param  averageAcrossBatch     If `true`, the loss is summed across the batch dimension and divided by the batch
    *                                size.
    * @param  lossFn                 Loss function to use that takes the predicted logits and the true labels as inputs
    *                                and returns the loss value. Defaults to `sparseSoftmaxCrossEntropy`.
    * @return Result as a new tensor.
    * @throws InvalidShapeException If any of `logits`, `labels`, or `weights` has invalid shape.
    */
  @throws[InvalidShapeException]
  def sequenceLoss[T: IsDecimal, I: IsInt32OrInt64](
      logits: Tensor[T],
      labels: Tensor[I],
      weights: Tensor[T] = null,
      averageAcrossTimeSteps: Boolean = true,
      averageAcrossBatch: Boolean = true,
      lossFn: (Tensor[T], Tensor[I]) => Tensor[T] = null
  ): Tensor[T] = {
    if (logits.rank != 3)
      throw InvalidShapeException(
        s"'logits' must have shape [batchSize, sequenceLength, numClasses], but had: ${logits.shape}.")
    if (labels.rank != 2)
      throw InvalidShapeException(s"'labels' must have shape [batchSize, sequenceLength], but had: ${labels.shape}.")
    if (weights != null && weights.rank != 2)
      throw InvalidShapeException(s"'weights' must have shape [batchSize, sequenceLength], but had: ${weights.shape}.")
    val numClasses = Basic.shape(logits, INT64).slice(2)
    val flattenedLogits = Basic.reshape(logits, Basic.stack(Seq[Tensor[Long]](-1L, numClasses)))
    val flattenedLabels = Basic.reshape(labels, Shape(-1))
    val epsilon = 1e-12.toTensor.cast(logits.dataType)
    var loss = {
      if (lossFn != null)
        lossFn(flattenedLogits, flattenedLabels)
      else
        sparseSoftmaxCrossEntropy(flattenedLogits, flattenedLabels)
    }
    if (weights != null)
      loss = loss * Basic.reshape(weights, Shape(-1))
    if (averageAcrossTimeSteps && averageAcrossBatch) {
      loss = Math.sum(loss)
      val totalSize = {
        if (weights != null)
          Math.sum(weights) + epsilon
        else
          Basic.size(flattenedLabels, INT32).cast(logits.dataType)
      }
      loss = Math.divide(loss, totalSize)
    } else {
      loss = Basic.reshape(loss, Basic.shape(logits, INT64).slice(0 :: 2))
    }
    if (averageAcrossTimeSteps && !averageAcrossBatch) {
      loss = Math.sum(loss, axes = 1)
      val totalSize = {
        if (weights != null)
          Math.sum(weights, axes = 1) + epsilon
        else
          Basic.shape(labels, INT64).slice(1).cast(logits.dataType)
      }
      loss = Math.divide(loss, totalSize)
    }
    if (!averageAcrossTimeSteps && averageAcrossBatch) {
      loss = Math.sum(loss, axes = 0)
      val totalSize = {
        if (weights != null)
          Math.sum(weights, axes = 0) + epsilon
        else
          Basic.shape(labels, INT64).slice(0).cast(logits.dataType)
      }
      loss = Math.divide(loss, totalSize)
    }
    loss
  }

  //endregion Loss Ops

  //region Normalization Ops

  /** $OpDocNNLocalResponseNormalization
    *
    * @group NNOps
    * @param  input       Input tensor.
    * @param  depthRadius Half-width of the 1-D normalization window.
    * @param  bias        Offset (usually positive to avoid dividing by 0).
    * @param  alpha       Scale factor (usually positive).
    * @param  beta        Exponent.
    * @return Created op output.
    */
  def lrn[T: IsBFloat16OrFloat16OrFloat32](
      input: Tensor[T],
      depthRadius: Int = 5,
      bias: Float = 1.0f,
      alpha: Float = 1.0f,
      beta: Float = 0.5f
  ): Tensor[T] = {
    localResponseNormalization(input, depthRadius, bias, alpha, beta)
  }

  /** $OpDocNNLocalResponseNormalization
    *
    * @group NNOps
    * @param  input       Input tensor.
    * @param  depthRadius Half-width of the 1-D normalization window.
    * @param  bias        Offset (usually positive to avoid dividing by 0).
    * @param  alpha       Scale factor (usually positive).
    * @param  beta        Exponent.
    * @return Created op output.
    */
  def localResponseNormalization[T: IsBFloat16OrFloat16OrFloat32](
      input: Tensor[T],
      depthRadius: Int = 5,
      bias: Float = 1.0f,
      alpha: Float = 1.0f,
      beta: Float = 0.5f
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.lRN(
      executionContext.value.nativeHandle, input.nativeHandle, depthRadius, bias, alpha, beta))
  }

  //endregion Normalization Ops

  /** $OpDocNNDropout
    *
    * @group NNOps
    * @param  input           Input tensor.
    * @param  keepProbability Probability (i.e., number in the interval `(0, 1]`) that each element is kept.
    * @param  scaleOutput     If `true`, the outputs will be divided by the keep probability.
    * @param  noiseShape      Rank-1 tensor representing the shape for the randomly generated keep/drop flags.
    * @param  seed            Optional random seed, used to generate a random seed pair for the random number
    *                         generator, when combined with the graph-level seed.
    * @return Result as a new tensor that has the same shape as `input`.
    */
  def dropout[T: IsFloat16OrFloat32OrFloat64](
      input: Tensor[T],
      keepProbability: Float,
      scaleOutput: Boolean = true,
      noiseShape: Tensor[Int] = null,
      seed: Option[Int] = None
  ): Tensor[T] = {
    require(keepProbability > 0.0 && keepProbability <= 1.0, s"'keepProbability' ($keepProbability) must be in (0, 1].")
    // Do nothing if we know that keepProbability == 1.
    if (keepProbability == 1.0) {
      input
    } else {
      val inferredNoiseShape = if (noiseShape == null) Basic.shape(input, INT32) else noiseShape
      // Uniform random variable in [keepProbability, 1.0 + keepProbability).
      val probability = keepProbability.toTensor.cast(input.dataType)
      val random = Random.randomUniform(
        dataType = input.dataType,
        shape = inferredNoiseShape)(
        minValue = probability,
        maxValue = probability + Cast.cast(Tensor(1), probability.dataType),
        seed = seed)
      // 0.0 if in [keepProbability, 1.0) and 1.0 if [1.0, 1.0 + keepProbability).
      val binaryTensor = Math.floor(random)
      if (scaleOutput) Math.divide(input, probability) * binaryTensor else input * binaryTensor
    }
  }

  /** $OpDocNNTopK
    *
    * @group NNOps
    * @param  input  Input tensor whose last axis has size at least `k`.
    * @param  k      Scalar tensor containing the number of top elements to look for along the last axis of `input`.
    * @param  sorted If `true`, the resulting `k` elements will be sorted by their values in descending order.
    * @return Tuple containing the created tensors: (i) `values`: the `k` largest elements along each last
    *         dimensional slice, and (ii) `indices`: the indices of `values` within the last axis of `input`.
    */
  def topK[T: IsNotQuantized](
      input: Tensor[T],
      k: Tensor[Int] = 1,
      sorted: Boolean = true
  ): (Tensor[T], Tensor[Int]) = {
    val outputs = NativeTensorOpsNN.topKV2(
      executionContext.value.nativeHandle, input.nativeHandle, k.nativeHandle, sorted)
    (Tensor.fromNativeHandle[T](outputs(0)), Tensor.fromNativeHandle[Int](outputs(1)))
  }

  /** $OpDocNNInTopK
    *
    * @group NNOps
    * @param  predictions Tensor containing the predictions.
    * @param  targets     Tensor containing the targets.
    * @param  k           Scalar tensor containing the number of top elements to look at.
    * @return Result as a new tensor.
    */
  def inTopK[I: IsInt32OrInt64](predictions: Tensor[Float], targets: Tensor[I], k: Tensor[I]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsNN.inTopKV2(
      executionContext.value.nativeHandle, predictions.nativeHandle, targets.nativeHandle, k.nativeHandle))
  }

  //region Convolution Ops

  /** $OpDocNNConv2D
    *
    * @group NNOps
    * @param  input         4-D tensor whose dimension order is interpreted according to the value of `dataFormat`.
    * @param  filter        4-D tensor with shape `[filterHeight, filterWidth, inChannels, outChannels]`.
    * @param  stride1       Stride of the sliding window along the second dimension of `input`.
    * @param  stride2       Stride of the sliding window along the third dimension of `input`.
    * @param  padding       Padding mode to use.
    * @param  dataFormat    Format of the input and output data.
    * @param  dilations     The dilation factor for each dimension of input. If set to `k > 1`, there will be `k - 1`
    *                       skipped cells between each filter element on that dimension. The dimension order is
    *                       determined by the value of `dataFormat`. Dilations in the batch and depth dimensions must
    *                       be set to `1`.
    * @param  useCuDNNOnGPU Boolean value indicating whether or not to use CuDNN for the created op, if its placed on a
    *                       GPU, as opposed to the TensorFlow implementation.
    * @return Result as a new 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def conv2D[T: IsDecimal](
      input: Tensor[T],
      filter: Tensor[T],
      stride1: Long,
      stride2: Long,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      // TODO: [OPS/NN] Enforce the batch and depth dilation constraint at compile time.
      dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
      useCuDNNOnGPU: Boolean = true
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.conv2D(
      executionContext.value.nativeHandle, input.nativeHandle, filter.nativeHandle,
      Array[Long](1, stride1, stride2, 1), padding.name.getBytes(StandardCharsets.ISO_8859_1), useCuDNNOnGPU,
      dataFormat.name.getBytes(StandardCharsets.ISO_8859_1),
      Array(dilations._1, dilations._2, dilations._3, dilations._4)))
  }

  /** $OpDocNNConv2DBackpropInput
    *
    * @group NNOps
    * @param  inputSizes     Integer vector representing the shape of the original input, which is a 4-D tensor.
    * @param  filter         4-D tensor with shape `[filterHeight, filterWidth, inChannels, outChannels]`.
    * @param  outputGradient 4-D tensor containing the gradients w.r.t. the output of the convolution and whose shape
    *                        depends on the value of `dataFormat`.
    * @param  stride1        Stride of the sliding window along the second dimension of `input`.
    * @param  stride2        Stride of the sliding window along the third dimension of `input`.
    * @param  padding        Padding mode to use.
    * @param  dataFormat     Format of the input and output data.
    * @param  dilations      The dilation factor for each dimension of input. If set to `k > 1`, there will be `k - 1`
    *                        skipped cells between each filter element on that dimension. The dimension order is
    *                        determined by the value of `dataFormat`. Dilations in the batch and depth dimensions must
    *                        be set to `1`.
    * @param  useCuDNNOnGPU  Boolean value indicating whether or not to use CuDNN for the created op, if its placed on a
    *                        GPU, as opposed to the TensorFlow implementation.
    * @return Result as a new 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def conv2DBackpropInput[T: IsDecimal](
      inputSizes: Tensor[Int],
      filter: Tensor[T],
      outputGradient: Tensor[T],
      stride1: Long,
      stride2: Long,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      // TODO: [OPS/NN] Enforce the batch and depth dilation constraint at compile time.
      dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
      useCuDNNOnGPU: Boolean = true
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.conv2DBackpropInput(
      executionContext.value.nativeHandle, inputSizes.nativeHandle, filter.nativeHandle, outputGradient.nativeHandle,
      Array[Long](1, stride1, stride2, 1), padding.name.getBytes(StandardCharsets.ISO_8859_1), useCuDNNOnGPU,
      dataFormat.name.getBytes(StandardCharsets.ISO_8859_1),
      Array(dilations._1, dilations._2, dilations._3, dilations._4)))
  }

  /** $OpDocNNConv2DBackpropFilter
    *
    * @group NNOps
    * @param  input          4-D tensor whose dimension order is interpreted according to the value of `dataFormat`.
    * @param  filterSizes    Integer vector representing the shape of the original filter, which is a 4-D tensor.
    * @param  outputGradient 4-D tensor containing the gradients w.r.t. the output of the convolution and whose shape
    *                        depends on the value of `dataFormat`.
    * @param  stride1        Stride of the sliding window along the second dimension of `input`.
    * @param  stride2        Stride of the sliding window along the third dimension of `input`.
    * @param  padding        Padding mode to use.
    * @param  dataFormat     Format of the input and output data.
    * @param  dilations      The dilation factor for each dimension of input. If set to `k > 1`, there will be `k - 1`
    *                        skipped cells between each filter element on that dimension. The dimension order is
    *                        determined by the value of `dataFormat`. Dilations in the batch and depth dimensions must
    *                        be set to `1`.
    * @param  useCuDNNOnGPU  Boolean value indicating whether or not to use CuDNN for the created op, if its placed on a
    *                        GPU, as opposed to the TensorFlow implementation.
    * @return Result as a new 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def conv2DBackpropFilter[T: IsDecimal](
      input: Tensor[T],
      filterSizes: Tensor[Int],
      outputGradient: Tensor[T],
      stride1: Long,
      stride2: Long,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      // TODO: [OPS/NN] Enforce the batch and depth dilation constraint at compile time.
      dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
      useCuDNNOnGPU: Boolean = true
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.conv2DBackpropFilter(
      executionContext.value.nativeHandle, input.nativeHandle, filterSizes.nativeHandle, outputGradient.nativeHandle,
      Array[Long](1, stride1, stride2, 1), padding.name.getBytes(StandardCharsets.ISO_8859_1), useCuDNNOnGPU,
      dataFormat.name.getBytes(StandardCharsets.ISO_8859_1),
      Array(dilations._1, dilations._2, dilations._3, dilations._4)))
  }

  //endregion Convolution Ops

  //region Pooling Ops

  /** $OpDocNNMaxPool
    *
    * @group NNOps
    * @param  input      4-D tensor whose dimension order is interpreted according to the value of `dataFormat`.
    * @param  windowSize The size of the pooling window for each dimension of the input tensor.
    * @param  stride1    Stride of the sliding window along the second dimension of `input`.
    * @param  stride2    Stride of the sliding window along the third dimension of `input`.
    * @param  padding    Padding mode to use.
    * @param  dataFormat Format of the input and output data.
    * @return Result as a new 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def maxPool[T: IsNotQuantized](
      input: Tensor[T],
      windowSize: Seq[Int],
      stride1: Int,
      stride2: Int,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.maxPool(
      executionContext.value.nativeHandle, input.nativeHandle, windowSize.map(_.toLong).toArray,
      Array[Long](1, stride1, stride2, 1), padding.name.getBytes(StandardCharsets.ISO_8859_1),
      dataFormat.name.getBytes(StandardCharsets.ISO_8859_1)))
  }

  /** $OpDocNNMaxPoolGrad
    *
    * @group NNOps
    * @param  originalInput  Original input tensor.
    * @param  originalOutput Original output tensor.
    * @param  outputGradient 4-D tensor containing the gradients w.r.t. the output of the max pooling and whose shape
    *                        depends on the value of `dataFormat`.
    * @param  windowSize     The size of the pooling window for each dimension of the input tensor.
    * @param  stride1        Stride of the sliding window along the second dimension of `input`.
    * @param  stride2        Stride of the sliding window along the third dimension of `input`.
    * @param  padding        Padding mode to use.
    * @param  dataFormat     Format of the input and output data.
    * @return Result as a new 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def maxPoolGrad[T: IsReal](
      originalInput: Tensor[T],
      originalOutput: Tensor[T],
      outputGradient: Tensor[T],
      windowSize: Seq[Int],
      stride1: Int,
      stride2: Int,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.maxPoolGrad(
      executionContext.value.nativeHandle, originalInput.nativeHandle, originalOutput.nativeHandle,
      outputGradient.nativeHandle, windowSize.map(_.toLong).toArray, Array[Long](1, stride1, stride2, 1),
      padding.name.getBytes(StandardCharsets.ISO_8859_1), dataFormat.name.getBytes(StandardCharsets.ISO_8859_1)))
  }

  /** $OpDocNNMaxPoolGradGrad
    *
    * @group NNOps
    * @param  originalInput  Original input tensor.
    * @param  originalOutput Original output tensor.
    * @param  outputGradient 4-D tensor containing the gradients w.r.t. the output of the max pooling and whose shape
    *                        depends on the value of `dataFormat`.
    * @param  windowSize     The size of the pooling window for each dimension of the input tensor.
    * @param  stride1        Stride of the sliding window along the second dimension of `input`.
    * @param  stride2        Stride of the sliding window along the third dimension of `input`.
    * @param  padding        Padding mode to use.
    * @param  dataFormat     Format of the input and output data.
    * @return Result as a new 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def maxPoolGradGrad[T: IsReal](
      originalInput: Tensor[T],
      originalOutput: Tensor[T],
      outputGradient: Tensor[T],
      windowSize: Seq[Int],
      stride1: Int,
      stride2: Int,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsNN.maxPoolGradGrad(
      executionContext.value.nativeHandle, originalInput.nativeHandle, originalOutput.nativeHandle,
      outputGradient.nativeHandle, windowSize.map(_.toLong).toArray, Array[Long](1, stride1, stride2, 1),
      padding.name.getBytes(StandardCharsets.ISO_8859_1), dataFormat.name.getBytes(StandardCharsets.ISO_8859_1)))
  }

  //endregion Pooling Ops
}

object NN extends NN {
  private[tensors] trait Implicits {
    implicit class MathNNOps[T: IsNotQuantized](val tensor: Tensor[T]) {
      //region Core Ops

      /** $OpDocNNAddBias
        *
        * @group NNOps
        * @param  bias          Bias tensor that must be one-dimensional (i.e., it must have rank 1).
        * @param  cNNDataFormat Data format of the input and output tensors. With the default format [[NWCFormat]], the
        *                       `bias` tensor will be added to the last dimension of the `value` tensor. Alternatively, the
        *                       format could be [[NCWFormat]], and the `bias` tensor would be added to the third-to-last
        *                       dimension.
        * @return Result as a new tensor.
        */
      def addBias(bias: Tensor[T], cNNDataFormat: CNNDataFormat = CNNDataFormat.default): Tensor[T] = {
        NN.addBias(tensor, bias, cNNDataFormat)
      }

      /** $OpDocNNLinear
        *
        * @group NNOps
        * @param  weights Weights tensor.
        * @param  bias    Bias tensor.
        * @return Result as a new tensor.
        */
      def linear(weights: Tensor[T], bias: Tensor[T] = null): Tensor[T] = NN.linear(tensor, weights, bias)

      //endregion CoreOps

      /** $OpDocNNTopK
        *
        * @group NNOps
        * @param  k      Scalar tensor containing the number of top elements to look for along the last axis of `input`.
        * @param  sorted If `true`, the resulting `k` elements will be sorted by their values in descending order.
        * @return Tuple containing the created op outputs: (i) `values`: the `k` largest elements along each last
        *         dimensional slice, and (ii) `indices`: the indices of `values` within the last axis of `input`.
        */
      def topK(k: Tensor[Int] = 1, sorted: Boolean = true): (Tensor[T], Tensor[Int]) = NN.topK(tensor, k, sorted)

      //region Pooling Ops

      /** $OpDocNNMaxPool
        *
        * @param  windowSize The size of the pooling window for each dimension of the input tensor.
        * @param  stride1    Stride of the sliding window along the second dimension of `input`.
        * @param  stride2    Stride of the sliding window along the third dimension of `input`.
        * @param  padding    Padding mode to use.
        * @param  dataFormat Format of the input and output data.
        * @return Result as a new 4-D tensor whose dimension order depends on the value of `dataFormat`.
        */
      def maxPool(
          windowSize: Seq[Int],
          stride1: Int,
          stride2: Int,
          padding: ConvPaddingMode,
          dataFormat: CNNDataFormat = CNNDataFormat.default
      ): Tensor[T] = {
        NN.maxPool(tensor, windowSize, stride1, stride2, padding, dataFormat)
      }

      //endregion Pooling Ops
    }

    implicit class RealNNOps[T: IsReal](val tensor: Tensor[T]) {
      //region Activation Ops

      /** $OpDocNNRelu
        *
        * @group NNOps
        * @param  alpha Slope of the negative section, also known as leakage parameter. If other than `0.0f`, the negative
        *               part will be equal to `alpha * x` instead of `0`. Defaults to `0`.
        * @return Result as a new tensor.
        */
      def relu(alpha: Float = 0.0f): Tensor[T] = NN.relu(tensor, alpha)

      /** $OpDocNNRelu6
        *
        * @group NNOps
        * @return Result as a new tensor.
        */
      def relu6: Tensor[T] = NN.relu6(tensor)

      /** $OpDocNNCrelu
        *
        * @group NNOps
        * @return Result as a new tensor.
        */
      def crelu: Tensor[T] = NN.crelu(tensor)

      /** $OpDocNNSoftplus
        *
        * @group NNOps
        * @return Result as a new tensor.
        */
      def softplus: Tensor[T] = NN.softplus(tensor)

      /** $OpDocNNSoftsign
        *
        * @group NNOps
        * @return Result as a new tensor.
        */
      def softsign: Tensor[T] = NN.softsign(tensor)

      //endregion Activation Ops
    }

    implicit class DecimalNNOps[T: IsDecimal](val tensor: Tensor[T]) {
      //region Activation Ops

      /** $OpDocNNElu
        *
        * @group NNOps
        * @return Result as a new tensor.
        */
      def elu: Tensor[T] = NN.elu(tensor)

      /** $OpDocNNSelu
        *
        * @group NNOps
        * @return Result as a new tensor.
        */
      def selu: Tensor[T] = NN.selu(tensor)

      //endregion Activation Ops

      /** $OpDocNNSoftmax
        *
        * @group NNOps
        * @param  axis Axis along which to perform the softmax. Defaults to `-1` denoting the last axis.
        * @return Result as a new tensor.
        */
      def softmax(axis: Int = -1): Tensor[T] = NN.softmax(tensor, axis)

      /** $OpDocNNLogSoftmax
        *
        * @group NNOps
        * @param  axis Axis along which to perform the log-softmax. Defaults to `-1` denoting the last axis.
        * @return Result as a new tensor.
        */
      def logSoftmax(axis: Int = -1): Tensor[T] = NN.logSoftmax(tensor, axis)

      //region Convolution Ops

      /** $OpDocNNConv2D
        *
        * @group NNOps
        * @param  filter        4-D tensor with shape `[filterHeight, filterWidth, inChannels, outChannels]`.
        * @param  stride1       Stride of the sliding window along the second dimension of this tensor.
        * @param  stride2       Stride of the sliding window along the third dimension of this tensor.
        * @param  padding       Padding mode to use.
        * @param  dataFormat    Format of the input and output data.
        * @param  dilations     The dilation factor for each dimension of input. If set to `k > 1`, there will be `k - 1`
        *                       skipped cells between each filter element on that dimension. The dimension order is
        *                       determined by the value of `dataFormat`. Dilations in the batch and depth dimensions must
        *                       be set to `1`.
        * @param  useCuDNNOnGPU Boolean value indicating whether or not to use CuDNN for the created op, if its placed on a
        *                       GPU, as opposed to the TensorFlow implementation.
        * @return Result as a new 4-D tensor whose dimension order depends on the value of `dataFormat`.
        */
      def conv2D(
          filter: Tensor[T],
          stride1: Long,
          stride2: Long,
          padding: ConvPaddingMode,
          dataFormat: CNNDataFormat = CNNDataFormat.default,
          // TODO: [OPS/NN] Enforce the batch and depth dilation constraint at compile time.
          dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
          useCuDNNOnGPU: Boolean = true
      ): Tensor[T] = {
        NN.conv2D(tensor, filter, stride1, stride2, padding, dataFormat, dilations, useCuDNNOnGPU)
      }

      //endregion Convolution Ops
    }

    implicit class BFloat16OrFloat16OrFloat32NNOps[T: IsBFloat16OrFloat16OrFloat32](val tensor: Tensor[T]) {
      //region Normalization Ops

      /** $OpDocNNLocalResponseNormalization
        *
        * @group NNOps
        * @param  depthRadius Half-width of the 1-D normalization window.
        * @param  bias        Offset (usually positive to avoid dividing by 0).
        * @param  alpha       Scale factor (usually positive).
        * @param  beta        Exponent.
        * @return Created op output.
        */
      def lrn(
          depthRadius: Int = 5,
          bias: Float = 1.0f,
          alpha: Float = 1.0f,
          beta: Float = 0.5f,
          name: String = "LRN"
      ): Tensor[T] = {
        NN.localResponseNormalization(tensor, depthRadius, bias, alpha, beta)
      }

      /** $OpDocNNLocalResponseNormalization
        *
        * @group NNOps
        * @param  depthRadius Half-width of the 1-D normalization window.
        * @param  bias        Offset (usually positive to avoid dividing by 0).
        * @param  alpha       Scale factor (usually positive).
        * @param  beta        Exponent.
        * @return Created op output.
        */
      def localResponseNormalization(
          depthRadius: Int = 5,
          bias: Float = 1.0f,
          alpha: Float = 1.0f,
          beta: Float = 0.5f
      ): Tensor[T] = {
        NN.localResponseNormalization(tensor, depthRadius, bias, alpha, beta)
      }

      //endregion Normalization Ops
    }

    implicit class Float16OrFloat32OrFloat64NNOps[T: IsFloat16OrFloat32OrFloat64](val tensor: Tensor[T]) {
      /** $OpDocNNDropout
        *
        * @group NNOps
        * @param  keepProbability Probability (i.e., number in the interval `(0, 1]`) that each element is kept.
        * @param  scaleOutput     If `true`, the outputs will be divided by the keep probability.
        * @param  noiseShape      Rank-1 tensor representing the shape for the randomly generated keep/drop flags.
        * @param  seed            Optional random seed, used to generate a random seed pair for the random number
        *                         generator, when combined with the graph-level seed.
        * @return Result as a new tensor that has the same shape as `input`.
        */
      def dropout(
          keepProbability: Float,
          scaleOutput: Boolean = true,
          noiseShape: Tensor[Int] = null,
          seed: Option[Int] = None
      ): Tensor[T] = {
        NN.dropout(tensor, keepProbability, scaleOutput, noiseShape, seed)
      }
    }

    implicit class Float32OrFloat64NNOps[T: IsFloat32OrFloat64](val tensor: Tensor[T]) {
      //region Core Ops

      /** $OpDocNNL2Normalize
        *
        * @group NNOps
        * @param  axes    Tensor containing the axes along which to normalize.
        * @param  epsilon Lower bound value for the norm. The created op will use `sqrt(epsilon)` as the divisor, if
        *                 `norm < sqrt(epsilon)`.
        * @return Result as a new tensor.
        */
      def l2Normalize(axes: Tensor[Int], epsilon: Float = 1e-12f): Tensor[T] = NN.l2Normalize(tensor, axes, epsilon)

      //endregion Core Ops
    }

    implicit class Float32NNOps(val tensor: Tensor[Float]) {
      /** $OpDocNNInTopK
        *
        * @group NNOps
        * @param  targets Tensor containing the targets.
        * @param  k       Scalar tensor containing the number of top elements to look at.
        * @return Result as a new tensor.
        */
      def inTopK[I: IsInt32OrInt64](targets: Tensor[I], k: Tensor[I]): Tensor[Boolean] = NN.inTopK(tensor, targets, k)
    }

    implicit def tensorConvertibleToMathNNOps[TC, T: IsNotQuantized](value: TC)(implicit f: TC => Tensor[T]): MathNNOps[T] = new MathNNOps(f(value))
    implicit def tensorConvertibleToRealNNOps[TC, T: IsReal](value: TC)(implicit f: TC => Tensor[T]): RealNNOps[T] = new RealNNOps(f(value))
    implicit def tensorConvertibleToDecimalNNOps[TC, T: IsDecimal](value: TC)(implicit f: TC => Tensor[T]): DecimalNNOps[T] = new DecimalNNOps(f(value))
    implicit def tensorConvertibleToBFloat16OrFloat16OrFloat32NNOps[TC, T: IsBFloat16OrFloat16OrFloat32](value: TC)(implicit f: TC => Tensor[T]): BFloat16OrFloat16OrFloat32NNOps[T] = new BFloat16OrFloat16OrFloat32NNOps(f(value))
    implicit def tensorConvertibleToFloat16OrFloat32OrFloat64NNOps[TC, T: IsFloat16OrFloat32OrFloat64](value: TC)(implicit f: TC => Tensor[T]): Float16OrFloat32OrFloat64NNOps[T] = new Float16OrFloat32OrFloat64NNOps(f(value))
    implicit def tensorConvertibleToFloat32OrFloat64NNOps[TC, T: IsFloat32OrFloat64](value: TC)(implicit f: TC => Tensor[T]): Float32OrFloat64NNOps[T] = new Float32OrFloat64NNOps(f(value))
    implicit def tensorConvertibleToFloat32NNOps[TC](value: TC)(implicit f: TC => Tensor[Float]): Float32NNOps = new Float32NNOps(f(value))
  }

  /** Creates an op that flattens the outer axes of `input` and keeps its last axis. */
  private[ops] def flattenOuterAxes[T](input: Tensor[T]): Tensor[T] = {
    val rank = Basic.rank(input)
    val lastAxisSize = Basic.slice(
      Basic.shape(input, INT32),
      Basic.expandDims(Math.subtract(rank, 1), -1),
      Tensor.fill(rank.dataType, Shape(1))(1))
    Basic.reshape(input, Basic.concatenate(Seq(Tensor.fill(rank.dataType, Shape(1))(-1), lastAxisSize), 0))
  }

  /** Creates an op that swaps the axes `axis1` and `axis2` in `input` and ignores all axes after `axis2`. */
  private[ops] def swapAxes[T](
      input: Tensor[T],
      axis1: Tensor[Int],
      axis2: Tensor[Int]
  ): Tensor[T] = {
    Basic.transpose(
      input,
      Basic.concatenate(Seq(
        Math.range(0, axis1),
        axis2,
        Math.range(axis1 + 1, axis2),
        axis1), 0))
  }

  /** Creates an op that moves `axis` to the end. */
  private[ops] def moveAxisToEnd[T](
      input: Tensor[T],
      axis: Int,
      rank: Tensor[Int]
  ): Tensor[T] = {
    if (axis == -1) {
      input
    } else {
      val axisOutput = Tensor.ofType(rank.dataType, axis)
      Basic.transpose(
        input,
        Basic.concatenate(Seq(
          Math.range(0, axisOutput),
          Math.range(axisOutput + 1, rank),
          axisOutput), 0))
    }
  }
}
