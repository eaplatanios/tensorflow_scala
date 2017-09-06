/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.NN.{CNNDataFormat, NCHWFormat, NHWCFormat}
import org.platanios.tensorflow.api.tensors.{Context, Tensor, TensorOps}
import org.platanios.tensorflow.api.types.{DataType, FLOAT16, FLOAT32, FLOAT64, INT32, INT64}
import org.platanios.tensorflow.jni.generated.tensors.{NN => NativeTensorOpsNN}

import scala.util.DynamicVariable

/** Contains functions for executing ops related to neural networks.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait NN {
  //region Core Ops

  /** $OpDocNNAddBias
    *
    * @group NNOps
    * @param  value         Value tensor.
    * @param  bias          Bias tensor that must be one-dimensional (i.e., it must have rank 1).
    * @param  cNNDataFormat Data format of the input and output tensors. With the default format [[NHWCFormat]], the
    *                       `bias` tensor will be added to the last dimension of the `value` tensor. Alternatively, the
    *                       format could be [[NCHWFormat]], and the `bias` tensor would be added to the third-to-last
    *                       dimension.
    * @return Result as a new tensor.
    */
  def addBias(
      value: Tensor, bias: Tensor, cNNDataFormat: CNNDataFormat = CNNDataFormat.default)(
      implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsNN.biasAdd(
      context.value.nativeHandle, value.nativeHandle, bias.nativeHandle, cNNDataFormat.toString.getBytes()))
  }

  /** $OpDocNNLinear
    *
    * @group NNOps
    * @param  x       Input tensor.
    * @param  weights Weights tensor.
    * @param  bias    Bias tensor.
    * @return Result as a new tensor.
    */
  def linear(x: Tensor, weights: Tensor, bias: Tensor): Tensor = addBias(Math.matmul(x, weights), bias)

  /** $OpDocNNL2Normalize
    *
    * @group NNOps
    * @param  x       Input tensor.
    * @param  axes    Tensor containing the axes along which to normalize.
    * @param  epsilon Lower bound value for the norm. The created op will use `sqrt(epsilon)` as the divisor, if
    *                 `norm < sqrt(epsilon)`.
    * @return Result as a new tensor.
    */
  def l2Normalize(
      x: Tensor, axes: Tensor, epsilon: Float = 1e-12f): Tensor = {
    val dataType = DataType.mostPrecise(x.dataType, FLOAT32)
    val preciseX = Math.cast(x, dataType)
    val squareSum = Math.sum(Math.square(preciseX), axes = axes, keepDims = true)
    val xInverseNorm = Math.rsqrt(Math.maximum(squareSum, Tensor(dataType, epsilon)))
    Math.cast(Math.multiply(preciseX, xInverseNorm), x.dataType)
  }

  //endregion Core Ops

  //region Activation Ops

  /** $OpDocNNRelu
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  alpha Slope of the negative section, also known as leakage parameter. If other than `0.0f`, the negative
    *               part will be equal to `alpha * x` instead of `0`. Defaults to `0`.
    * @return Result as a new tensor.
    */
  def relu(input: Tensor, alpha: Float = 0.0f)(implicit context: DynamicVariable[Context]): Tensor = {
    def reluOp[T: TensorOps](i: T): T = {
      implicitly[TensorOps[T]]
          .applyUnary(i, t =>
            Tensor.fromNativeHandle(NativeTensorOpsNN.relu(context.value.nativeHandle, t.nativeHandle)))
    }
    if (alpha == 0.0) {
      reluOp(input)
    } else {
      val positive = reluOp(input)
      val negative = reluOp(-input)
      positive - (Tensor(negative.dataType, alpha) * negative)
    }
  }

  /** $OpDocNNRelu6
    *
    * @group NNOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def relu6(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsNN.relu6(context.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocNNCrelu
    *
    * @group NNOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def crelu(input: Tensor): Tensor = relu(Basic.concatenate(Seq(input, -input), axis = -1))

  /** $OpDocNNElu
    *
    * @group NNOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def elu(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsNN.elu(context.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocNNSelu
    *
    * @group NNOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def selu(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsNN.selu(context.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocNNSoftplus
    *
    * @group NNOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def softplus(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsNN.softplus(context.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocNNSoftsign
    *
    * @group NNOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def softsign(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsNN.softsign(context.value.nativeHandle, input.nativeHandle))
  }

  //endregion Activation Ops

  /** Helper function for [[softmax]] and [[logSoftmax]] that reshapes and transposes the input logits into
    * two-dimensional tensors and then creates the corresponding native op. The output is transposed and reshaped
    * back. */
  private[this] def softmaxHelper(logits: Tensor, opFunction: (Long) => Long, axis: Int = -1): Tensor = {
    // We need the original shape of the logits for shape inference.
    val shape = logits.shape
    val isLastAxis = axis == -1 || axis == shape.rank - 1
    if (shape.rank == 2 && isLastAxis) {
      Tensor.fromNativeHandle(opFunction(logits.nativeHandle))
    } else if (isLastAxis) {
      // If axis is the last axis, we simply reshape the logits to a matrix and apply the internal softmax.
      val inputShape = Basic.shape(logits)
      val flattenedLogits = NN.flattenOuterAxes(logits)
      val output = Tensor.fromNativeHandle(opFunction(flattenedLogits.nativeHandle))
      Basic.reshape(output, inputShape)
    } else {
      // If axis is not the last dimension, we have to do a reshape and transpose so that we can still perform softmax
      // on its last dimension.
      // We swap the logits' axis of axis and its last axis.
      val inputRank = Basic.rank(logits)
      val swappedLogits = NN.swapAxes(logits, axis, Math.subtract(inputRank, 1))
      val shapeAfterSwap = Basic.shape(swappedLogits)
      // We reshape the logits into a matrix.
      val flattenedLogits = NN.flattenOuterAxes(swappedLogits)
      // We perform the actual softmax on the last axis.
      var output = Tensor.fromNativeHandle(opFunction(flattenedLogits.nativeHandle))
      // We transform back the output tensor.
      output = Basic.reshape(output, shapeAfterSwap)
      output = NN.swapAxes(output, axis, Math.subtract(inputRank, 1))
      output
    }
  }

  /** $OpDocNNSoftmax
    *
    * @group NNOps
    * @param  logits Tensor containing the logits with data type [[FLOAT16]], [[FLOAT32]], or [[FLOAT64]].
    * @param  axis   Axis along which to perform the softmax. Defaults to `-1` denoting the last axis.
    * @return Result as a new tensor.
    */
  def softmax(logits: Tensor, axis: Int = -1)(implicit context: DynamicVariable[Context]): Tensor = {
    softmaxHelper(logits, NativeTensorOpsNN.softmax(context.value.nativeHandle, _), axis)
  }

  /** $OpDocNNLogSoftmax
    *
    * @group NNOps
    * @param  logits Tensor containing the logits with data type [[FLOAT16]], [[FLOAT32]], or [[FLOAT64]].
    * @param  axis   Axis along which to perform the log-softmax. Defaults to `-1` denoting the last axis.
    * @return Result as a new tensor.
    */
  def logSoftmax(logits: Tensor, axis: Int = -1)(implicit context: DynamicVariable[Context]): Tensor = {
    softmaxHelper(logits, NativeTensorOpsNN.logSoftmax(context.value.nativeHandle, _), axis)
  }

  //region Loss Ops

  /** $OpDocNNL2Loss
    *
    * @group NNOps
    * @param  input [[FLOAT16]], [[FLOAT32]], or [[FLOAT64]] input tensor.
    * @return Result as a new tensor.
    */
  def l2Loss(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsNN.l2Loss(context.value.nativeHandle, input.nativeHandle))
  }

  private[this] def nativeCrossEntropyProxy(
      features: Long, labels: Long, function: (Long, Long, Long) => Array[Long])(
      implicit context: DynamicVariable[Context]): Array[Long] = {
    function(context.value.nativeHandle, features, labels)
  }

  /** $OpDocNNSoftmaxCrossEntropy
    *
    * @group NNOps
    * @param  logits Tensor of shape `[D0, D1, ..., Dr-1, numClasses]` and data type [[FLOAT16]], [[FLOAT32]], or
    *                [[FLOAT64]], containing unscaled log probabilities.
    * @param  labels Tensor of shape `[D0, D1, ..., Dr-1, numClasses]` and data type [[FLOAT16]], [[FLOAT32]], or
    *                [[FLOAT64]], where each row must be a valid probability distribution.
    * @param  axis   The class axis, along which the softmax is computed. Defaults to `-1`, which is the last axis.
    * @return Result as a new tensor, with rank one less than that of `logits` and the same data type as `logits`,
    *         containing the softmax cross entropy loss.
    */
  def softmaxCrossEntropy(logits: Tensor, labels: Tensor, axis: Int = -1): Tensor = {
    // Labels and logits must be of the same data type.
    val preciseLogits = if (logits.dataType == FLOAT16) Math.cast(logits, FLOAT32) else logits
    val preciseLabels = Math.cast(labels, preciseLogits.dataType)
    val inputRank = Basic.rank(preciseLogits)
    // We move axis to the end, if it's not already the last axis.
    val transposedLogits = NN.moveAxisToEnd(preciseLogits, axis, inputRank)
    val transposedLabels = NN.moveAxisToEnd(preciseLabels, axis, inputRank)
    val inputShape = Basic.shape(preciseLogits)
    // Flatten transposedLogits and transposedLabels into matrices.
    val flattenedLogits = NN.flattenOuterAxes(transposedLogits)
    val flattenedLabels = NN.flattenOuterAxes(transposedLabels)
    // Create the native op.
    // The second output tensor contains the gradients, which is used for the gradient computation.
    val output = Tensor.fromNativeHandle(nativeCrossEntropyProxy(
      flattenedLogits.nativeHandle, flattenedLabels.nativeHandle,
      NativeTensorOpsNN.softmaxCrossEntropyWithLogits).apply(0))
    // The output shape should be the input shape without the axis over which the cross entropy was computed.
    val outputShape = Basic.slice(
      inputShape,
      Tensor.fill(inputShape.dataType, Shape(1))(0),
      Basic.expandDims(Math.subtract(inputRank, 1), -1))
    val reshapedTensor = Basic.reshape(output, outputShape)
    // We cast back to the original logits data type, if necessary.
    if (logits.dataType == FLOAT16)
      Math.cast(reshapedTensor, FLOAT16)
    else
      reshapedTensor
  }

  /** $OpDocNNSparseSoftmaxCrossEntropy
    *
    * @group NNOps
    * @param  logits Tensor of shape `[D0, D1, ..., Dr-1, numClasses]` (where `r` is the rank of `labels` and of the
    *                result) and data type [[FLOAT16]], [[FLOAT32]], or [[FLOAT64]], containing unscaled log
    *                probabilities.
    * @param  labels Tensor of shape `[D0, D1, ..., Dr-1]` (where `r` is the rank of `labels` and of the result) and
    *                data type [[INT32]] or [[INT64]]. Each entry in `labels` must be an index in `[0, numClasses)`.
    *                Other values will raise an exception when this op is run on a CPU, and return `NaN` values for the
    *                corresponding loss and gradient rows when this op is run on a GPU.
    * @param  axis   The class axis, along which the softmax is computed. Defaults to `-1`, which is the last axis.
    * @return Result as a new tensor, with the same shape as `labels` and the same data type as `logits`, containing the
    *         softmax cross entropy loss.
    */
  def sparseSoftmaxCrossEntropy(logits: Tensor, labels: Tensor, axis: Int = -1): Tensor = {
    val preciseLogits = if (logits.dataType == FLOAT16) Math.cast(logits, FLOAT32) else logits
    // Check if no reshapes are required.
    val output = {
      if (logits.rank == 2) {
        // Create the native op.
        // The second output tensor contains the gradients, which is used for the gradient computation.
        Tensor.fromNativeHandle(nativeCrossEntropyProxy(
          preciseLogits.nativeHandle, labels.nativeHandle,
          NativeTensorOpsNN.sparseSoftmaxCrossEntropyWithLogits).apply(0))
      } else {
        // Reshape logits to rank 2 and labels to rank 1.
        val flattenedLogits = NN.flattenOuterAxes(preciseLogits)
        val flattenedLabels = Basic.reshape(labels, -1)
        // Create the native op.
        // The second output tensor contains the gradients, which is used for the gradient computation.
        val output = Tensor.fromNativeHandle(nativeCrossEntropyProxy(
          flattenedLogits.nativeHandle, flattenedLabels.nativeHandle,
          NativeTensorOpsNN.sparseSoftmaxCrossEntropyWithLogits).apply(0))
        Basic.reshape(output, Basic.shape(labels))
      }
    }
    // We cast back to the original logits data type, if necessary.
    if (logits.dataType == FLOAT16)
      Math.cast(output, FLOAT16)
    else
      output
  }

  /** $OpDocNNSigmoidCrossEntropy
    *
    * @group NNOps
    * @param  logits  Tensor of shape `[D0, D1, ..., Dr-1, numClasses]` and data type [[FLOAT16]], [[FLOAT32]], or
    *                 [[FLOAT64]], containing unscaled log probabilities.
    * @param  labels  Tensor of shape `[D0, D1, ..., Dr-1, numClasses]` and data type [[FLOAT16]], [[FLOAT32]], or
    *                 [[FLOAT64]], where each row must be a valid probability distribution.
    * @param  weights Optionally, a coefficient to use for the positive examples.
    * @return Result as a new tensor, with rank one less than that of `logits` and the same data type as `logits`,
    *         containing the sigmoid cross entropy loss.
    */
  def sigmoidCrossEntropy(logits: Tensor, labels: Tensor, weights: Tensor = null): Tensor = {
    val output = {
      if (weights == null) {
        // Labels and logits must be of the same data type.
        val preciseLogits = if (logits.dataType == FLOAT16) Math.cast(logits, FLOAT32) else logits
        val preciseLabels = Math.cast(labels, preciseLogits.dataType)
        // The logistic loss formula from above is:
        //   x - x * z + log(1 + exp(-x))
        // For x < 0, a more numerically stable formula is:
        //   -x * z + log(1 + exp(x))
        // Note that these two expressions can be combined into the following single expression:
        //   max(x, 0) - x * z + log(1 + exp(-abs(x)))
        // To allow computing gradients at zero, we define custom versions of the max and the abs functions.
        val zeros = Tensor.zerosLike(preciseLogits)
        val condition = Math.greaterEqual(preciseLogits, zeros)
        val reluLogits = Math.select(condition, preciseLogits, zeros)
        val negativeAbsLogits = Math.select(condition, -preciseLogits, preciseLogits)
        Math.add(reluLogits - (preciseLogits * preciseLabels), Math.log1p(Math.exp(negativeAbsLogits)))
      } else {
        // Labels and logits must be of the same data type.
        val preciseLogits = if (logits.dataType == FLOAT16) Math.cast(logits, FLOAT32) else logits
        val preciseLabels = Math.cast(labels, preciseLogits.dataType)
        val preciseWeights = Math.cast(weights, preciseLogits.dataType)
        // The logistic loss formula from above is:
        //   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
        // For x < 0, a more numerically stable formula is:
        //   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
        // To avoid branching, we use the following single expression:
        //   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
        val logWeights = ((preciseWeights - 1) * preciseLabels) + 1
        Math.add(
          (1 - preciseLabels) * preciseLogits,
          (logWeights * Math.log1p(Math.exp(-Math.abs(preciseLogits)))) + relu(-preciseLogits))
      }
    }
    // We cast back to the original logits data type, if necessary.
    if (logits.dataType == FLOAT16)
      Math.cast(output, FLOAT16)
    else
      output
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
  def logPoissonLoss(logPredictions: Tensor, targets: Tensor, computeFullLoss: Boolean = false): Tensor = {
    val output = Math.exp(logPredictions) - (logPredictions * targets)
    if (computeFullLoss) {
      // Need to create constant tensors here so that their data types can be matched to that of the targets.
      val pointFive = Tensor(targets.dataType, 0.5)
      val twoPi = Tensor(targets.dataType, 2 * math.Pi)
      val stirlingApproximation = (targets * Math.log(targets)) - targets + (pointFive * Math.log(twoPi * targets))
      val zeros = Tensor.zerosLike(targets)
      val ones = Tensor.onesLike(targets)
      val condition = Math.logicalAnd(Math.greaterEqual(targets, zeros), Math.lessEqual(targets, ones))
      output + Math.select(condition, zeros, stirlingApproximation)
    } else {
      output
    }
  }

  //endregion Loss Ops

  /** $OpDocNNDropout
    *
    * @group NNOps
    * @param  input           Input tensor.
    * @param  keepProbability Probability (i.e., number in the interval `(0, 1]`) that each element is kept.
    * @param  noiseShape      [[INT32]] rank-1 tensor representing the shape for the randomly generated keep/drop flags.
    * @param  seed            Optional random seed, used to generate a random seed pair for the random number
    *                         generator, when combined with the graph-level seed.
    * @return Result as a new tensor that has the same shape as `input`.
    */
  def dropout(input: Tensor, keepProbability: Float, noiseShape: Tensor = null, seed: Option[Int] = None): Tensor = {
    require(keepProbability > 0.0 && keepProbability <= 1.0, s"'keepProbability' ($keepProbability) must be in (0, 1].")
    // Do nothing if we know that keepProbability == 1.
    if (keepProbability == 1.0) {
      input
    } else {
      val inferredNoiseShape = if (noiseShape == null) Basic.shape(input) else noiseShape
      // Uniform random variable in [keepProbability, 1.0 + keepProbability).
      val probability = Tensor(input.dataType, keepProbability)
      val random = Random.randomUniform(
        input.dataType, inferredNoiseShape, minValue = probability, maxValue = probability + 1.0, seed = seed)
      // 0.0 if in [keepProbability, 1.0) and 1.0 if [1.0, 1.0 + keepProbability).
      Math.divide(input, probability) * Math.floor(random)
    }
  }

  /** $OpDocNNTopK
    *
    * @group NNOps
    * @param  input  Input tensor whose last axis has size at least `k`.
    * @param  k      Scalar [[INT32]] tensor containing the number of top elements to look for along the last axis of
    *                `input`.
    * @param  sorted If `true`, the resulting `k` elements will be sorted by their values in descending order.
    * @return Tuple containing the created tensors: (i) `values`: the `k` largest elements along each last
    *         dimensional slice, and (ii) `indices`: the indices of `values` within the last axis of `input`.
    */
  def topK(
      input: Tensor, k: Tensor = 1, sorted: Boolean = true)(
      implicit context: DynamicVariable[Context]): (Tensor, Tensor) = {
    val outputs = NativeTensorOpsNN.topKV2(
      context.value.nativeHandle, input.nativeHandle, k.nativeHandle, sorted)
    (Tensor.fromNativeHandle(outputs(0)), Tensor.fromNativeHandle(outputs(1)))
  }

  /** $OpDocNNInTopK
    *
    * @group NNOps
    * @param  predictions [[FLOAT32]] tensor containing the predictions.
    * @param  targets     [[INT32]] or [[INT64]] tensor containing the targets.
    * @param  k           Scalar [[INT32]] or [[INT64]] tensor containing the number of top elements to look at.
    * @return Result as a new tensor.
    */
  def inTopK(predictions: Tensor, targets: Tensor, k: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    val mostPreciseDataType = DataType.mostPrecise(targets.dataType, k.dataType)
    Tensor.fromNativeHandle(NativeTensorOpsNN.inTopKV2(
      context.value.nativeHandle, predictions.nativeHandle, targets.cast(mostPreciseDataType).nativeHandle,
      k.cast(mostPreciseDataType).nativeHandle))
  }
}

private[api] object NN extends NN {
  private[ops] trait Implicits {
    implicit def tensorToNNOps(value: Tensor): NNOps = NNOps(value)
    implicit def tensorConvertibleToNNOps[T](value: T)(implicit f: (T) => Tensor): NNOps = NNOps(f(value))
  }

  case class NNOps private[ops](tensor: Tensor) {
    //region Core Ops

    /** $OpDocNNAddBias
      *
      * @group NNOps
      * @param  bias          Bias tensor that must be one-dimensional (i.e., it must have rank 1).
      * @param  cNNDataFormat Data format of the input and output tensors. With the default format [[NHWCFormat]], the
      *                       `bias` tensor will be added to the last dimension of the `value` tensor. Alternatively, the
      *                       format could be [[NCHWFormat]], and the `bias` tensor would be added to the third-to-last
      *                       dimension.
      * @return Result as a new tensor.
      */
    def addBias(bias: Tensor, cNNDataFormat: CNNDataFormat = CNNDataFormat.default): Tensor = {
      NN.addBias(tensor, bias, cNNDataFormat)
    }

    /** $OpDocNNLinear
      *
      * @group NNOps
      * @param  weights Weights tensor.
      * @param  bias    Bias tensor.
      * @return Result as a new tensor.
      */
    def linear(weights: Tensor, bias: Tensor): Tensor = NN.linear(tensor, weights, bias)

    /** $OpDocNNL2Normalize
      *
      * @group NNOps
      * @param  axes    Tensor containing the axes along which to normalize.
      * @param  epsilon Lower bound value for the norm. The created op will use `sqrt(epsilon)` as the divisor, if
      *                 `norm < sqrt(epsilon)`.
      * @return Result as a new tensor.
      */
    def l2Normalize(axes: Tensor, epsilon: Float = 1e-12f): Tensor = NN.l2Normalize(tensor, axes, epsilon)

    //endregion Core Ops

    //region Activation Ops

    /** $OpDocNNRelu
      *
      * @group NNOps
      * @param  alpha Slope of the negative section, also known as leakage parameter. If other than `0.0f`, the negative
      *               part will be equal to `alpha * x` instead of `0`. Defaults to `0`.
      * @return Result as a new tensor.
      */
    def relu(alpha: Float = 0.0f): Tensor = NN.relu(tensor, alpha)

    /** $OpDocNNRelu6
      *
      * @group NNOps
      * @return Result as a new tensor.
      */
    def relu6: Tensor = NN.relu6(tensor)

    /** $OpDocNNCrelu
      *
      * @group NNOps
      * @return Result as a new tensor.
      */
    def crelu: Tensor = NN.crelu(tensor)

    /** $OpDocNNElu
      *
      * @group NNOps
      * @return Result as a new tensor.
      */
    def elu: Tensor = NN.elu(tensor)

    /** $OpDocNNSelu
      *
      * @group NNOps
      * @return Result as a new tensor.
      */
    def selu: Tensor = NN.selu(tensor)

    /** $OpDocNNSoftplus
      *
      * @group NNOps
      * @return Result as a new tensor.
      */
    def softplus: Tensor = NN.softplus(tensor)

    /** $OpDocNNSoftsign
      *
      * @group NNOps
      * @return Result as a new tensor.
      */
    def softsign: Tensor = NN.softsign(tensor)

    //endregion Activation Ops

    /** $OpDocNNSoftmax
      *
      * @group NNOps
      * @param  axis Axis along which to perform the softmax. Defaults to `-1` denoting the last axis.
      * @return Result as a new tensor.
      */
    def softmax(axis: Int = -1): Tensor = NN.softmax(tensor, axis)

    /** $OpDocNNLogSoftmax
      *
      * @group NNOps
      * @param  axis Axis along which to perform the log-softmax. Defaults to `-1` denoting the last axis.
      * @return Result as a new tensor.
      */
    def logSoftmax(axis: Int = -1): Tensor = NN.logSoftmax(tensor, axis)

    /** $OpDocNNDropout
      *
      * @group NNOps
      * @param  keepProbability Probability (i.e., number in the interval `(0, 1]`) that each element is kept.
      * @param  noiseShape      [[INT32]] rank-1 tensor representing the shape for the randomly generated keep/drop flags.
      * @param  seed            Optional random seed, used to generate a random seed pair for the random number
      *                         generator, when combined with the graph-level seed.
      * @return Result as a new tensor that has the same shape as `input`.
      */
    def dropout(keepProbability: Float, noiseShape: Tensor = null, seed: Option[Int] = None): Tensor = {
      NN.dropout(tensor, keepProbability, noiseShape, seed)
    }

    /** $OpDocNNTopK
      *
      * @group NNOps
      * @param  k      Scalar [[INT32]] tensor containing the number of top elements to look for along the last axis of
      *                `input`.
      * @param  sorted If `true`, the resulting `k` elements will be sorted by their values in descending order.
      * @return Tuple containing the created op outputs: (i) `values`: the `k` largest elements along each last
      *         dimensional slice, and (ii) `indices`: the indices of `values` within the last axis of `input`.
      */
    def topK(k: Tensor = 1, sorted: Boolean = true): (Tensor, Tensor) = NN.topK(tensor, k, sorted)

    /** $OpDocNNInTopK
      *
      * @group NNOps
      * @param  targets [[INT32]] or [[INT64]] tensor containing the targets.
      * @param  k       Scalar [[INT32]] or [[INT64]] tensor containing the number of top elements to look at.
      * @return Result as a new tensor.
      */
    def inTopK(targets: Tensor, k: Tensor): Tensor = NN.inTopK(tensor, targets, k)
  }

  /** Creates an op that flattens the outer axes of `input` and keeps its last axis. */
  private[ops] def flattenOuterAxes(input: Tensor): Tensor = {
    val rank = Basic.rank(input)
    val lastAxisSize = Basic.slice(
      Basic.shape(input),
      Basic.expandDims(Math.subtract(rank, 1), -1),
      Tensor.fill(rank.dataType, Shape(1))(1))
    val output = Basic.reshape(input, Basic.concatenate(
      Seq(Tensor.fill(rank.dataType, Shape(1))(-1), lastAxisSize), 0))
    // Set the output shape, if known.
    val shape = input.shape
    if (shape.rank != -1) {
      val product = shape(0 :: -1).asArray.product
      if (product > -1)
        output.setShape(Shape(product, shape(-1)))
    }
    output
  }

  /** Creates an op that swaps the axes `axis1` and `axis2` in `input` and ignores all axes after `axis2`. */
  private[ops] def swapAxes(input: Tensor, axis1: Tensor, axis2: Tensor): Tensor = {
    Basic.transpose(
      input,
      Basic.concatenate(Seq(
        Math.range(0, axis1),
        axis2,
        Math.range(axis1 + 1, axis2),
        axis1), 0))
  }

  /** Creates an op that moves `axis` to the end. */
  private[ops] def moveAxisToEnd(input: Tensor, axis: Int, rank: Tensor): Tensor = {
    if (axis == -1) {
      input
    } else {
      val axisOutput = Tensor(rank.dataType, axis)
      Basic.transpose(
        input,
        Basic.concatenate(Seq(
          Math.range(0, axisOutput),
          Math.range(axisOutput + 1, rank),
          axisOutput), 0))
    }
  }
}
