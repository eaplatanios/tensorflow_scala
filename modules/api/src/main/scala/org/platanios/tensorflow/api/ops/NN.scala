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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.{InvalidArgumentException, InvalidShapeException}
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.NN._
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.api.ops.math.Math
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault

import scala.collection.compat.immutable.ArraySeq
import scala.language.postfixOps

/** Contains functions for constructing ops related to neural networks.
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
    * @param  name          Name for the created op.
    * @return Created op output.
    */
  def addBias[T: TF : IsNumeric](
      value: Output[T],
      bias: Output[T],
      cNNDataFormat: CNNDataFormat = CNNDataFormat.default,
      name: String = "AddBias"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "BiasAdd",
      name = name,
      input = (value, bias)
    ).setAttribute("data_format", cNNDataFormat.toString)
        .setGradientFn(addBiasGradient(_, _)(TF[T], IsNumeric[T]))
        .build().output
  }

  protected def addBiasGradient[T: TF : IsNumeric](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val cNNDataFormatName = {
      try {
        op.stringAttribute("data_format")
      } catch {
        case _: Throwable => CNNDataFormat.default.toString
      }
    }
    val gradient = Op.Builder[Output[T], Output[T]](
      opType = "BiasAddGrad",
      name = "BiasAddGradient",
      input = outputGradient
    ).setAttribute("data_format", cNNDataFormatName)
        .setGradientFn(addBiasHessian(_, _)(TF[T], IsNumeric[T]))
        .build().output
    (outputGradient, gradient)
  }

  protected def addBiasHessian[T: TF : IsNumeric](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val cNNDataFormatName = {
      try {
        op.stringAttribute("data_format")
      } catch {
        case _: Throwable => CNNDataFormat.default.toString
      }
    }
    val valueShape = Basic.shape(op.input)
    val biasShape = Basic.shape(outputGradient)
    val (expandedShape, tileMultiples) = cNNDataFormatName match {
      case "NHWC" =>
        val valuesLeft = valueShape(0 :: -1)
        val expandedShape = Basic.concatenate(Seq(Basic.onesLike(valuesLeft), biasShape), axis = 0)
        val tileMultiples = Basic.concatenate[Long](Seq(valuesLeft, 1), 0)
        (expandedShape, tileMultiples)
      case "NCHW" =>
        val valuesLeft = valueShape(0 :: -3)
        val valuesRight = valueShape(-2 ::)
        val expandedShape = Basic.concatenate(
          Seq(Basic.onesLike(valuesLeft), biasShape, Basic.onesLike(valuesRight)), axis = 0)
        val tileMultiples = Basic.concatenate[Long](Seq(valuesLeft, 1, valuesRight), 0)
        (expandedShape, tileMultiples)
    }
    Basic.tile(Basic.reshape(outputGradient, expandedShape), tileMultiples)
  }

  /** $OpDocNNLinear
    *
    * @group NNOps
    * @param  x       Input tensor.
    * @param  weights Weights tensor.
    * @param  bias    Bias tensor.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def linear[T: TF : IsNotQuantized](
      x: Output[T],
      weights: Output[T],
      bias: Output[T] = null,
      name: String = "Linear"
  ): Output[T] = {
    Op.nameScope(name) {
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
  }

  /** $OpDocNNL2Normalize
    *
    * @group NNOps
    * @param  x       Input tensor.
    * @param  axes    Tensor containing the axes along which to normalize.
    * @param  epsilon Lower bound value for the norm. The created op will use `sqrt(epsilon)` as the divisor, if
    *                 `norm < sqrt(epsilon)`.
    * @param  name    Name for the created op.
    * @return Created op output.
    */
  def l2Normalize[T: TF : IsNotQuantized, I: TF : IsIntOrLong](
      x: Output[T],
      axes: Output[I],
      epsilon: Float = 1e-12f,
      name: String = "L2Normalize"
  ): Output[T] = {
    Op.nameScope(name) {
      if (x.dataType == FLOAT64) {
        val squareSum = Math.sum(Math.square(x), axes = axes, keepDims = true)
        val xInverseNorm = Math.rsqrt(Math.maximum(squareSum, Basic.constant(epsilon).castTo[T]))
        Math.multiply(x, xInverseNorm)
      } else {
        val preciseX = x.castTo[Float]
        val squareSum = Math.sum(Math.square(preciseX), axes = axes, keepDims = true)
        val xInverseNorm = Math.rsqrt(Math.maximum(squareSum, Basic.constant(epsilon)))
        val result = Math.multiply(preciseX, xInverseNorm)
        result.castTo[T]
      }
    }
  }

  //endregion Core Ops

  //region Activation Ops

  /** $OpDocNNRelu
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  alpha Slope of the negative section, also known as leakage parameter. If other than `0.0f`, the negative
    *               part will be equal to `alpha * x` instead of `0`. Defaults to `0`.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def relu[T: TF : IsReal](
      input: Output[T],
      alpha: Float = 0.0f,
      name: String = "ReLU"
  ): Output[T] = {
    def reluOp[OL[A] <: OutputLike[A]](i: OL[T], n: String)(implicit
        ev: OutputOps.Aux[OL, T]
    ): OL[T] = {
      ev.applyUnary(i, o => {
        if (o.dataType.isInteger) {
          Op.Builder[Output[Float], Output[Float]](
            opType = "Relu",
            name = n,
            input = o.castTo[Float]
          ).setGradientFn(reluGradient(_, _)(TF[Float], IsReal[Float]))
              .build().output.castTo[T]
        } else {
          Op.Builder[Output[T], Output[T]](
            opType = "Relu",
            name = n,
            input = o
          ).setGradientFn(reluGradient(_, _)(TF[T], IsReal[T]))
              .build().output
        }
      })
    }

    if (alpha == 0.0f) {
      reluOp(input, name)
    } else {
      Op.nameScope(name) {
        val positive = reluOp(input, s"$name/PositivePart")
        val negative = reluOp(-input, s"$name/NegativePart")
        positive - (Basic.constant(alpha, Shape.scalar()).castTo[T] * negative)
      }
    }
  }

  protected def reluGradient[T: TF : IsReal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "ReluGrad",
      name = "ReLUGradient",
      input = (outputGradient, op.output)
    ).setGradientFn(reluHessian(_, _)(TF[T], IsReal[T]))
        .build().output
  }

  protected def reluHessian[T: TF : IsReal](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val gradient = Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "ReluGrad",
      name = "ReLUHessian",
      input = (outputGradient, op.input._2)
    ).setGradientFn(reluHessian(_, _)(TF[T], IsReal[T]))
        .build().output
    (gradient, Basic.zerosLike(op.input._2))
  }

  /** $OpDocNNRelu6
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def relu6[T: TF : IsReal, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "ReLU6"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(input, o => {
      Op.Builder[Output[T], Output[T]](
        opType = "Relu6",
        name = name,
        input = o
      ).setGradientFn(relu6Gradient(_, _)(TF[T], IsReal[T]))
          .build().output
    })
  }

  protected def relu6Gradient[T: TF : IsReal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Relu6Grad",
      name = "ReLU6Gradient",
      input = (outputGradient, op.input)
    ).setGradientFn(relu6Hessian(_, _)(TF[T], IsReal[T]))
        .build().output
  }

  protected def relu6Hessian[T: TF : IsReal](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val gradient = Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Relu6Grad",
      name = "ReLU6Hessian",
      input = (outputGradient, op.input._2)
    ).setGradientFn(reluHessian(_, _)(TF[T], IsReal[T]))
        .build().output
    (gradient, Basic.zerosLike(op.input._2))
  }

  /** $OpDocNNCrelu
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  axis  Along along which the output values are concatenated along.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def crelu[T: TF : IsReal](
      input: Output[T],
      axis: Output[Int] = -1,
      name: String = "CReLU"
  ): Output[T] = {
    Op.nameScope(name) {
      relu(Basic.concatenate(Seq(input, -input), axis = axis))
    }
  }

  /** $OpDocNNElu
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def elu[T: TF : IsReal, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "ELU"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(input, o => {
      Op.Builder[Output[T], Output[T]](
        opType = "Elu",
        name = name,
        input = o
      ).setGradientFn(eluGradient(_, _)(TF[T], IsReal[T]))
          .build().output
    })
  }

  protected def eluGradient[T: TF : IsReal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "EluGrad",
      name = "ELUGradient",
      input = (outputGradient, op.input)
    ).setGradientFn(eluHessian(_, _)(TF[T], IsReal[T]))
        .build().output
  }

  protected def eluHessian[T: TF : IsReal](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val zero = Basic.zeros[T](Shape())
    val gradient0 = Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "EluGrad",
      name = "ELUGradient",
      input = (outputGradient, op.input._2)
    ).setGradientFn(eluHessian(_, _)(TF[T], IsReal[T]))
        .build().output
    val gradient1 = Math.select(
      Math.less(op.input._2, zero),
      Math.multiply(outputGradient, op.input._1),
      Basic.zerosLike(op.input._2))
    (gradient0, gradient1)
  }

  /** $OpDocNNSelu
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def selu[T: TF : IsReal, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "SELU"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(input, o => {
      Op.Builder[Output[T], Output[T]](
        opType = "Selu",
        name = name,
        input = o
      ).setGradientFn(seluGradient(_, _)(TF[T], IsReal[T]))
          .build().output
    })
  }

  protected def seluGradient[T: TF : IsReal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "SeluGrad",
      name = "SELUGradient",
      input = (outputGradient, op.input)
    ).setGradientFn(eluHessian(_, _)(TF[T], IsReal[T]))
        .build().output
  }

  protected def seluHessian[T: TF : IsReal](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    Op.nameScope(s"${op.name}/SELUHessian") {
      val zero = Basic.zeros[T](Shape())
      val alpha = Basic.constant(1.7580993408473768599402175208123, name = "Alpha")
          .castTo[T]
      val gradient0 = Op.Builder[(Output[T], Output[T]), Output[T]](
        opType = "EluGrad",
        name = "ELUGradient",
        input = (outputGradient, op.output)
      ).setGradientFn(eluHessian(_, _)(TF[T], IsReal[T]))
          .build().output
      val gradient1 = Math.select(
        Math.less(op.input._2, zero),
        Op.Builder[(Output[T], Output[T]), Output[T]](
          opType = "EluGrad",
          name = "ELUGradient",
          input = (outputGradient, Math.add(op.output, alpha))
        ).setGradientFn(eluHessian(_, _)(TF[T], IsReal[T]))
            .build().output,
        Basic.zerosLike(op.input._2))

      Math.select(
        Math.less(op.input._2, zero),
        Math.multiply(outputGradient, op.input._1),
        Basic.zerosLike(op.input._2))
      (gradient0, gradient1)
    }
  }

  /** $OpDocNNSoftplus
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def softplus[T: TF : IsDecimal, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "Softplus"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(input, o => {
      Op.Builder[Output[T], Output[T]](
        opType = "Softplus",
        name = name,
        input = o
      ).setGradientFn(softplusGradient(_, _)(TF[T], IsDecimal[T]))
          .build().output
    })
  }

  protected def softplusGradient[T: TF : IsDecimal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "SoftplusGrad",
      name = "SoftplusGradient",
      input = (outputGradient, op.input)
    ).setGradientFn(softplusHessian(_, _)(TF[T], IsDecimal[T]))
        .build().output
  }

  protected def softplusHessian[T: TF : IsDecimal](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    Op.nameScope(s"${op.name}/SoftplusHessian") {
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val dy = op.input._1
        val x = op.input._2
        val ddy = Op.Builder[(Output[T], Output[T]), Output[T]](
          opType = "SoftplusGrad",
          name = "SoftplusGradient",
          input = (outputGradient, x)
        ).setGradientFn(softplusHessian(_, _)(TF[T], IsDecimal[T]))
            .build().output
        val two = Basic.constant(2.0f).castTo[T]
        val d2x = Math.multiply(outputGradient, dy) / (Math.exp(-x) + two + Math.exp(x))
        (ddy, d2x)
      }
    }
  }

  /** $OpDocNNSoftsign
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def softsign[T: TF : IsDecimal, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "Softsign"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(input, o => {
      Op.Builder[Output[T], Output[T]](
        opType = "Softsign",
        name = name,
        input = o
      ).setGradientFn(softsignGradient(_, _)(TF[T], IsDecimal[T]))
          .build().output
    })
  }

  protected def softsignGradient[T: TF : IsDecimal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "SoftsignGrad",
      name = "SoftsignGradient",
      input = (outputGradient, op.input)
    ).build().output
  }

  /** Helper function for [[softmax]] and [[logSoftmax]] that reshapes and transposes the input logits into
    * two-dimensional tensors and then creates the corresponding native op. The output is transposed and reshaped
    * back. */
  protected def softmaxHelper[T: TF : IsDecimal](
      logits: Output[T],
      opType: String,
      axis: Int = -1,
      name: String = "Softmax"
  ): Output[T] = {
    // We need the original shape of the logits for shape inference.
    val shape = logits.shape
    val isLastAxis = axis == -1 || axis == shape.rank - 1
    if (shape.rank == 2 && isLastAxis) {
      Op.Builder[Output[T], Output[T]](
        opType = opType,
        name = name,
        input = logits
      ).setGradientFn[Output[T], Output[T]]({
        if (opType == "Softmax")
          softmaxGradient(_, _)(TF[T], IsDecimal[T])
        else
          logSoftmaxGradient(_, _)(TF[T], IsDecimal[T])
      }).build().output
    } else if (isLastAxis) {
      Op.nameScope(name) {
        // If axis is the last axis, we simply reshape the logits to a matrix and apply the internal softmax.
        val inputShape = Basic.shape(logits)
        val flatLogits = flattenOuterAxes(logits)
        val output = Op.Builder[Output[T], Output[T]](
          opType = opType,
          name = name,
          input = flatLogits
        ).setGradientFn[Output[T], Output[T]]({
          if (opType == "Softmax")
            softmaxGradient(_, _)(TF[T], IsDecimal[T])
          else
            logSoftmaxGradient(_, _)(TF[T], IsDecimal[T])
        }).build().output
        Basic.reshape(output, inputShape)
      }
    } else {
      Op.nameScope(name) {
        // If axis is not the last dimension, we have to do a reshape and transpose so that we can still perform softmax
        // on its last dimension.
        // We swap the logits' axis of axis and its last axis.
        val inputRank = Basic.rank(logits)
        val modAxis = Basic.constant(axis) % Basic.rank(logits)
        val swappedLogits = swapAxes(logits, modAxis, Math.subtract(inputRank, 1))
        val shapeAfterSwap = Basic.shape(swappedLogits)
        // We reshape the logits into a matrix.
        val flatLogits = flattenOuterAxes(swappedLogits)
        // We perform the actual softmax on the last axis.
        var output = Op.Builder[Output[T], Output[T]](
          opType = opType,
          name = name,
          input = flatLogits
        ).setGradientFn[Output[T], Output[T]]({
          if (opType == "Softmax")
            softmaxGradient(_, _)(TF[T], IsDecimal[T])
          else
            logSoftmaxGradient(_, _)(TF[T], IsDecimal[T])
        }).build().output
        // We transform back the output tensor.
        output = Basic.reshape(output, shapeAfterSwap)
        output = swapAxes(output, modAxis, Math.subtract(inputRank, 1))
        // We make shape inference work since the reshape and the transpose may erase the static shape information.
        output.setShape(shape)
        output
      }
    }
  }

  /** $OpDocNNSoftmax
    *
    * @group NNOps
    * @param  logits Tensor containing the logits.
    * @param  axis   Axis along which to perform the softmax. Defaults to `-1` denoting the last axis.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def softmax[T: TF : IsDecimal](
      logits: Output[T],
      axis: Int = -1,
      name: String = "Softmax"
  ): Output[T] = {
    softmaxHelper(logits, "Softmax", axis, name)
  }

  /** $OpDocNNLogSoftmax
    *
    * @group NNOps
    * @param  logits Tensor containing the logits.
    * @param  axis   Axis along which to perform the log-softmax. Defaults to `-1` denoting the last axis.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def logSoftmax[T: TF : IsDecimal](
      logits: Output[T],
      axis: Int = -1,
      name: String = "LogSoftmax"
  ): Output[T] = {
    softmaxHelper(logits, "LogSoftmax", axis, name)
  }

  protected def softmaxGradient[T: TF : IsDecimal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val softmax = op.output
    (outputGradient - Math.sum(outputGradient * softmax, 1, keepDims = true)) * softmax
  }

  protected def logSoftmaxGradient[T: TF : IsDecimal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    val softmax = Math.exp(op.output)
    (outputGradient - Math.sum(outputGradient * softmax, 1, keepDims = true)) * softmax
  }

  //endregion Activation Ops

  //region Loss Ops

  /** $OpDocNNL2Loss
    *
    * @group NNOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def l2Loss[T: TF : IsDecimal](
      input: Output[T],
      name: String = "L2Loss"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "L2Loss",
      name = name,
      input = input
    ).setGradientFn(l2LossGradient(_, _)(TF[T], IsDecimal[T]))
        .build().output
  }

  protected def l2LossGradient[T: TF : IsDecimal](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Math.multiply(op.input, outputGradient)
  }

  /** $OpDocNNSoftmaxCrossEntropy
    *
    * @group NNOps
    * @param  logits Tensor of shape `[D0, D1, ..., Dr-1, numClasses]` which contains unscaled log probabilities.
    * @param  labels Tensor of shape `[D0, D1, ..., Dr-1, numClasses]`, where each row must be a valid probability
    *                distribution.
    * @param  axis   The class axis, along which the softmax is computed. Defaults to `-1`, which is the last axis.
    * @param  name   Name for the created op.
    * @return Created op output, with rank one less than that of `logits` and the same data type as `logits`, containing
    *         the softmax cross entropy loss.
    */
  def softmaxCrossEntropy[T: TF : IsDecimal](
      logits: Output[T],
      labels: Output[T],
      axis: Int = -1,
      name: String = "SoftmaxCrossEntropy"
  ): Output[T] = {
    Op.nameScope(name) {
      if (logits.dataType == FLOAT16) {
        val preciseLogits = logits.castTo[Float]
        val preciseLabels = labels.castTo[Float]
        val crossEntropy = softmaxCrossEntropy(preciseLogits, preciseLabels, axis)
        crossEntropy.castTo[T]
      } else {
        val inputRank = Basic.rank(logits)
        // We need the original shape of the logits for shape inference.
        val shape = logits.shape
        // We move axis to the end, if it's not already the last axis.
        val transposedLogits = moveAxisToEnd(logits, axis, inputRank)
        val transposedLabels = moveAxisToEnd(labels, axis, inputRank)
        val inputShape = Basic.shape(logits)
        // Flatten transposedLogits and transposedLabels into matrices.
        val flatLogits = flattenOuterAxes(transposedLogits)
        val flatLabels = flattenOuterAxes(transposedLabels)
        // Create the native op.
        // The second output tensor contains the gradients, which is used for the gradient computation.
        val output = Op.Builder[(Output[T], Output[T]), (Output[T], Output[T])](
          opType = "SoftmaxCrossEntropyWithLogits",
          name = name,
          input = (flatLogits, flatLabels)
        ).setGradientFn(softmaxCrossEntropyGradient(_, _)(TF[T], IsDecimal[T]))
            .build().output._1
        // The output shape should be the input shape without the axis over which the cross entropy was computed.
        val outputShape = Basic.slice[Long, Long](
          inputShape,
          Basic.zeros[Long](Shape(1)),
          Basic.expandDims(Math.subtract(inputRank, 1), -1))
        val reshapedOutput = Basic.reshape(output, outputShape)
        // We make shape inference work since the reshape and the transpose may erase the static shape information.
        if (shape.rank > -1) {
          def removeAt(array: Array[Int], axis: Int): Array[Int] = axis match {
            case a if a < 0 => removeAt(array, axis + array.length)
            case a if a == 0 => array.drop(1)
            case a if a == array.length => array.dropRight(1)
            case _ => array.take(axis) ++ array.drop(axis + 1)
          }

          reshapedOutput.setShape(Shape(removeAt(shape.asArray, axis)))
        }

        reshapedOutput
      }
    }
  }

  protected def softmaxCrossEntropyGradient[T: TF : IsDecimal](
      op: Op[(Output[T], Output[T]), (Output[T], Output[T])],
      outputGradient: (Output[T], Output[T])
  ): (Output[T], Output[T]) = {
    // outputGradient._1 is the back-propagated gradient for the cross entropy, and we multiply it with the gradients
    // (which is op.output._2). outputGradient._2 is the back-propagated gradient for the softmax gradient. There is
    // no gradient for the labels.
    val lossGradient = outputGradient._1
    val gradGradient = outputGradient._2
    val softmaxGradient = op.output._2
    val resultGradient = Basic.expandDims(lossGradient, -1) * softmaxGradient

    // Some introspection to check if the gradient is feeding zeros.
    val isGradGradientZero = {
      if (gradGradient.op.opType == "Zeros" || gradGradient.op.opType == "ZerosLike") {
        true
      } else {
        val constantFillValue = Output.constantValue(gradGradient)
        constantFillValue.isDefined && constantFillValue.get.entriesIterator.forall(_ == 0)
      }
    }

    val logits = op.input._1
    val labelsGradient = Basic.expandDims(lossGradient, -1) * -logSoftmax(logits)
    if (!isGradGradientZero) {
      val logitsSoftmax = softmax(logits)
      val gradient = resultGradient + (
          (gradGradient - Basic.squeeze(
            Math.matmul(
              Basic.expandDims(gradGradient, 1),
              Basic.expandDims(logitsSoftmax, 2)),
            Seq(1))) * logitsSoftmax)
      (gradient, labelsGradient)
    } else {
      (resultGradient, labelsGradient)
    }
  }

  /** $OpDocNNSparseSoftmaxCrossEntropy
    *
    * @group NNOps
    * @param  logits Tensor of shape `[D0, D1, ..., Dr-1, numClasses]` (where `r` is the rank of `labels` and of the
    *                result), which contains unscaled log probabilities.
    * @param  labels Tensor of shape `[D0, D1, ..., Dr-1]` (where `r` is the rank of `labels` and of the result). Each
    *                entry in `labels` must be an index in `[0, numClasses)`.
    *                Other values will raise an exception when this op is run on a CPU, and return `NaN` values for the
    *                corresponding loss and gradient rows when this op is run on a GPU.
    * @param  axis   The class axis, along which the softmax is computed. Defaults to `-1`, which is the last axis.
    * @param  name   Name for the created op.
    * @return Created op output, with the same shape as `labels` and the same data type as `logits`, containing the
    *         softmax cross entropy loss.
    */
  def sparseSoftmaxCrossEntropy[T: TF : IsDecimal, I: TF : IsIntOrLong](
      logits: Output[T],
      labels: Output[I],
      axis: Int = -1,
      name: String = "SparseSoftmaxCrossEntropy"
  ): Output[T] = {
    Op.nameScope(name) {
      if (logits.dataType == FLOAT16) {
        val preciseLogits = logits.castTo[Float]
        val crossEntropy = sparseSoftmaxCrossEntropy(preciseLogits, labels, axis)
        crossEntropy.castTo[T]
      } else if (logits.rank == 2) { // Check if no reshapes are required.
        // Create the native op.
        // The second output tensor contains the gradients, which is used for the gradient computation.
        Op.Builder[(Output[T], Output[I]), (Output[T], Output[T])](
          opType = "SparseSoftmaxCrossEntropyWithLogits",
          name = name,
          input = (logits, labels)
        ).setGradientFn(sparseSoftmaxCrossEntropyGradient(_, _)(TF[T], IsDecimal[T], TF[I], IsIntOrLong[I]))
            .build().output._1
      } else {
        // Reshape logits to rank 2 and labels to rank 1.
        val flatLogits = flattenOuterAxes(logits)
        val flatLabels = Basic.reshape(labels, Shape(-1))
        // Create the native op.
        // The second output tensor contains the gradients, which is used for the gradient computation.
        val output = Op.Builder[(Output[T], Output[I]), (Output[T], Output[T])](
          opType = "SparseSoftmaxCrossEntropyWithLogits",
          name = name,
          input = (flatLogits, flatLabels)
        ).setGradientFn(sparseSoftmaxCrossEntropyGradient(_, _)(TF[T], IsDecimal[T], TF[I], IsIntOrLong[I]))
            .build().output._1
        val reshapedOutput = Basic.reshape(output, Basic.shape(labels))
        reshapedOutput.setShape(labels.shape)
        reshapedOutput
      }
    }
  }

  protected def sparseSoftmaxCrossEntropyGradient[T: TF : IsDecimal, I: TF : IsIntOrLong](
      op: Op[(Output[T], Output[I]), (Output[T], Output[T])],
      outputGradient: (Output[T], Output[T])
  ): (Output[T], Output[I]) = {
    // outputGradients(0) is the back-propagated gradient for the cross entropy, and we multiply it with the gradients
    // (which is op.outputs(1)). There is no gradient for the labels.
    val lossGradient = outputGradient._1
    // Currently there is no way to take the second derivative of this op due to the fused implementation's
    // interaction with tf.gradients(). Therefore, we make sure we silently prevent incorrect results by raising an
    // error if the second derivative is requested via Basic.preventGradient().
    val softmaxGradient = Basic.preventGradient(
      op.output._2, message = "Currently there is no way to take the second derivative of " +
          "SparseSoftmaxCrossEntropyWithLogits due to the fused implementation's interaction with tf.gradients().")
    (Basic.expandDims(lossGradient, axis = -1) * softmaxGradient, null)
  }

  /** $OpDocNNSigmoidCrossEntropy
    *
    * @group NNOps
    * @param  logits  Tensor of shape `[D0, D1, ..., Dr-1, numClasses]`, which contains unscaled log probabilities.
    * @param  labels  Tensor of shape `[D0, D1, ..., Dr-1, numClasses]`, where each row must be a valid probability
    *                 distribution.
    * @param  weights Optionally, a coefficient to use for the positive examples.
    * @param  name    Name for the created op.
    * @return Created op output, with rank one less than that of `logits` and the same data type as `logits`, containing
    *         the sigmoid cross entropy loss.
    */
  def sigmoidCrossEntropy[T: TF : IsDecimal](
      logits: Output[T],
      labels: Output[T],
      weights: Output[T] = null,
      name: String = "SigmoidCrossEntropy"
  ): Output[T] = {
    Op.nameScope(name) {
      if (logits.dataType == FLOAT16) {
        val preciseLogits = logits.castTo[Float]
        val preciseLabels = labels.castTo[Float]
        val preciseWeights = if (weights == null) null else weights.castTo[Float]
        val crossEntropy = sigmoidCrossEntropy(preciseLogits, preciseLabels, preciseWeights, name)
        crossEntropy.castTo[T]
      } else if (weights == null) {
        // The logistic loss formula from above is:
        //   x - x * z + log(1 + exp(-x))
        // For x < 0, a more numerically stable formula is:
        //   -x * z + log(1 + exp(x))
        // Note that these two expressions can be combined into the following single expression:
        //   max(x, 0) - x * z + log(1 + exp(-abs(x)))
        // To allow computing gradients at zero, we define custom versions of the max and the abs functions.
        val zeros = Basic.zerosLike(logits)
        val condition = Math.greaterEqual(logits, zeros)
        val reluLogits = Math.select(condition, logits, zeros)
        val negativeAbsLogits = Math.select(condition, -logits, logits)
        Math.add(
          reluLogits - (logits * labels),
          Math.log1p(Math.exp(negativeAbsLogits)))
      } else {
        // The logistic loss formula from above is:
        //   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
        // For x < 0, a more numerically stable formula is:
        //   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
        // To avoid branching, we use the following single expression:
        //   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
        val one = Basic.ones[T](Shape())
        val logWeights = ((weights - one) * labels) + one
        Math.add(
          (one - labels) * logits,
          (logWeights * Math.log1p(Math.exp(-Math.abs(logits)))) + relu(-logits))
      }
    }
  }

  /** $OpDocNNLogPoissonLoss
    *
    * @group NNOps
    * @param  logPredictions  Tensor containing the log-predictions.
    * @param  targets         Tensor with the same shape as `logPredictions`, containing the target values.
    * @param  computeFullLoss If `true`, Stirling's Approximation is used to approximate the full loss. Defaults to
    *                         `false`, meaning that the constant term is ignored.
    * @param  name            Name for the created op.
    * @return Created op output.
    */
  def logPoissonLoss[T: TF : IsDecimal](
      logPredictions: Output[T],
      targets: Output[T],
      computeFullLoss: Boolean = false,
      name: String = "LogPoissonLoss"
  ): Output[T] = {
    Op.nameScope(name) {
      val output = Math.exp(logPredictions) - (logPredictions * targets)
      if (computeFullLoss) {
        // Need to create constant tensors here so that their data types can be matched to that of the targets.
        val pointFive = Basic.constant(0.5).castTo[T]
        val twoPi = Basic.constant(2 * scala.math.Pi).castTo[T]
        val stirlingApproximation = (targets * Math.log(targets)) - targets + (pointFive * Math.log(twoPi * targets))
        val zeros = Basic.zerosLike(targets)
        val ones = Basic.onesLike(targets)
        val condition = Math.logicalAnd(Math.greaterEqual(targets, zeros), Math.lessEqual(targets, ones))
        output + Math.select(condition, zeros, stirlingApproximation)
      } else {
        output
      }
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
    * @param  name                   Name prefix to use for the created ops.
    * @return Created op output.
    * @throws InvalidShapeException If any of `logits`, `labels`, or `weights` has invalid shape.
    */
  @throws[InvalidShapeException]
  def sequenceLoss[T: TF : IsDecimal, L: TF](
      logits: Output[T],
      labels: Output[L],
      lossFn: (Output[T], Output[L]) => Output[T],
      weights: Output[T] = null,
      averageAcrossTimeSteps: Boolean = true,
      averageAcrossBatch: Boolean = true,
      name: String = "SequenceLoss"
  ): Output[T] = {
    if (logits.rank != 3)
      throw InvalidShapeException(
        s"'logits' must have shape [batchSize, sequenceLength, numClasses], but had: ${logits.shape}.")
    if (labels.rank != 2)
      throw InvalidShapeException(s"'labels' must have shape [batchSize, sequenceLength], but had: ${labels.shape}.")
    if (weights != null && weights.rank != 2)
      throw InvalidShapeException(s"'weights' must have shape [batchSize, sequenceLength], but had: ${weights.shape}.")

    Op.nameScope(name) {
      val numClasses = Basic.shape(logits).slice(2)
      val flatLogits = Basic.reshape(logits, Basic.stack[Long](Seq(-1, numClasses)))
      val flatLabels = Basic.reshape(labels, Shape(-1))
      var loss = lossFn(flatLogits, flatLabels)

      if (weights != null)
        loss = loss * Basic.reshape(weights, Shape(-1))

      if (averageAcrossTimeSteps && averageAcrossBatch) {
        loss = Math.sum(loss)
        val totalSize = {
          if (weights != null) {
            val eps = Basic.constant(1e-12f, name = "Epsilon").castTo[T]
            Math.sum(weights) + eps
          } else {
            Basic.size(flatLabels).castTo[T]
          }
        }
        loss = Math.divide(loss, totalSize)
      } else {
        loss = Basic.reshape(loss, Basic.shape(logits).slice(0 :: 2))
        loss.setShape(logits.shape(0 :: 2))
      }

      if (averageAcrossTimeSteps && !averageAcrossBatch) {
        loss = Math.sum(loss, axes = 1)
        val totalSize = {
          if (weights != null) {
            val eps = Basic.constant(1e-12f, name = "Epsilon").castTo[T]
            Math.sum(weights, axes = 1) + eps
          } else {
            Basic.shape(labels).slice(1).castTo[T]
          }
        }
        loss = Math.divide(loss, totalSize)
      }

      if (!averageAcrossTimeSteps && averageAcrossBatch) {
        loss = Math.sum(loss, axes = 0)
        val totalSize = {
          if (weights != null) {
            val eps = Basic.constant(1e-12f, name = "Epsilon").castTo[T]
            Math.sum(weights, axes = 0) + eps
          } else {
            Basic.shape(labels).slice(0).castTo[T]
          }
        }
        loss = Math.divide(loss, totalSize)
      }
      loss
    }
  }

  //endregion Loss Ops

  /** Returns the `noiseShape` for the provided input, making the best effort possible to deal with unknown sizes. */
  private[api] def getNoiseShape[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      noiseShape: Output[I]
  ): Output[I] = {
    if (noiseShape == null) {
      Basic.shape(input).castTo[I]
    } else if (input.rank != -1 && input.rank == noiseShape.rank) {
      Shape.fromSeq(ArraySeq.unsafeWrapArray(input.shape.asArray.zip(noiseShape.shape.asArray).map {
        case (inputAxisSize, noiseShapeAxisSize)
          if noiseShapeAxisSize == -1 && inputAxisSize != -1 => inputAxisSize
        case (_, noiseShapeAxisSize) => noiseShapeAxisSize
      })).toOutput.castTo[I]
    } else {
      noiseShape
    }
  }

  /** $OpDocNNDropout
    *
    * @group NNOps
    * @param  input           Input tensor.
    * @param  keepProbability Probability (i.e., number in the interval `(0, 1]`) that each element is kept.
    * @param  scaleOutput     If `true`, the outputs will be divided by the keep probability.
    * @param  noiseShape      Rank-1 tensor representing the shape for the randomly generated keep/drop flags.
    * @param  seed            Optional random seed, used to generate a random seed pair for the random number
    *                         generator, when combined with the graph-level seed.
    * @param  name            Name for the created op.
    * @return Created op output that has the same shape as `input`.
    * @throws IllegalArgumentException If `keepProbability` is not in the interval `(0, 1]`.
    */
  @throws[IllegalArgumentException]
  def dropout[T: TF : IsHalfOrFloatOrDouble, I: IntDefault : TF : IsIntOrLong](
      input: Output[T],
      keepProbability: Float,
      scaleOutput: Boolean = true,
      noiseShape: Output[I] = null,
      seed: Option[Int] = None,
      name: String = "Dropout"
  ): Output[T] = {
    require(
      keepProbability > 0.0 && keepProbability <= 1.0,
      s"'keepProbability' ($keepProbability) must be in (0, 1].")
    // Do nothing if we know that keepProbability == 1.
    if (keepProbability == 1.0) {
      input
    } else {
      val keepProbabilityOutput = Basic.constant(keepProbability).castTo[T]
      dynamicDropout(input, keepProbabilityOutput, scaleOutput, noiseShape, seed, name)
    }
  }

  /** $OpDocNNDropout
    *
    * @group NNOps
    * @param  input           Input tensor.
    * @param  keepProbability Probability (i.e., scalar in the interval `(0, 1]`) that each element is kept.
    * @param  scaleOutput     If `true`, the outputs will be divided by the keep probability.
    * @param  noiseShape      Rank-1 tensor representing the shape for the randomly generated keep/drop flags.
    * @param  seed            Optional random seed, used to generate a random seed pair for the random number
    *                         generator, when combined with the graph-level seed.
    * @param  name            Name for the created op.
    * @return Created op output that has the same shape as `input`.
    */
  def dynamicDropout[T: TF : IsHalfOrFloatOrDouble, I: IntDefault : TF : IsIntOrLong](
      input: Output[T],
      keepProbability: Output[T],
      scaleOutput: Boolean = true,
      noiseShape: Output[I] = null,
      seed: Option[Int] = None,
      name: String = "Dropout"
  ): Output[T] = {
    Op.nameScope(name) {
      val one = Basic.ones[T](Shape())
      val noiseShapeWithDefault = getNoiseShape(input, noiseShape)
      // Uniform random variable in [keepProbability, 1.0 + keepProbability).
      val random = Random.randomUniform(
        noiseShapeWithDefault,
        minValue = keepProbability,
        maxValue = keepProbability + one,
        seed = seed)
      // 0.0 if in [keepProbability, 1.0) and 1.0 if [1.0, 1.0 + keepProbability).
      val binaryTensor = Math.floor(random)
      val output = {
        if (scaleOutput)
          Math.divide(input, keepProbability) * binaryTensor
        else
          input * binaryTensor
      }
      output.setShape(input.shape)
      output
    }
  }

  /** $OpDocNNTopK
    *
    * @group NNOps
    * @param  input  Input tensor whose last axis has size at least `k`.
    * @param  k      Scalar tensor containing the number of top elements to look for along the last axis of `input`.
    * @param  sorted If `true`, the resulting `k` elements will be sorted by their values in descending order.
    * @param  name   Name for the created op.
    * @return Tuple containing the created op outputs: (i) `values`: the `k` largest elements along each last
    *         dimensional slice, and (ii) `indices`: the indices of `values` within the last axis of `input`.
    */
  def topK[T: TF : IsReal](
      input: Output[T],
      k: Output[Int],
      sorted: Boolean = true,
      name: String = "TopK"
  ): (Output[T], Output[Int]) = {
    Op.Builder[(Output[T], Output[Int]), (Output[T], Output[Int])](
      opType = "TopKV2",
      name = name,
      input = (input, k)
    ).setAttribute("sorted", sorted)
        .setGradientFn(topKGradient(_, _)(TF[T], IsReal[T]))
        .build().output
  }

  protected def topKGradient[T: TF : IsReal](
      op: Op[(Output[T], Output[Int]), (Output[T], Output[Int])],
      outputGradient: (Output[T], Output[Int])
  ): (Output[T], Output[Int]) = {
    // Flatten indices to 2-D.
    val indicesShape = Basic.shape(op.output._2)
    val indicesLastAxis = Basic.gather(indicesShape, Basic.size(indicesShape) - 1L, axis = 0)
    val indices2D = Basic.reshape(op.output._2.castTo[Long], Basic.stack[Long](Seq(-1L, indicesLastAxis)))

    val inputShape = Basic.shape(op.input._1)
    val inputLastAxis = Basic.gather(inputShape, Basic.size(inputShape) - 1L, axis = 0)
    val outerAxis = Basic.shape(indices2D).slice(0)

    // Compute linear indices (flattened to 1-D).
    val flatIndices = Basic.reshape(
      indices2D + Basic.expandDims(Math.range(
        start = 0L,
        limit = outerAxis * indicesLastAxis,
        delta = inputLastAxis), axis = -1), -1)

    // Substitute gradient to appropriate locations and fill the rest with zeros, finally reshaping it to the original
    // input shape.
    (Basic.reshape(
      SparseOutput(
        indices = flatIndices,
        values = Basic.reshape(outputGradient._1, -1),
        denseShape = Basic.reshape(Math.prod(inputShape), 1)).toOutput(validateIndices = false),
      inputShape), Basic.zeros[Int](Shape.scalar()))
  }

  /** $OpDocNNInTopK
    *
    * @group NNOps
    * @param  predictions Tensor containing the predictions.
    * @param  targets     Tensor containing the targets.
    * @param  k           Scalar tensor containing the number of top elements to look at.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def inTopK[I: TF : IsIntOrLong](
      predictions: Output[Float],
      targets: Output[I],
      k: Output[I],
      name: String = "InTopK"
  ): Output[Boolean] = {
    Op.Builder[(Output[Float], Output[I], Output[I]), Output[Boolean]](
      opType = "InTopKV2",
      name = name,
      input = (predictions, targets, k)
    ).build().output
  }

  //region Convolution Ops

  /** $OpDocNNConv2D
    *
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
    * @param  name          Name for the created op.
    * @return Created op output, which is a 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def conv2D[T: TF : IsDecimal](
      input: Output[T],
      filter: Output[T],
      stride1: Long,
      stride2: Long,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      // TODO: [OPS|NN] Enforce the batch and depth dilation constraint at compile time.
      dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
      useCuDNNOnGPU: Boolean = true,
      name: String = "Conv2D"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Conv2D",
      name = name,
      input = (input, filter)
    ).setAttribute("strides", Array[Long](1, stride1, stride2, 1))
        .setAttribute("padding", padding.name)
        .setAttribute("data_format", dataFormat.name)
        .setAttribute("dilations", Array[Long](dilations._1, dilations._2, dilations._3, dilations._4))
        .setAttribute("use_cudnn_on_gpu", useCuDNNOnGPU)
        .setGradientFn(conv2DGradient(_, _)(TF[T], IsDecimal[T]))
        .build().output
  }

  protected def conv2DGradient[T: TF : IsDecimal](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val strides = op.longArrayAttribute("strides")
    val padding = ConvPaddingMode.fromName(op.stringAttribute("padding"))
    val dataFormat = CNNDataFormat.fromName(op.stringAttribute("data_format"))
    val dilations = op.longArrayAttribute("dilations")
    val useCuDNNOnGPU = op.booleanAttribute("use_cudnn_on_gpu")
    val inputShapes = Basic.shapeN[T, Int](Seq(op.input._1, op.input._2))
    (conv2DBackpropInput(
      inputShapes(0), op.input._2, outputGradient, strides(1).toInt, strides(2).toInt, padding, dataFormat,
      (dilations(0).toInt, dilations(1).toInt, dilations(2).toInt, dilations(3).toInt), useCuDNNOnGPU),
        conv2DBackpropFilter(
          op.input._1, inputShapes(1), outputGradient, strides(1).toInt, strides(2).toInt, padding, dataFormat,
          (dilations(0).toInt, dilations(1).toInt, dilations(2).toInt, dilations(3).toInt), useCuDNNOnGPU))
  }

  /** $OpDocNNConv2DBackpropInput
    *
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
    * @param  name           Name for the created op.
    * @return Created op output, which is a 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def conv2DBackpropInput[T: TF : IsDecimal](
      inputSizes: Output[Int],
      filter: Output[T],
      outputGradient: Output[T],
      stride1: Long,
      stride2: Long,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      // TODO: [OPS/NN] Enforce the batch and depth dilation constraint at compile time.
      dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
      useCuDNNOnGPU: Boolean = true,
      name: String = "Conv2DBackpropInput"
  ): Output[T] = {
    Op.Builder[(Output[Int], Output[T], Output[T]), Output[T]](
      opType = "Conv2DBackpropInput",
      name = name,
      input = (inputSizes, filter, outputGradient)
    ).setAttribute("strides", Array[Long](1, stride1, stride2, 1))
        .setAttribute("padding", padding.name)
        .setAttribute("data_format", dataFormat.name)
        .setAttribute("dilations", Array[Long](dilations._1, dilations._2, dilations._3, dilations._4))
        .setAttribute("use_cudnn_on_gpu", useCuDNNOnGPU)
        .build().output
  }

  /** $OpDocNNConv2DBackpropFilter
    *
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
    * @param  name           Name for the created op.
    * @return Created op output, which is a 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def conv2DBackpropFilter[T: TF : IsDecimal](
      input: Output[T],
      filterSizes: Output[Int],
      outputGradient: Output[T],
      stride1: Long,
      stride2: Long,
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      // TODO: [OPS/NN] Enforce the batch and depth dilation constraint at compile time.
      dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
      useCuDNNOnGPU: Boolean = true,
      name: String = "Conv2DBackpropFilter"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[Int], Output[T]), Output[T]](
      opType = "Conv2DBackpropFilter",
      name = name,
      input = (input, filterSizes, outputGradient)
    ).setAttribute("strides", Array[Long](1, stride1, stride2, 1))
        .setAttribute("padding", padding.name)
        .setAttribute("data_format", dataFormat.name)
        .setAttribute("dilations", Array[Long](dilations._1, dilations._2, dilations._3, dilations._4))
        .setAttribute("use_cudnn_on_gpu", useCuDNNOnGPU)
        .build().output
  }

  //endregion Convolution Ops

  //region Pooling Ops

  /** $OpDocNNMaxPool
    *
    * @param  input      4-D tensor whose dimension order is interpreted according to the value of `dataFormat`.
    * @param  windowSize The size of the pooling window for each dimension of the input tensor.
    * @param  strides    Strides for the sliding window. Strides in the batch and depth dimensions must be set to `1`.
    * @param  padding    Padding mode to use.
    * @param  dataFormat Format of the input and output data.
    * @param  name       Name for the created op.
    * @return Created op output, which is a 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def maxPool[T: TF : IsNumeric](
      input: Output[T],
      windowSize: Output[Int],
      // TODO: [OPS|NN] Enforce the batch and depth stride constraint at compile time.
      strides: Output[Int],
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      name: String = "MaxPool"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[Int], Output[Int]), Output[T]](
      opType = "MaxPoolV2",
      name = name,
      input = (input, windowSize, strides)
    ).setAttribute("padding", padding.name)
        .setAttribute("data_format", dataFormat.name)
        .setGradientFn(maxPoolGradient(_, _)(TF[T], IsNumeric[T]))
        .build().output
  }

  protected def maxPoolGradient[T: TF : IsNumeric](
      op: Op[(Output[T], Output[Int], Output[Int]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[Int], Output[Int]) = {
    val windowSize = op.input._2
    val strides = op.input._3
    val padding = ConvPaddingMode.fromName(op.stringAttribute("padding"))
    val dataFormat = CNNDataFormat.fromName(op.stringAttribute("data_format"))
    (maxPoolGrad(
      op.input._1, op.output, outputGradient, windowSize, strides,
      padding, dataFormat), null, null)
  }

  /** $OpDocNNMaxPoolGrad
    *
    * @param  originalInput  Original input tensor.
    * @param  originalOutput Original output tensor.
    * @param  outputGradient 4-D tensor containing the gradients w.r.t. the output of the max pooling and whose shape
    *                        depends on the value of `dataFormat`.
    * @param  windowSize     The size of the pooling window for each dimension of the input tensor.
    * @param  strides        Strides for the sliding window. Strides in the batch and depth dimensions must be set to `1`.
    * @param  padding        Padding mode to use.
    * @param  dataFormat     Format of the input and output data.
    * @param  name           Name for the created op.
    * @return Created op output, which is a 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def maxPoolGrad[T: TF : IsNumeric](
      originalInput: Output[T],
      originalOutput: Output[T],
      outputGradient: Output[T],
      windowSize: Output[Int],
      // TODO: [OPS|NN] Enforce the batch and depth stride constraint at compile time.
      strides: Output[Int],
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      name: String = "MaxPoolGrad"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T], Output[T], Output[Int], Output[Int]), Output[T]](
      opType = "MaxPoolGradV2",
      name = name,
      input = (originalInput, originalOutput, outputGradient, windowSize, strides)
    ).setAttribute("padding", padding.name)
        .setAttribute("data_format", dataFormat.name)
        .setGradientFn(maxPoolHessian(_, _)(TF[T], IsNumeric[T]))
        .build().output
  }

  protected def maxPoolHessian[T: TF : IsNumeric](
      op: Op[(Output[T], Output[T], Output[T], Output[Int], Output[Int]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T], Output[T], Output[Int], Output[Int]) = {
    val windowSize = op.input._4
    val strides = op.input._5
    val padding = ConvPaddingMode.fromName(op.stringAttribute("padding"))
    val dataFormat = CNNDataFormat.fromName(op.stringAttribute("data_format"))
    (Basic.zerosLike(op.input._1),
        Basic.zerosLike(op.input._2),
        maxPoolGradGrad(
          op.input._1, op.input._2, outputGradient, windowSize, strides,
          padding, dataFormat), null, null)
  }

  /** $OpDocNNMaxPoolGradGrad
    *
    * @param  originalInput  Original input tensor.
    * @param  originalOutput Original output tensor.
    * @param  outputGradient 4-D tensor containing the gradients w.r.t. the output of the max pooling and whose shape
    *                        depends on the value of `dataFormat`.
    * @param  windowSize     The size of the pooling window for each dimension of the input tensor.
    * @param  strides        Strides for the sliding window. Strides in the batch and depth dimensions must be set to `1`.
    * @param  padding        Padding mode to use.
    * @param  dataFormat     Format of the input and output data.
    * @param  name           Name for the created op.
    * @return Created op output, which is a 4-D tensor whose dimension order depends on the value of `dataFormat`.
    */
  def maxPoolGradGrad[T: TF : IsNumeric](
      originalInput: Output[T],
      originalOutput: Output[T],
      outputGradient: Output[T],
      windowSize: Output[Int],
      // TODO: [OPS|NN] Enforce the batch and depth stride constraint at compile time.
      strides: Output[Int],
      padding: ConvPaddingMode,
      dataFormat: CNNDataFormat = CNNDataFormat.default,
      name: String = "MaxPoolGradGrad"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T], Output[T], Output[Int], Output[Int]), Output[T]](
      opType = "MaxPoolGradGradV2",
      name = name,
      input = (originalInput, originalOutput, outputGradient, windowSize, strides)
    ).setAttribute("padding", padding.name)
        .setAttribute("data_format", dataFormat.name)
        .setGradientFn(maxPoolHessianGradient(_, _)(TF[T], IsNumeric[T]))
        .build().output
  }

  protected def maxPoolHessianGradient[T: TF : IsNumeric](
      op: Op[(Output[T], Output[T], Output[T], Output[Int], Output[Int]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T], Output[T], Output[Int], Output[Int]) = {
    val windowSize = op.input._4
    val strides = op.input._5
    val padding = ConvPaddingMode.fromName(op.stringAttribute("padding"))
    val dataFormat = CNNDataFormat.fromName(op.stringAttribute("data_format"))
    (Basic.zerosLike(op.input._1),
        Basic.zerosLike(op.input._2),
        maxPoolGrad(
          op.input._1, op.input._2, outputGradient, windowSize, strides,
          padding, dataFormat), null, null)
  }

  //endregion Pooling Ops

  //region Normalization Ops

  /** $OpDocNNLocalResponseNormalization
    *
    * @group NNOps
    * @param  input       Input tensor.
    * @param  depthRadius Half-width of the 1-D normalization window.
    * @param  bias        Offset (usually positive to avoid dividing by 0).
    * @param  alpha       Scale factor (usually positive).
    * @param  beta        Exponent.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def lrn[T: TF : IsTruncatedHalfOrHalfOrFloat](
      input: Output[T],
      depthRadius: Int = 5,
      bias: Float = 1.0f,
      alpha: Float = 1.0f,
      beta: Float = 0.5f,
      name: String = "LRN"
  ): Output[T] = {
    localResponseNormalization(input, depthRadius, bias, alpha, beta, name)
  }

  /** $OpDocNNLocalResponseNormalization
    *
    * @group NNOps
    * @param  input       Input tensor.
    * @param  depthRadius Half-width of the 1-D normalization window.
    * @param  bias        Offset (usually positive to avoid dividing by 0).
    * @param  alpha       Scale factor (usually positive).
    * @param  beta        Exponent.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  def localResponseNormalization[T: TF : IsTruncatedHalfOrHalfOrFloat](
      input: Output[T],
      depthRadius: Int = 5,
      bias: Float = 1.0f,
      alpha: Float = 1.0f,
      beta: Float = 0.5f,
      name: String = "LocalResponseNormalization"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "LRN",
      name = name,
      input = input
    ).setAttribute("depth_radius", depthRadius)
        .setAttribute("bias", bias)
        .setAttribute("alpha", alpha)
        .setAttribute("beta", beta)
        .setGradientFn(lrnGradient(_, _)(TF[T], IsTruncatedHalfOrHalfOrFloat[T]))
        .build().output
  }

  protected def lrnGradient[T: TF : IsTruncatedHalfOrHalfOrFloat](
      op: Op[Output[T], Output[T]],
      outputGradient: Output[T]
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T], Output[T]), Output[T]](
      opType = "LRNGrad",
      name = "LRNGrad",
      input = (outputGradient, op.input, op.output)
    ).setAttribute("depth_radius", op.longAttribute("depth_radius"))
        .setAttribute("bias", op.floatAttribute("bias"))
        .setAttribute("alpha", op.floatAttribute("alpha"))
        .setAttribute("beta", op.floatAttribute("beta"))
        .build().output
  }

  /** $OpDocNNBatchNormalization
    *
    * @param  x        Input tensor of arbitrary dimensionality.
    * @param  mean     Mean tensor.
    * @param  variance Variance tensor.
    * @param  offset   Optional offset tensor, often denoted `beta` in equations.
    * @param  scale    Optional scale tensor, often denoted `gamma` in equations.
    * @param  epsilon  Small floating point number added to the variance to avoid division by zero.
    * @param  name     Name for the created ops.
    * @return Batch-normalized tensor `x`.
    */
  def batchNormalization[T: TF : IsDecimal](
      x: Output[T],
      mean: Output[T],
      variance: Output[T],
      offset: Option[Output[T]] = None,
      scale: Option[Output[T]] = None,
      epsilon: Output[T],
      name: String = "BatchNormalization"
  ): Output[T] = {
    Op.nameScope(name) {
      val inv = Math.rsqrt(variance + epsilon)
      val scaledInv = scale.map(inv * _).getOrElse(inv)
      x * scaledInv + offset.map(_ - mean * scaledInv).getOrElse(-mean * scaledInv)
    }
  }

  /** $OpDocNNFusedBatchNormalization
    *
    * @param  x          Input tensor with 4 dimensions.
    * @param  scale      Vector used for scaling.
    * @param  offset     Vector used as an added offset.
    * @param  mean       Optional population mean vector, used for inference only.
    * @param  variance   Optional population variance vector, used for inference only.
    * @param  epsilon    Small floating point number added to the variance to avoid division by zero.
    * @param  dataFormat Data format for `x`.
    * @param  isTraining Boolean value indicating whether the operation is used for training or inference.
    * @param  name       Name for the created ops.
    * @return Batch normalized tensor `x`, along with the a batch mean vector, and a batch variance vector.
    * @throws IllegalArgumentException If `isTraining == false` and `mean` and `variance` are both `None`.
    */
  @throws[IllegalArgumentException]
  def fusedBatchNormalization[T: TF : IsDecimal](
      x: Output[T],
      scale: Output[Float],
      offset: Output[Float],
      mean: Option[Output[Float]] = None,
      variance: Option[Output[Float]] = None,
      epsilon: Float = 0.0001f,
      dataFormat: CNNDataFormat = NWCFormat,
      isTraining: Boolean = true,
      name: String = "FusedBatchNormalization"
  ): (Output[T], Output[Float], Output[Float], Output[Float], Output[Float]) = {
    require(
      !isTraining || (mean.isEmpty && variance.isEmpty),
      "Both `mean` and `variance` must be `None` if `isTraining == true`.")
    // Set a minimum epsilon to 1.001e-5f, which is a requirement by CuDNN to prevent an exception.
    val minEpsilon = 1.001e-5f
    Op.Builder[(Output[T], Output[Float], Output[Float], Output[Float], Output[Float]), (Output[T], Output[Float], Output[Float], Output[Float], Output[Float])](
      opType = "FusedBatchNormV2",
      name = name,
      input = (x,
          scale,
          offset,
          mean.getOrElse(Basic.zeros[Float](Shape(0))),
          variance.getOrElse(Basic.zeros[Float](Shape(0))))
    ).setAttribute("epsilon", if (epsilon > minEpsilon) epsilon else minEpsilon)
        .setAttribute("data_format", dataFormat.name)
        .setAttribute("is_training", isTraining)
        .setGradientFn(fusedBatchNormalizationGradient(_, _)(TF[T], IsDecimal[T]))
        .build().output
  }

  protected def fusedBatchNormalizationGradient[T: TF : IsDecimal](
      op: Op[(Output[T], Output[Float], Output[Float], Output[Float], Output[Float]), (Output[T], Output[Float], Output[Float], Output[Float], Output[Float])],
      outputGradient: (Output[T], Output[Float], Output[Float], Output[Float], Output[Float])
  ): (Output[T], Output[Float], Output[Float], Output[Float], Output[Float]) = {
    var x = op.input._1
    var gradY = outputGradient._1
    val scale = op.input._2
    val epsilon = op.floatAttribute("epsilon")
    val dataFormat = CNNDataFormat.fromName(op.stringAttribute("data_format"))
    val isTraining = op.booleanAttribute("is_training")

    if (!isTraining && dataFormat == NCWFormat) {
      x = Basic.transpose(x, Seq(0, 2, 3, 1))
      gradY = Basic.transpose(gradY, Seq(0, 2, 3, 1))
    }

    val (popMean, popVariance) = {
      if (isTraining)
        (op.output._4, op.output._5)
      else
        (op.input._4, op.input._5)
    }

    val gradients = Op.Builder[(Output[T], Output[T], Output[Float], Output[Float], Output[Float]), (Output[T], Output[Float], Output[Float], Output[Float], Output[Float])](
      opType = "FusedBatchNormGradV2",
      name = "FusedBatchNormalizationGradient",
      input = (gradY, x, scale, popMean, popVariance)
    ).setAttribute("epsilon", epsilon)
        .setAttribute("data_format", if (isTraining) dataFormat.name else NWCFormat.name)
        .setAttribute("is_training", isTraining)
        .build().output

    if (isTraining) {
      (gradients._1, gradients._2, gradients._3, null, null)
    } else {
      val dx = {
        if (dataFormat == NCWFormat)
          Basic.transpose(gradients._1, Seq(0, 3, 1, 2))
        else
          gradients._1
      }
      (dx, gradients._2, gradients._3, null, null)
    }
  }

  //endregion Normalization Ops
}

object NN extends NN {
  /** Padding mode. */
  sealed trait ConvPaddingMode {
    val name: String

    override def toString: String = name
  }

  object ConvPaddingMode {
    def fromName(name: String): ConvPaddingMode = fromString(name)

    @throws[InvalidArgumentException]
    def fromString(name: String): ConvPaddingMode = name match {
      case SameConvPadding.name => SameConvPadding
      case ValidConvPadding.name => ValidConvPadding
      case _ => throw InvalidArgumentException(
        s"Invalid convolution/pooling padding mode '$name' provided.")
    }
  }

  case object SameConvPadding extends NN.ConvPaddingMode { override val name: String = "SAME" }
  case object ValidConvPadding extends NN.ConvPaddingMode { override val name: String = "VALID" }

  sealed trait CNNDataFormat {
    val name: String
    override def toString: String = name
  }

  object CNNDataFormat {
    val default: CNNDataFormat = NWCFormat

    def fromName(name: String): CNNDataFormat = fromString(name)

    @throws[InvalidArgumentException]
    def fromString(name: String): CNNDataFormat = name match {
      case NWCFormat.name => NWCFormat
      case NCWFormat.name => NCWFormat
      case _ => throw InvalidArgumentException(s"Invalid convolution/pooling data format '$name' provided.")
    }
  }

  case object NWCFormat extends CNNDataFormat { override val name: String = "NHWC" }
  case object NCWFormat extends CNNDataFormat { override val name: String = "NCHW" }

  /** Creates an op that flattens the outer axes of `input` and keeps its last axis. */
  private[ops] def flattenOuterAxes[T: TF](
      input: Output[T]
  ): Output[T] = {
    val rank = Basic.rank(input)
    val lastAxisSize = Basic.slice[Long, Int](
      Basic.shape(input),
      Basic.expandDims(Math.subtract[Int](rank, 1), -1),
      Basic.ones[Int](Shape(1)))
    val output = Basic.reshape(input, Basic.concatenate(
      Seq(Basic.constant(-1L, Shape(1)), lastAxisSize),
      axis = 0))
    // Set the output shape, if known.
    val shape = input.shape
    if (shape.rank != -1 && !shape.asArray.contains(-1))
      output.setShape(Shape(shape(0 :: -1).asArray.product, shape(-1)))
    output
  }

  /** Creates an op that swaps the axes `axis1` and `axis2` in `input` and ignores all axes after `axis2`. */
  private[ops] def swapAxes[T: TF, I: TF : IsIntOrLong](
      input: Output[T],
      axis1: Output[I],
      axis2: Output[I],
      name: String = "SwapAxes"
  ): Output[T] = {
    val zero = Basic.zeros[I](Shape())
    val one = Basic.ones[I](Shape())
    Basic.transpose(
      input,
      Basic.concatenate(Seq(
        Math.range(zero, axis1),
        axis2,
        Math.range(axis1 + one, axis2),
        axis1), axis = 0),
      conjugate = false,
      name = name)
  }

  /** Creates an op that moves `axis` to the end. */
  private[ops] def moveAxisToEnd[T: TF](
      input: Output[T],
      axis: Int,
      rank: Output[Int],
      name: String = "SwapAxes"
  ): Output[T] = {
    if (axis == -1) {
      input
    } else {
      val axisOutput = Basic.constant(axis)
      Basic.transpose(
        input,
        Basic.concatenate(Seq(
          Math.range(0, axisOutput),
          Math.range(axisOutput + 1, rank),
          axisOutput), axis = 0),
        conjugate = false,
        name = name)
    }
  }

  /** @define OpDocNNAddBias
    *   The `addBias` op adds `bias` to `value`.
    *
    *   The op is (mostly) a special case of `add` where `bias` is restricted to be one-dimensional (i.e., has rank
    *   1). Broadcasting is supported and so `value` may have any number of dimensions. Unlike `add`, the type of
    *   `bias`is allowed to differ from that of value `value` in the case where both types are quantized.
    *
    * @define OpDocNNLinear
    *   The `linear` op computes `x * weights + bias`.
    *
    * @define OpDocNNL2Normalize
    *   The `l2Normalize` op normalizes along axes `axes` using an L2 norm.
    *
    *   For a 1-D tensor with `axes = 0`, the op computes:
    *   `output = x / sqrt(max(sum(x^2), epsilon))`
    *
    *   For higher-dimensional `x`, the op independently normalizes each 1-D slice along axes `axes`.
    *
    * @define OpDocNNRelu
    *   The `relu` op computes the rectified linear unit activation function.
    *
    *   The rectified linear unit activation function is defined as `relu(x) = max(x, 0)`.
    *
    * @define OpDocNNRelu6
    *   The `relu6` op computes the rectified linear unit 6 activation function.
    *
    *   The rectified linear unit 6 activation function is defined as `relu6(x) = min(max(x, 0), 6)`.
    *
    *   Source: [Convolutional Deep Belief Networks on CIFAR-10. A. Krizhevsky](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf)
    *
    * @define OpDocNNCrelu
    *   The `crelu` op computes the concatenated rectified linear unit activation function.
    *
    *   The op concatenates a ReLU which selects only the positive part of the activation with a ReLU which selects only
    *   the *negative* part of the activation. Note that as a result this non-linearity doubles the depth of the
    *   activations.
    *
    *   Source: [Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units](https://arxiv.org/abs/1603.05201)
    *
    * @define OpDocNNElu
    *   The `elu` op computes the exponential linear unit activation function.
    *
    *   The exponential linear unit activation function is defined as `elu(x) = x`, if `x > 0`, and
    *   `elu(x) = exp(x) - 1`, otherwise.
    *
    *   Source: [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.07289)
    *
    * @define OpDocNNSelu
    *   The `selu` op computes the scaled exponential linear unit activation function.
    *
    *   The scaled exponential linear unit activation function is defined as `selu(x) = scale * x`, if `x > 0`, and
    *   `elu(x) = scale * alpha * (exp(x) - 1)`, otherwise, where `scale = 1.0507` and `alpha = 1.7581`.
    *
    *   Source: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    *
    * @define OpDocNNSoftplus
    *   The `softplus` op computes the softplus activation function.
    *
    *   The softplus activation function is defined as `softplus(x) = log(exp(x) + 1)`.
    *
    * @define OpDocNNSoftsign
    *   The `softsign` op computes the softsign activation function.
    *
    *   The softsign activation function is defined as `softsign(x) = x / (abs(x) + 1)`.
    *
    * @define OpDocNNSoftmax
    *   The `softmax` op computes softmax activations.
    *
    *   For each batch `i` and class `j` we have `softmax = exp(logits) / sum(exp(logits), axis)`, where `axis`
    *   indicates the axis the softmax should be performed on.
    *
    * @define OpDocNNLogSoftmax
    *   The `logSoftmax` op computes log-softmax activations.
    *
    *   For each batch `i` and class `j` we have `log_softmax = logits - log(sum(exp(logits), axis))`, where `axis`
    *   indicates the axis the log-softmax should be performed on.
    *
    * @define OpDocNNL2Loss
    *   The `l2Loss` op computes half of the L2 norm of a tensor without the square root.
    *
    *   The output is equal to `sum(input^2) / 2`.
    *
    * @define OpDocNNSoftmaxCrossEntropy
    *   The `softmaxCrossEntropy` op computes the softmax cross entropy between `logits` and `labels`.
    *
    *   The op measures the probabilistic error in discrete classification tasks in which the classes are mutually
    *   exclusive (each entry belongs to exactly one class). For example, each CIFAR-10 image is labeled with one and
    *   only one label: an image can be a dog or a truck, but not both.
    *
    *   Back-propagation will happen into both `logits` and `labels`. To disallow back-propagation into `labels`, pass
    *   the label tensors through a `stopGradients` op before feeding it to this function.
    *
    *   '''NOTE:''' While the classes are mutually exclusive, their probabilities need not be. All that is required is
    *   that each row of `labels` is a valid probability distribution. If they are not, the computation of the gradient
    *   will be incorrect. If using exclusive `labels` (wherein one and only one class is true at a time), see
    *   [[sparseSoftmaxCrossEntropy]].
    *
    *   '''WARNING:''' The op expects unscaled logits, since it performs a `softmax` on `logits` internally for
    *   efficiency. Do not call this op with the output of `softmax`, as it will produce incorrect results.
    *
    *   `logits` and `labels` must have the same shape. A common use case if to have `logits` and `labels` of shape
    *   `[batchSize, numClasses]`, but higher dimensions are also supported.
    *
    *   `logits` and `labels` must have data type [[FLOAT16]], [[FLOAT32]], or [[FLOAT64]].
    *
    * @define OpDocNNSparseSoftmaxCrossEntropy
    *   The `sparseSoftmaxCrossEntropy` op computes the sparse softmax cross entropy between `logits` and `labels`.
    *
    *   The op measures the probabilistic error in discrete classification tasks in which the classes are mutually
    *   exclusive (each entry belongs to exactly one class). For example, each CIFAR-10 image is labeled with one and
    *   only one label: an image can be a dog or a truck, but not both.
    *
    *   '''NOTE:''' For the op, the probability of a given label is considered exclusive. That is, soft classes are not
    *   allowed, and the `labels` vector must provide a single specific index for the true class for each row of
    *   `logits` (i.e., each batch instance). For soft softmax classification with a probability distribution for each
    *   entry, see [[softmaxCrossEntropy]].
    *
    *   '''WARNING:''' The op expects unscaled logits, since it performs a `softmax` on `logits` internally for
    *   efficiency. Do not call this op with the output of `softmax`, as it will produce incorrect results.
    *
    *   A common use case if to have `logits` of shape `[batchSize, numClasses]` and `labels` of shape `[batchSize]`,
    *   but higher dimensions are also supported.
    *
    * @define OpDocNNSigmoidCrossEntropy
    *   The `sigmoidCrossEntropy` op computes the sigmoid cross entropy between `logits` and `labels`.
    *
    *   The op measures the probability error in discrete classification tasks in which each class is independent and
    *   not mutually exclusive. For instance, one could perform multi-label classification where a picture can contain
    *   both an elephant and a dog at the same time.
    *
    *   For brevity, let `x = logits` and `z = labels`. The sigmoid cross entropy (also known as logistic loss) is
    *   defined as:
    *   `  z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))`
    *   `= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))`
    *   `= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))`
    *   `= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))`
    *   `= (1 - z) * x + log(1 + exp(-x))`
    *   `= x - x * z + log(1 + exp(-x))`
    *
    *   For `x < 0`, to avoid numerical overflow in `exp(-x)`, we reformulate the above to:
    *   `  x - x * z + log(1 + exp(-x))`
    *   `= log(exp(x)) - x * z + log(1 + exp(-x))`
    *   `= - x * z + log(1 + exp(x))`
    *
    *   Hence, to ensure stability and avoid numerical overflow, the implementation uses this equivalent formulation:
    *   `max(x, 0) - x * z + log(1 + exp(-abs(x)))`
    *
    *   If `weights` is not `null`, then the positive examples are weighted. A value `weights > 1` decreases the false
    *   negative count, hence increasing recall. Conversely setting `weights < 1` decreases the false positive count and
    *   increases precision. This can be seen from the fact that `weight` is introduced as a multiplicative coefficient
    *   for the positive targets term in the loss expression (where `q = weights`, for brevity):
    *   `  qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))`
    *   `= qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))`
    *   `= qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))`
    *   `= qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))`
    *   `= (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))`
    *   `= (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))`
    *
    *   Setting `l = 1 + (q - 1) * z`, to ensure stability and avoid numerical overflow, the implementation uses this
    *   equivalent formulation:
    *   `(1 - z) * x + l * (max(-x, 0) + log(1 + exp(-abs(x))))`
    *
    *   `logits` and `labels` must have the same shape.
    *
    * @define OpDocNNLogPoissonLoss
    *   The `logPoissonLoss` op computes the log-Poisson loss between `logPredictions` and `targets`.
    *
    *   The op computes the log-likelihood loss between the predictions and the targets under the assumption that the
    *   targets have a Poisson distribution. **Caveat:** By default, this is not the exact loss, but the loss minus a
    *   constant term (`log(z!)`). That has no effect for optimization purposes, but it does not play well with relative
    *   loss comparisons. To compute an approximation of the log factorial term, please set `computeFullLoss` to `true`,
    *   to enable Stirling's Approximation.
    *
    *   For brevity, let `c = log(x) = logPredictions`, `z = targets`.  The log-Poisson loss is defined as:
    *   `  -log(exp(-x) * (x^z) / z!)`
    *   `= -log(exp(-x) * (x^z)) + log(z!)`
    *   `~ -log(exp(-x)) - log(x^z) [z * log(z) - z + 0.5 * log(2 * pi * z)]` (Note that the second term is Stirling's
    *                                                                          Approximation for `log(z!)`. It is
    *                                                                          invariant to `x` and does not affect
    *                                                                          optimization, though it is important for
    *                                                                          correct relative loss comparisons. It is
    *                                                                          only computed when
    *                                                                          `computeFullLoss == true`)
    *   `= x - z * log(x) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]`
    *   `= exp(c) - z * c [+ z * log(z) - z + 0.5 * log(2 * pi * z)]`.
    *
    * @define OpDocNNSequenceLoss
    *   The `sequenceLoss` op computes an optionally weighted loss for a sequence of predicted logits.
    *
    *   Depending on the values of `averageAcrossTimeSteps` and `averageAcrossBatch`, the returned tensor will have rank
    *   0, 1, or 2 as these arguments reduce the cross-entropy each label, which has shape
    *   `[batchSize, sequenceLength]`, over their respective dimensions. For examplem if `averageAcrossTimeSteps` is
    *   `true` and `averageAcrossBatch` is `false`, then the returned tensor will have shape `[batchSize]`.
    *
    * @define OpDocNNLocalResponseNormalization
    *   The `localResponseNormalization` op treats the input 4-D tensor as a 3-D array of 1-D vectors (along the last
    *   dimension), and each vector is normalized independently. Within a given vector, each component is divided by the
    *   weighted, squared sum of the inputs within `depthRadius`. In detail:
    *
    *   {{{
    *    sqrSum[a, b, c, d] = sum(input[a, b, c, d - depthRadius : d + depthRadius + 1] **   2)
    *    output = input / (bias + alpha *   sqrSum) **   beta
    *   }}}
    *
    *   For details, see
    *   [[http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks (NIPS 2012).]]
    *
    * @define OpDocNNDropout
    *   The `dropout` op computes a dropout layer.
    *
    *   With probability `keepProbability`, the op outputs the input element scaled up by `1 / keepProbability`,
    *   otherwise it outputs `0`. The scaling is such that the expected sum remains unchanged.
    *
    *   By default, each element is kept or dropped independently. If `noiseShape` is specified, it must be
    *   [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) to the shape of `input`, and only
    *   dimensions with `noiseShape(i) == x.shape(i)` will make independent decisions. For example, if
    *   `x.shape = [k, l, m, n]` and `noiseShape = [k, 1, 1, n]`, each `k` and `n` component will be kept independently
    *   and each `l` and `m` component will be kept or not kept together.
    *
    * @define OpDocNNTopK
    *   The `topK` op finds values and indices of the `k` largest entries for the last dimension of `input`.
    *
    *   If `input` is a vector (i.e., rank-1 tensor), the op finds the `k` largest entries in the vector and outputs
    *   their values and their indices as vectors. Thus, `values(j)` will be the `j`-th largest entry in `input`, and
    *   `indices(j)` will be its index.
    *
    *   For matrices (and respectively, higher rank input tensors), the op computes the top `k` entries in each row
    *   (i.e., vector along the last dimension of the tensor). Thus,
    *   `values.shape = indices.shape = input.shape(0 :: -1) + k`.
    *
    *   If two elements are equal, the lower-index element appears first.
    *
    * @define OpDocNNInTopK
    *   The `inTopK` op checks whether the `targets` are in the top `K` `predictions`.
    *
    *   The op outputs a boolean tensor with shape `[batchSize]`, with entry `output(i)` being `true` if the target
    *   class is among the top `k` predictions, among all predictions for example `i`. Note that the behavior of
    *   [[inTopK]] differs from [[topK]] in its handling of ties; if multiple classes have the same prediction value and
    *   straddle the top-`k` boundary, then all of those classes are considered to be in the top `k`.
    *
    *   More formally, let:
    *     - `predictions(i, ::)` be the predictions for all classes for example `i`,
    *     - `targets(i)` be the target class for example `i`, and
    *     - `output(i)` be the output for example `i`.
    *   Then `output(i) = predictions(i, targets(i)) \in TopKIncludingTies(predictions(i))`.
    *
    * @define OpDocNNConv2D
    *   The `conv2D` op computes a 2-D convolution given 4-D `input` and `filter` tensors.
    *
    *   Given an input tensor of shape `[batch, inHeight, inWidth, inChannels]` and a filter / kernel tensor of shape
    *   `[filterHeight, filterWidth, inChannels, outChannels]`, the op performs the following:
    *
    *     1. Flattens the filter to a 2-D matrix with shape `[filterHeight * filterWidth * inChannels, outputChannels]`.
    *     2. Extracts image patches from the input tensor to form a *virtual* tensor of shape
    *        `[batch, outHeight, outWidth, filterHeight * filterWidth * inChannels]`.
    *     3. For each patch, right-multiplies the filter matrix and the image patch vector.
    *
    *   For example, for the default [[NWCFormat]]:
    *   {{{
    *     output(b,i,j,k) = sum_{di,dj,q} input(b, stride1 * i + di, stride2 * j + dj, q) * filter(di,dj,q,k).
    *   }}}
    *
    *   Must have `strides[0] = strides[3] = 1`.  For the most common case of the same horizontal and vertices strides,
    *   `strides = [1, stride, stride, 1]`.
    *
    * @define OpDocNNConv2DBackpropInput
    *   The `conv2DBackpropInput` op computes the gradient of the `conv2D` op with respect to its input tensor.
    *
    * @define OpDocNNConv2DBackpropFilter
    *   The `conv2DBackpropFilter` op computes the gradient of the `conv2D` op with respect to its filter tensor.
    *
    * @define OpDocNNMaxPool
    *   The `maxPool` op performs max pooling on the input tensor.
    *
    * @define OpDocNNMaxPoolGrad
    *   The `maxPoolGrad` op computes the gradient of the `maxPool` op.
    *
    * @define OpDocNNMaxPoolGradGrad
    *   The `maxPoolGradGrad` op computes the gradient of the `maxPoolGrad` op.
    *
    * @define OpDocNNBatchNormalization
    *   The `batchNormalization` op applies batch normalization to input `x`, as described in
    *   [[http://arxiv.org/abs/1502.03167]].
    *
    *   The op normalizes a tensor by `mean` and `variance`, and optionally applies a `scale` and `offset` to it
    *   `beta + scale * (x - mean) / variance`. `mean`, `variance`, `offset` and `scale` are all expected to be of one
    *   of two shapes:
    *
    *     - In all generality, they can have the same number of dimensions as the input `x`, with identical sizes as `x`
    *       for the dimensions that are not normalized over the "depth" dimension(s), and size 1 for the others, which
    *       are being normalized over. `mean` and `variance` in this case would typically be the outputs of
    *       `tf.moments(..., keepDims = true)` during training, or running averages thereof during inference.
    *     - In the common case where the "depth" dimension is the last dimension in the input tensor `x`, they may be
    *       one-dimensional tensors of the same size as the "depth" dimension. This is the case, for example, for the
    *       common `[batch, depth]` layout of fully-connected layers, and `[batch, height, width, depth]` for
    *       convolutions. `mean` and `variance` in this case would typically be the outputs of
    *       `tf.moments(..., keepDims = false)` during training, or running averages thereof during inference.
    *
    * @define OpDocNNFusedBatchNormalization
    *   The `fusedBatchNormalization` applies batch normalization to input `x`, as described in
    *   [[http://arxiv.org/abs/1502.03167]].
    */
  private[ops] trait Documentation
}
