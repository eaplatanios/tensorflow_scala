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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.learn.{Mode, TRAINING, layers}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.NN._
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
object NN {
  private[layers] trait API {
    type BatchNormalization[T] = layers.BatchNormalization[T]
    type Softmax[T] = layers.Softmax[T]
    type LogSoftmax[T] = layers.LogSoftmax[T]
    type LRN[T] = layers.LRN[T]
    type Dropout[T] = layers.Dropout[T]
    type Conv2D[T] = layers.Conv2D[T]
    type MaxPool[T] = layers.MaxPool[T]

    val BatchNormalization: layers.BatchNormalization.type = layers.BatchNormalization
    val Softmax           : layers.Softmax.type            = layers.Softmax
    val LogSoftmax        : layers.LogSoftmax.type         = layers.LogSoftmax
    val LRN               : layers.LRN.type                = layers.LRN
    val Dropout           : layers.Dropout.type            = layers.Dropout
    val Conv2D            : layers.Conv2D.type             = layers.Conv2D
    val MaxPool           : layers.MaxPool.type            = layers.MaxPool
  }

  object API extends API
}

case class BatchNormalization[T: TF : IsDecimal](
    override val name: String,
    axis: Int = -1,
    momentum: Float = 0.9f,
    epsilon: Float = 1e-3f,
    center: Boolean = true,
    scale: Boolean = true,
    betaInitializer: tf.VariableInitializer = tf.ZerosInitializer,
    gammaInitializer: tf.VariableInitializer = tf.OnesInitializer,
    movingMeanInitializer: tf.VariableInitializer = tf.ZerosInitializer,
    movingVarianceInitializer: tf.VariableInitializer = tf.OnesInitializer,
    betaRegularizer: tf.VariableRegularizer = null,
    gammaRegularizer: tf.VariableRegularizer = null,
    renorm: Boolean = false, // TODO: [LAYERS] Renorm clipping
    renormMomentum: Float = 0.9f,
    fused: Boolean = true
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = "BatchNormalization"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    val inputRank = input.rank
    val computedAxis = if (axis < 0) inputRank + axis else axis

    // Currently fused batch norm does not support renormalization. It also only supports input tensors with rank 4,
    // and with the channel dimension being dimension 1 or 3.
    val useFused = fused && !renorm && inputRank == 4 && (axis == 1 || axis == 3)

    val dataFormat = (useFused, axis) match {
      case (true, 1) => NCWFormat
      case (true, 3) => NWCFormat
      case _ => null // This is a dummy default that should never be used.
    }

    val parametersShape = Shape(input.shape(computedAxis))

    val beta = {
      if (center)
        getParameter[Float]("Beta", parametersShape, betaInitializer, betaRegularizer)
      else
        tf.zeros[Float](parametersShape)
    }

    val gamma = {
      if (scale)
        getParameter[Float]("Gamma", parametersShape, gammaInitializer, gammaRegularizer)
      else
        tf.ones[Float](parametersShape)
    }

    val movingMean = tf.variable[Float]("MovingMean", parametersShape, movingMeanInitializer, trainable = false)
    val movingVariance = tf.variable[Float]("MovingVariance", parametersShape, movingVarianceInitializer, trainable = false)

    val renormVariables = Option(renorm).collect {
      case true =>
        val renormMean = tf.variable[Float]("RenormMean", parametersShape, tf.ZerosInitializer, trainable = false)
        val renormMeanWeight = tf.variable[Float]("RenormMeanWeight", Shape(), tf.ZerosInitializer, trainable = false)
        val renormStdDev = tf.variable[Float]("RenormStdDev", parametersShape, tf.ZerosInitializer, trainable = false)
        val renormStdDevWeight = tf.variable[Float]("RenormStdDevWeight", Shape(), tf.ZerosInitializer, trainable = false)
        ???
    }

    if (useFused) {
      val outputs = mode match {
        case TRAINING =>
          fusedBatchNormalization(
            input, gamma, beta,
            epsilon = epsilon, dataFormat = dataFormat)
        case _ =>
          fusedBatchNormalization(
            input, gamma, beta, Some(movingMean), Some(movingVariance),
            epsilon = epsilon, dataFormat = dataFormat)
      }

      val momentum = mode match {
        case TRAINING => this.momentum
        case _ => 1.0f
      }

      val updatedMean = assignMovingAverage(movingMean, outputs._2, momentum)
      val updatedVariable = assignMovingAverage(movingVariance, outputs._3, momentum)

      tf.createWith(controlDependencies = Set(updatedMean.op, updatedVariable.op)) {
        tf.identity(outputs._1)
      }
    } else {
      // Compute the axes along which to reduce the mean / variance.
      val reductionAxes = (0 until inputRank).filter(_ != computedAxis)

      // Broadcasting is only necessary for single-axis batch norm where the axis is not the last dimension.
      val broadcastShape = Array.fill(inputRank)(1)
      broadcastShape(computedAxis) = input.shape(computedAxis)

      def broadcast[V: TF](v: Output[V]): Output[V] = {
        if (v.rank != inputRank && (reductionAxes != (0 until inputRank - 1)))
          tf.reshape[V, Int](v, broadcastShape)
        else
          v
      }

      val offsetValue = if (center) Some(broadcast(beta).castTo[T]) else None
      val scaleValue = if (scale) Some(broadcast(gamma).castTo[T]) else None

      val (mean, variance) = mode match {
        case TRAINING =>
          val (mean, variance) = tf.moments(input, reductionAxes, keepDims = false)
          if (renorm) {
            ??? // TODO: [LAYERS] Batch renorm.
          }
          val meanUpdate = assignMovingAverage(movingMean, mean.toFloat, momentum)
          val varianceUpdate = assignMovingAverage(movingVariance, variance.toFloat, momentum)
          val meanIdentity = tf.createWith(controlDependencies = Set(meanUpdate.op))(tf.identity(mean))
          val varianceIdentity = tf.createWith(controlDependencies = Set(varianceUpdate.op))(tf.identity(variance))
          (meanIdentity, varianceIdentity)
        case _ =>
          (movingMean.value.castTo[T], movingVariance.value.castTo[T])
      }

      val epsilonCast = Tensor(epsilon).castTo[T]
      val output = tf.batchNormalization[T](
        input, broadcast(mean), broadcast(variance),
        offsetValue, scaleValue, epsilonCast)

      // If some components of the shape got lost due to adjustments, fix that, and return the result.
      output.setShape(input.shape)
      output
    }
  }

  protected def assignMovingAverage(
      variable: Variable[Float],
      value: Output[Float],
      momentum: Output[Float]
  ): Output[Float] = {
    Op.nameScope(s"${variable.name}/AssignMovingAverage") {
      Op.colocateWith(Set(variable.op), ignoreExisting = true) {
        val updateDelta = ops.Math.multiply(ops.Math.subtract(variable.value, value), ops.Math.subtract(1.0f, momentum))
        variable.assignSub(updateDelta)
      }
    }
  }
}

case class Softmax[T: TF : IsDecimal](
    override val name: String
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = "Softmax"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.softmax(input, name = name)
  }
}

case class LogSoftmax[T: TF : IsDecimal](
    override val name: String
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = "LogSoftmax"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.logSoftmax(input, name = name)
  }
}

case class LRN[T: TF : IsBFloat16OrFloat16OrFloat32](
    override val name: String,
    depthRadius: Int = 5,
    bias: Float = 1.0f,
    alpha: Float = 1.0f,
    beta: Float = 0.5f
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = "LRN"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.lrn(input, depthRadius, bias, alpha, beta, name = name)
  }
}

case class Dropout[T: TF : IsFloat16OrFloat32OrFloat64](
    override val name: String,
    keepProbability: Float,
    scaleOutput: Boolean = true,
    noiseShape: Shape = null,
    seed: Option[Int] = None
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = s"Dropout[$keepProbability]"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    mode match {
      case TRAINING =>
        val noise = if (noiseShape == null) null else noiseShape.toOutput
        ops.NN.dropout[T, Long](input, keepProbability, scaleOutput, noise, seed, name)
      case _ => input
    }
  }
}

case class Conv2D[T: TF : IsDecimal](
    override val name: String,
    filterShape: Shape,
    stride1: Long,
    stride2: Long,
    padding: ConvPaddingMode,
    dataFormat: CNNDataFormat = CNNDataFormat.default,
    dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
    useCuDNNOnGPU: Boolean = true,
    weightsInitializer: tf.VariableInitializer = tf.RandomNormalInitializer()
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = s"Conv2D[${filterShape.asArray.mkString(",")}]"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    val weights = getParameter[T]("Weights", filterShape, weightsInitializer)
    ops.NN.conv2D(input, weights, stride1, stride2, padding, dataFormat, dilations, useCuDNNOnGPU)
  }
}

case class MaxPool[T: TF : IsNumeric](
    override val name: String,
    windowSize: Seq[Int],
    stride1: Int,
    stride2: Int,
    padding: ConvPaddingMode,
    dataFormat: CNNDataFormat = CNNDataFormat.default
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = s"MaxPool[${windowSize.mkString(",")}]"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.maxPool(input, windowSize, Tensor(1, stride1, stride2, 1), padding, dataFormat, name)
  }
}
