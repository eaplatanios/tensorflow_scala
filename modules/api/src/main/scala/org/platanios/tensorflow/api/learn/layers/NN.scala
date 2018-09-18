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
import org.platanios.tensorflow.api.learn.{Mode, TRAINING, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.NN._
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
object NN {
  private[layers] trait API {
    type BatchNormalization = layers.BatchNormalization
    type Softmax = layers.Softmax
    type LogSoftmax = layers.LogSoftmax
    type LRN = layers.LRN
    type Dropout = layers.Dropout
    type Conv2D = layers.Conv2D
    type MaxPool = layers.MaxPool

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

case class BatchNormalization(
    override val name: String,
    axis: Int = -1,
    momentum: Float = 0.9f,
    epsilon: Float = 1e-3f,
    center: Boolean = true,
    scale: Boolean = true,
    betaInitializer: Initializer = ZerosInitializer,
    gammaInitializer: Initializer = OnesInitializer,
    movingMeanInitializer: Initializer = ZerosInitializer,
    movingVarianceInitializer: Initializer = OnesInitializer,
    betaRegularizer: Regularizer = null,
    gammaRegularizer: Regularizer = null,
    renorm: Boolean = false, // TODO: [LAYERS] Renorm clipping
    renormMomentum: Float = 0.9f,
    fused: Boolean = true,
    dataType: DataType[_] = FLOAT32
) extends Layer[Output, Output](name) {
  override val layerType: String = "BatchNormalization"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
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

    // Raise the parameters of FLOAT16 batch norm to FLOAT32.
    val parametersDataType = {
      if (dataType == FLOAT16 || dataType == BFLOAT16)
        FLOAT32
      else
        dataType
    }

    val parametersShape = Shape(input.shape(computedAxis))

    val beta = {
      if (center)
        getParameter(
          name = "Beta", parametersDataType, parametersShape,
          initializer = betaInitializer, regularizer = betaRegularizer)
      else
        tf.constant(0.0f, parametersDataType, parametersShape)
    }

    val gamma = {
      if (scale)
        getParameter(
          name = "Gamma", parametersDataType, parametersShape,
          initializer = gammaInitializer, regularizer = gammaRegularizer)
      else
        tf.constant(1.0f, parametersDataType, parametersShape)
    }

    val movingMean = tf.variable(
      name = "MovingMean", parametersDataType, parametersShape,
      initializer = movingMeanInitializer, trainable = false)

    val movingVariance = tf.variable(
      name = "MovingVariance", parametersDataType, parametersShape,
      initializer = movingVarianceInitializer, trainable = false)

    val renormVariables = Option(renorm).collect {
      case true =>
        val renormMean = tf.variable(
          name = "RenormMean", parametersDataType, parametersShape,
          initializer = ZerosInitializer, trainable = false)
        val renormMeanWeight = tf.variable(
          name = "RenormMeanWeight", parametersDataType, Shape(),
          initializer = ZerosInitializer, trainable = false)
        val renormStdDev = tf.variable(
          name = "RenormStdDev", parametersDataType, parametersShape,
          initializer = ZerosInitializer, trainable = false)
        val renormStdDevWeight = tf.variable(
          name = "RenormStdDevWeight", parametersDataType, Shape(),
          initializer = ZerosInitializer, trainable = false)
    }

    if (useFused) {
      val outputs = mode match {
        case TRAINING => fusedBatchNormalization(
          input, gamma, beta, epsilon = epsilon, dataFormat = dataFormat)
        case _ => fusedBatchNormalization(
          input, gamma, beta, mean = Some(movingMean), variance = Some(movingVariance),
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

      def broadcast(v: Output): Output = {
        if (v.rank != inputRank && (reductionAxes != (0 until inputRank - 1)))
          tf.reshape(v, broadcastShape)
        else
          v
      }

      val offsetValue = if (center) Some(broadcast(beta).cast(input.dataType)) else None
      val scaleValue = if (scale) Some(broadcast(gamma).cast(input.dataType)) else None

      val (mean, variance) = mode match {
        case TRAINING =>
          val (mean, variance) = tf.moments(input, reductionAxes, keepDims = false)
          if (renorm) {
            ??? // TODO: [LAYERS] Batch renorm.
          }
          val meanUpdate = assignMovingAverage(movingMean, mean, momentum)
          val varianceUpdate = assignMovingAverage(movingVariance, variance, momentum)
          val meanCast = tf.createWith(controlDependencies = Set(meanUpdate.op))(mean.cast(input.dataType))
          val varianceCast = tf.createWith(controlDependencies = Set(varianceUpdate.op))(variance.cast(input.dataType))
          (meanCast, varianceCast)
        case _ =>
          (movingMean.value.cast(input.dataType), movingVariance.value.cast(input.dataType))
      }

      val output = tf.batchNormalization(
        input, broadcast(mean), broadcast(variance), offsetValue, scaleValue, epsilon = epsilon)

      // If some components of the shape got lost due to adjustments, fix that, and return the result.
      output.setShape(input.shape)
      output
    }
  }

  protected def assignMovingAverage(variable: Variable, value: Output, momentum: Output): Output = {
    Op.createWith(nameScope = s"${variable.name}/AssignMovingAverage") {
      Op.colocateWith(Set(variable.op), ignoreExisting = true) {
        val decay = (1.0f - momentum).cast(variable.dataType)
        val updateDelta = (variable.value - value.cast(variable.dataType)) * decay
        variable.assignSub(updateDelta)
      }
    }
  }
}

case class Softmax(override val name: String)
    extends Layer[Output, Output](name) {
  override val layerType: String = "Softmax"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.softmax(input, name = name)
  }
}

case class LogSoftmax(override val name: String)
    extends Layer[Output, Output](name) {
  override val layerType: String = "LogSoftmax"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.logSoftmax(input, name = name)
  }
}

case class LRN(
    override val name: String,
    depthRadius: Int = 5,
    bias: Float = 1.0f,
    alpha: Float = 1.0f,
    beta: Float = 0.5f
) extends Layer[Output, Output](name) {
  override val layerType: String = "LRN"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.lrn(input, depthRadius, bias, alpha, beta, name = name)
  }
}

case class Dropout(
    override val name: String,
    keepProbability: Float,
    scaleOutput: Boolean = true,
    noiseShape: Shape = null,
    seed: Option[Int] = None
) extends Layer[Output, Output](name) {
  override val layerType: String = s"Dropout[$keepProbability]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    mode match {
      case TRAINING =>
        val noise = if (noiseShape == null) null else noiseShape.toOutput()
        ops.NN.dropout(input, keepProbability, scaleOutput, noise, seed, name)
      case _ => input
    }
  }
}

case class Conv2D(
    override val name: String,
    filterShape: Shape,
    stride1: Long,
    stride2: Long,
    padding: ConvPaddingMode,
    dataFormat: CNNDataFormat = CNNDataFormat.default,
    dilations: (Int, Int, Int, Int) = (1, 1, 1, 1),
    useCuDNNOnGPU: Boolean = true,
    weightsInitializer: Initializer = RandomNormalInitializer()
) extends Layer[Output, Output](name) {
  override val layerType: String = s"Conv2D[${filterShape.asArray.mkString(",")}]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    val weights = getParameter("Weights", input.dataType, filterShape, weightsInitializer)
    ops.NN.conv2D(input, weights, stride1, stride2, padding, dataFormat, dilations, useCuDNNOnGPU)
  }
}

case class MaxPool(
    override val name: String,
    windowSize: Seq[Long],
    stride1: Long,
    stride2: Long,
    padding: ConvPaddingMode,
    dataFormat: CNNDataFormat = CNNDataFormat.default
) extends Layer[Output, Output](name) {
  override val layerType: String = s"MaxPool[${windowSize.mkString(",")}]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.maxPool(input, windowSize, stride1, stride2, padding, dataFormat, name)
  }
}
