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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Indexer.Implicits._
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.ops.NN._

import scala.language.postfixOps

/** Contains functions for constructing ops related to neural networks.
  *
  * @author Emmanouil Antonios Platanios
  */
trait NN {
  /** Creates an op that adds `bias` to `value`.
    *
    * The op is (mostly) a special case of `tf.add` where `bias` is restricted to be one-dimensional (i.e., has rank 1).
    * Broadcasting is supported and so `value` may have any number of dimensions. Unlike `tf.add`, the type of `bias` is
    * allowed to differ from that of value `value` in the case where both types are quantized.
    *
    * @param  value         Value tensor.
    * @param  bias          Bias tensor that must be one-dimensional (i.e., it must have rank 1).
    * @param  cNNDataFormat Data format of the input and output tensors. With the default format [[NHWCFormat]], the
    *                       `bias` tensor will be added to the last dimension of the `value` tensor. Alternatively, the
    *                       format could be [[NCHWFormat]], and the `bias` tensor would be added to the third-to-last
    *                       dimension.
    * @param  name          Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If the `bias` tensor is not one-dimensional.
    */
  @throws[IllegalArgumentException]
  def addBias(
      value: Output, bias: Output, cNNDataFormat: CNNDataFormat = CNNDataFormat.default,
      name: String = "AddBias"): Output = {
    if (bias.rank != 1)
      throw new IllegalArgumentException(s"'bias' (rank = ${bias.rank}) should have rank 1.")
    Op.Builder(opType = "BiasAdd", name = name)
        .addInput(value)
        .addInput(bias)
        .setAttribute("data_format", cNNDataFormat.toString)
        .build().outputs(0)
  }
}

object NN extends NN {
  sealed trait CNNDataFormat {
    val name: String
    override def toString: String = name
  }
  case object NHWCFormat extends CNNDataFormat {override val name: String = "NHWC"}
  case object NCHWFormat extends CNNDataFormat {override val name: String = "NCHW"}

  object CNNDataFormat {
    val default = NHWCFormat

    def fromName(name: String): CNNDataFormat = fromString(name)

    @throws[IllegalArgumentException]
    def fromString(name: String): CNNDataFormat = name match {
      case "NHWC" => NHWCFormat
      case "NCHW" => NCHWFormat
      case _ => throw new IllegalArgumentException(s"Unsupported CNN data format string: '$name'.")
    }
  }

  private[api] object Gradients {
    GradientsRegistry.register("BiasAdd", biasAddGradient)
    GradientsRegistry.register("BiasAddGrad", biasAddHessian)

    private[this] def biasAddGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      val cNNDataFormatName = {
        try {
          op.stringAttribute("data_format")
        } catch {
          case _: Throwable => CNNDataFormat.default.toString
        }
      }
      val gradient = Op.Builder(opType = "BiasAddGrad", name = "BiasAddGradient")
          .addInput(outputGradient)
          .setAttribute("data_format", cNNDataFormatName)
          .build().outputs(0)
      outputGradients :+ gradient
    }

    private[this] def biasAddHessian(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      val cNNDataFormatName = {
        try {
          op.stringAttribute("data_format")
        } catch {
          case _: Throwable => CNNDataFormat.default.toString
        }
      }
      val valueShape = Basic.shape(op.inputs(0))
      val biasShape = Basic.shape(outputGradient)
      val (expandedShape, tileMultiples) = cNNDataFormatName match {
        case "NHWC" =>
          val valuesLeft = valueShape(0 :: -1)
          val expandedShape = Basic.concatenate(Seq(Basic.onesLike(valuesLeft), biasShape), axis = 0)
          val tileMultiples = Basic.concatenate(Seq(valuesLeft, 1), 0)
          (expandedShape, tileMultiples)
        case "NCHW" =>
          val valuesLeft = valueShape(0 :: -3)
          val valuesRight = valueShape(-2 ::)
          val expandedShape = Basic.concatenate(
            Seq(Basic.onesLike(valuesLeft), biasShape, Basic.onesLike(valuesRight)), axis = 0)
          val tileMultiples = Basic.concatenate(Seq(valuesLeft, 1, valuesRight), 0)
          (expandedShape, tileMultiples)
      }
      Seq(Basic.tile(Basic.reshape(outputGradient, expandedShape), tileMultiples))
    }
  }
}
