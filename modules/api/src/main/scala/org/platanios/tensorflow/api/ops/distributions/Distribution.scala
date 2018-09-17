///* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"); you may not
// * use this file except in compliance with the License. You may obtain a copy of
// * the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// * License for the specific language governing permissions and limitations under
// * the License.
// */
//
//package org.platanios.tensorflow.api.ops.distributions
//
//import org.platanios.tensorflow.api.core.Shape
//import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
//import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, OutputLike}
//import org.platanios.tensorflow.api.types.{DataType, INT32}
//
///**
//  *
//  * @param  dataType               Data type of tensors handled by this distribution.
//  * @param  reparameterizationType Describes how samples from this distribution are reparameterized.
//  * @param  validateArguments      Boolean value indicating whether possibly expensive argument validity checks are
//  *                                enabled.
//  * @param  allowNaNStatistics     Boolean value indicating how to handle undefined statistic values. Statistics return
//  *                                +/- infinity when it makes sense. E.g., the variance of a Cauchy distribution is
//  *                                infinity. However, sometimes the statistic is undefined (e.g., if a distribution's
//  *                                PDF does not achieve a maximum within the support of the distribution, its mode is
//  *                                undefined). If the mean is undefined, then by definition the variance is undefined.
//  *                                E.g., the mean for Student's T for `degreesOfFreedom = 1` is undefined (there is no
//  *                                clear way to say it is either + or - infinity), and so the variance is also
//  *                                undefined.
//  * @param  parameters             Parameters used to instantiate this distribution.
//  * @param  graphParents
//  * @param  name                   Name for this distribution, which is prepended to all ops it creates.
//  *
//  * @author Emmanouil Antonios Platanios
//  */
//abstract class Distribution(
//    val dataType: DataType,
//    val reparameterizationType: ReparameterizationType,
//    val validateArguments: Boolean = false,
//    val allowNaNStatistics: Boolean = true,
//    val parameters: Map[String, OutputLike] = Map.empty[String, OutputLike],
//    val graphParents: Set[OutputLike] = Set.empty[OutputLike],
//    val name: String = "Distribution") {
//  protected def _parameterShapes(sampleShape: Output): Map[String, Output]
//  protected def _batchShape(): Output
//  protected def _eventShape(): Output
//  protected def _sampleN(n: Output, seed: Long): Output
//
//  def parameterShapes(sampleShape: Output, name: String = s"$name/ParameterShapes"): Map[String, Output] = {
//    Op.createWithNameScope(name, Set(sampleShape.op)) {
//      _parameterShapes(sampleShape)
//    }
//  }
//
//  def parameterStaticShapes(sampleShape: Shape): Map[String, Shape] = {
//    if (!sampleShape.isFullyDefined)
//      throw InvalidArgumentException(s"'sampleShape' ($sampleShape) must be fully defined.")
//    parameterShapes(sampleShape).mapValues(shape => {
//      Output.constantValueAsShape(shape).getOrElse(
//        throw InvalidArgumentException("All parameter shapes must be statically known."))
//    })
//  }
//
//  /** Shape of a single sample from a single event index, which may be partially defined or even unknown. The batch
//    * dimensions are indexes into independent and non-identical parameterizations of this distribution. */
//  def batchStaticShape(): Shape = Shape.unknown()
//
//  /** Shape of a single sample from a single event index, which may be partially defined or even unknown, represented as
//    * a one-dimensional `INT32` tensor. The batch dimensions are indexes into independent and non-identical
//    * parameterizations of this distribution. */
//  def batchShape(name: String = s"$name/BatchShape"): Output = {
//    Op.createWithNameScope(name) {
//      if (batchStaticShape().isFullyDefined)
//        batchStaticShape().toOutput(INT32)
//      else
//        _batchShape()
//    }
//  }
//
//  /** Shape of a single sample from a single batch, which may be partially defined or even unknown. */
//  def eventStaticShape(): Shape = Shape.unknown()
//
//  /** Shape of a single sample from a single batch, which may be partially defined or even unknown, represented as a
//    * one-dimensional `INT32` tensor. */
//  def eventShape(name: String = s"$name/EventShape"): Output = {
//    Op.createWithNameScope(name) {
//      if (batchStaticShape().isFullyDefined)
//        batchStaticShape().toOutput(INT32)
//      else
//        _batchShape()
//    }
//  }
//
//  /** `BOOLEAN` tensor indicating whether the batch shape is scalar. */
//  def isScalarBatch(name: String = s"$name/IsScalarBatch"): Output = {
//    Op.createWithNameScope(name) {
//      Distribution.isScalarHelper(batchStaticShape(), () => batchShape())
//    }
//  }
//
//  /** `BOOLEAN` tensor indicating whether the event shape is scalar. */
//  def isScalarEvent(name: String = s"$name/IsScalarEvent"): Output = {
//    Op.createWithNameScope(name) {
//      Distribution.isScalarHelper(eventStaticShape(), () => eventShape())
//    }
//  }
//
//  def sample(shape: Output, seed: Long = -1L, name: String = s"$name/Sample"): Output = {
//    Op.createWithNameScope(name, Set(shape.op)) {
//
//    }
//  }
//
//  // TODO: Add support for "copy".
//}
//
//object Distribution {
//  /** Helper method for determining whether a shape is scalar or not. */
//  private[Distribution] def isScalarHelper(staticShape: Shape, dynamicShapeFn: () => Output): Output = {
//    if (staticShape.rank != -1) {
//      Basic.constant(staticShape.rank == 0)
//    } else {
//      val shape = dynamicShapeFn()
//      if (shape.rank != -1 && shape.shape(0) != -1) {
//        // If the static shape is correctly written then we should never execute this branch. We keep it just in case
//        // there's some unimagined corner case.
//        Basic.constant(shape.shape == Shape(0))
//      } else {
//        Math.equal(Basic.shape(shape)(0), 0)
//      }
//    }
//  }
//
//  private[Distribution] def expandSampleShapeToVector(shape: Output): Output = {
//    val staticShape = Output.constantValue(shape)
//    val product: Output = staticShape.map(s => Basic.constant(s.prod())).getOrElse(Math.prod(shape))
//    val rank = shape.rank
//    if (rank == -1) {
//      // Maybe expand dimensions
//      val dynamicRank = Basic.rank(shape)
//
//    }
//  }
//}
