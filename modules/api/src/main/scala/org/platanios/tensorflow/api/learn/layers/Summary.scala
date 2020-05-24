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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.types.{IsReal, TF, UByte}
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Summary[T: TF : IsReal](
    override val name: String
) extends Layer[Output[T], Output[T]](name)

object Summary {
  private[layers] trait API {
    type Summary[T] = layers.Summary[T]
    type ScalarSummary[T] = layers.ScalarSummary[T]
    type HistogramSummary[T] = layers.HistogramSummary[T]
    type ImageSummary[T] = layers.ImageSummary[T]
    type AudioSummary = layers.AudioSummary

    val ScalarSummary   : layers.ScalarSummary.type    = layers.ScalarSummary
    val HistogramSummary: layers.HistogramSummary.type = layers.HistogramSummary
    val ImageSummary    : layers.ImageSummary.type     = layers.ImageSummary
    val AudioSummary    : layers.AudioSummary.type     = layers.AudioSummary
  }

  object API extends API
}

case class ScalarSummary[T: TF : IsReal](
    override val name: String,
    tag: String,
    family: String = null,
    collections: Set[Graph.Key[Output[Any]]] = Set(Graph.Keys.SUMMARIES)
) extends Summary(name) {
  override val layerType: String = "ScalarSummary"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.Summary.scalar(tag, input, collections, family)
    input
  }
}

case class HistogramSummary[T: TF : IsReal](
    override val name: String,
    tag: String,
    family: String = null,
    collections: Set[Graph.Key[Output[Any]]] = Set(Graph.Keys.SUMMARIES)
) extends Summary(name) {
  override val layerType: String = "HistogramSummary"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.Summary.histogram(tag, input, collections, family)
    input
  }
}

case class ImageSummary[T: TF : IsReal](
    override val name: String,
    tag: String,
    badColor: Tensor[UByte] = Tensor[UByte](
      Tensor.fill[UByte](Shape())(UByte(255.toByte)),
      Tensor.fill[UByte](Shape())(UByte(0.toByte)),
      Tensor.fill[UByte](Shape())(UByte(0.toByte)),
      Tensor.fill[UByte](Shape())(UByte(255.toByte))),
    maxOutputs: Int = 3,
    family: String = null,
    collections: Set[Graph.Key[Output[Any]]] = Set(Graph.Keys.SUMMARIES)
) extends Summary(name) {
  override val layerType: String = "ImageSummary"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.Summary.image(tag, input, badColor, maxOutputs, collections, family)
    input
  }
}

case class AudioSummary(
    override val name: String,
    tag: String,
    samplingRate: Tensor[Float],
    maxOutputs: Int = 3,
    family: String = null,
    collections: Set[Graph.Key[Output[Any]]] = Set(Graph.Keys.SUMMARIES)
) extends Summary[Float](name) {
  override val layerType: String = "AudioSummary"

  override def forwardWithoutContext(
      input: Output[Float]
  )(implicit mode: Mode): Output[Float] = {
    ops.Summary.audio(tag, input, samplingRate.toOutput, maxOutputs, collections, family)
  input
  }
}
