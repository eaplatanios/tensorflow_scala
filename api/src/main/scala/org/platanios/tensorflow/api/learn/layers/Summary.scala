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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.UINT8

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Summary(override protected val name: String) extends Layer[Output, Output](name)

object Summary {
  trait API {
    type ScalarSummary = layers.ScalarSummary
    type HistogramSummary = layers.HistogramSummary
    type ImageSummary = layers.ImageSummary
    type AudioSummary = layers.AudioSummary

    val ScalarSummary   : layers.ScalarSummary.type    = layers.ScalarSummary
    val HistogramSummary: layers.HistogramSummary.type = layers.HistogramSummary
    val ImageSummary    : layers.ImageSummary.type     = layers.ImageSummary
    val AudioSummary    : layers.AudioSummary.type     = layers.AudioSummary
  }

  object API extends API
}

case class ScalarSummary(
    tag: String,
    family: String = null,
    collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
    override protected val name: String = "ScalarSummary"
) extends Summary(name) {
  override val layerType: String = "ScalarSummary"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    ops.Summary.scalar(uniquifiedName, input, collections, family)
    LayerInstance(input, input)
  }
}

case class HistogramSummary(
    tag: String,
    family: String = null,
    collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
    override protected val name: String = "HistogramSummary"
) extends Summary(name) {
  override val layerType: String = "HistogramSummary"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    ops.Summary.histogram(uniquifiedName, input, collections, family)
    LayerInstance(input, input)
  }
}

case class ImageSummary(
    tag: String,
    badColor: Tensor = Tensor(UINT8, 255, 0, 0, 255),
    maxOutputs: Int = 3,
    family: String = null,
    collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
    override protected val name: String = "ImageSummary"
) extends Summary(name) {
  override val layerType: String = "ImageSummary"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    ops.Summary.image(uniquifiedName, input, badColor, maxOutputs, collections, family)
    LayerInstance(input, input)
  }
}

case class AudioSummary(
    tag: String,
    samplingRate: Tensor,
    maxOutputs: Int = 3,
    family: String = null,
    collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES),
    override protected val name: String = "AudioSummary"
) extends Summary(name) {
  override val layerType: String = "AudioSummary"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    ops.Summary.audio(uniquifiedName, input, samplingRate, maxOutputs, collections, family)
    LayerInstance(input, input)
  }
}
