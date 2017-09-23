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
import org.platanios.tensorflow.api.learn.layers
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.UINT8

/**
  * @author Emmanouil Antonios Platanios
  */
trait Summary extends NetworkLayer[Output, Output]

object Summary {
  trait API {
    val ScalarSummary   : layers.ScalarSummary.type    = layers.ScalarSummary
    val HistogramSummary: layers.HistogramSummary.type = layers.HistogramSummary
    val ImageSummary    : layers.ImageSummary.type     = layers.ImageSummary
    val AudioSummary    : layers.AudioSummary.type     = layers.AudioSummary
  }

  object API extends API
}

case class ScalarSummary private[layers](
    override val name: String,
    family: String = null,
    collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES)) extends Summary {
  override val layerType: String           = s"ScalarSummary"
  override val forward  : Output => Output = { input =>
    ops.Summary.scalar(name, input, collections, family)
    input
  }
}

case class HistogramSummary private[layers](
    override val name: String,
    family: String = null,
    collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES)) extends Summary {
  override val layerType: String           = s"HistogramSummary"
  override val forward  : Output => Output = { input =>
    ops.Summary.histogram(name, input, collections, family)
    input
  }
}

case class ImageSummary private[layers](
    override val name: String,
    badColor: Tensor = Tensor(UINT8, 255, 0, 0, 255),
    maxOutputs: Int = 3,
    family: String = null,
    collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES)) extends Summary {
  override val layerType: String           = s"ImageSummary"
  override val forward  : Output => Output = { input =>
    ops.Summary.image(name, input, badColor, maxOutputs, collections, family)
    input
  }
}

case class AudioSummary private[layers](
    override val name: String,
    samplingRate: Tensor,
    maxOutputs: Int = 3,
    family: String = null,
    collections: Set[Graph.Key[Output]] = Set(Graph.Keys.SUMMARIES)) extends Summary {
  override val layerType: String           = s"AudioSummary"
  override val forward  : Output => Output = { input =>
    ops.Summary.audio(name, input, samplingRate, maxOutputs, collections, family)
    input
  }
}
