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

package org.platanios.tensorflow.api.learn

import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.ops.{Clip, Op, OutputLike}

/** Represents a gradient-clipping method that can be used while training.
  *
  * @author Emmanouil Antonios Platanios
  */
trait ClipGradients {
  /** Clips the provided gradients paired with the corresponding variables and returns the result. */
  def apply(gradientsAndVariables: Seq[(OutputLike, Variable)]): Seq[(OutputLike, Variable)] = {
    clipGradients(gradientsAndVariables)
  }

  /** Clips the provided gradients paired with the corresponding variables and returns the result. */
  def clipGradients(gradientsAndVariables: Seq[(OutputLike, Variable)]): Seq[(OutputLike, Variable)]
}

/** Represents no clipping of the gradients (i.e., identity operation). */
case object NoClipGradients extends ClipGradients {
  override def clipGradients(gradientsAndVariables: Seq[(OutputLike, Variable)]): Seq[(OutputLike, Variable)] = {
    gradientsAndVariables
  }
}

/** Clips the gradients using the `clipByValue` op.
  *
  * $OpDocClipClipByValue
  *
  * @param  clipValueMin Minimum value to clip by.
  * @param  clipValueMax Maximum value to clip by.
  */
case class ClipGradientsByValue(clipValueMin: Float, clipValueMax: Float) extends ClipGradients {
  /** Clips the provided gradients paired with the corresponding variables and returns the result. */
  override def clipGradients(gradientsAndVariables: Seq[(OutputLike, Variable)]): Seq[(OutputLike, Variable)] = {
    gradientsAndVariables.map(gv => {
      (Clip.clipByValue(gv._1, clipValueMin, clipValueMax), gv._2)
    })
  }
}

/** Clips the gradients using the `clipByNorm` op.
  *
  * $OpDocClipClipByNorm
  *
  * @param  clipNorm Maximum norm clipping value (must be > 0).
  */
case class ClipGradientsByNorm(clipNorm: Float) extends ClipGradients {
  /** Clips the provided gradients paired with the corresponding variables and returns the result. */
  override def clipGradients(gradientsAndVariables: Seq[(OutputLike, Variable)]): Seq[(OutputLike, Variable)] = {
    gradientsAndVariables.map(gv => {
      (Clip.clipByNorm(gv._1, clipNorm), gv._2)
    })
  }
}

/** Clips the gradients using the `clipByAverageNorm` op.
  *
  * $OpDocClipClipByAverageNorm
  *
  * @param  clipNorm Maximum average norm clipping value (must be > 0).
  */
case class ClipGradientsByAverageNorm(clipNorm: Float) extends ClipGradients {
  /** Clips the provided gradients paired with the corresponding variables and returns the result. */
  override def clipGradients(gradientsAndVariables: Seq[(OutputLike, Variable)]): Seq[(OutputLike, Variable)] = {
    gradientsAndVariables.map(gv => {
      (Clip.clipByAverageNorm(gv._1, clipNorm), gv._2)
    })
  }
}

/** Clips the gradients using the `clipByGlobalNorm` op.
  *
  * $OpDocClipClipByGlobalNorm
  *
  * @param  clipNorm Maximum norm clipping value (must be > 0).
  */
case class ClipGradientsByGlobalNorm(clipNorm: Float) extends ClipGradients {
  /** Clips the provided gradients paired with the corresponding variables and returns the result. */
  override def clipGradients(gradientsAndVariables: Seq[(OutputLike, Variable)]): Seq[(OutputLike, Variable)] = {
    Op.createWithNameScope("ClipGradients") {
      Clip.clipByGlobalNorm(gradientsAndVariables.map(_._1), clipNorm)._1.zip(gradientsAndVariables.map(_._2))
    }
  }
}
