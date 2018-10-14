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

package org.platanios.tensorflow.api.ops.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.Variant
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.ops.{Basic, Function, InstantiatedFunction, Op, Output}

/** Contains implementations for some experimental dataset ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Experimental {
  /** Creates a dataset that deterministically chooses elements from `inputDatasets`.
    *
    * For example:
    * {{{
    *   val inputDatasets = Seq(
    *     tf.data.datasetFromTensors(Tensor("foo")).repeat(),
    *     tf.data.datasetFromTensors(Tensor("bar")).repeat(),
    *     tf.data.datasetFromTensors(Tensor("baz")).repeat())
    *
    *   // The following dataset contains: [0, 1, 2, 0, 1, 2, 0, 1, 2].
    *   val selectorDataset = tf.data.datasetFromRange(0, 3).repeat(3)
    *
    *   // The following dataset contains: ["foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"].
    *   val result = tf.data.directedInterleave(selectorDataset, inputDatasets)
    * }}}
    *
    * @param  selectorDataset Selector dataset that determines the interleaving order.
    * @param  inputDatasets   Input datasets that are interleaved.
    * @param  name            Name for the new dataset.
    * @tparam T Symbolic tensor type of the element (symbolic equivalent of `V`).
    * @tparam V Value tensor type of the element.
    * @tparam D Data type of the element.
    * @tparam S Shape of the element.
    * @return New dataset.
    */
  def chooseFromDatasets[T, V, D, S](
      selectorDataset: Dataset[Output[Long]],
      inputDatasets: Seq[Dataset[T]],
      name: String = "ChooseFromDatasets"
  )(implicit evT: NestedStructure.Aux[T, V, D, S]): Dataset[T] = {
    val providedName = name
    new Dataset[T] {
      override val name: String = providedName

      private var mostSpecificFlattenedShapes: Option[Seq[Shape]] = None

      override def createHandle[VV, DD, SS]()(implicit evT: NestedStructure.Aux[T, VV, DD, SS]): Output[Variant] = {
        mostSpecificFlattenedShapes = Some(
          inputDatasets.map(_.flatOutputShapes).reduceLeft[Seq[Shape]] {
            case (specificShapes, shapes) =>
              specificShapes.zip(shapes).map(p => {
                Shape.fromSeq(p._1.asArray.zip(p._2.asArray).map {
                  case (d1, d2) if d1 == d2 => d1
                  case (d1, d2) if d1 == -1 => d2
                  case (d1, d2) if d2 == -1 => d1
                  case _ => -1
                })
              })
          })
        Op.Builder[(Output[Variant], Seq[Output[Variant]]), Output[Variant]](
          opType = "ExperimentalDirectedInterleaveDataset",
          name = name,
          input = (
              selectorDataset.createHandle(),
              inputDatasets.map(_.createHandle()))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[VV, DD, SS](implicit evT: NestedStructure.Aux[T, VV, DD, SS]): DD = {
        inputDatasets.head.outputDataTypes
      }

      override def outputShapes[VV, DD, SS](implicit evT: NestedStructure.Aux[T, VV, DD, SS]): SS = {
        evT.decodeShapeFromDataType(
          outputDataTypes,
          mostSpecificFlattenedShapes.get
        )._1
      }
    }
  }

  // TODO: [DATA] `sampleFromDatasets`
}

object Experimental extends Experimental
