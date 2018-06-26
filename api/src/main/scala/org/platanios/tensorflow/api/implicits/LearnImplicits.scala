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

package org.platanios.tensorflow.api.implicits

import org.platanios.tensorflow.api.learn.{Configuration, Mode, SupervisedTrainableModel, UnsupervisedTrainableModel}
import org.platanios.tensorflow.api.learn.estimators.Estimator.{SupervisedModelFunction, UnsupervisedModelFunction}
import org.platanios.tensorflow.api.learn.layers.{Layer, Map}

import scala.collection.TraversableLike
import scala.collection.generic.CanBuildFrom

/** Groups together all implicits related to the learn API.
  *
  * @author Emmanouil Antonios Platanios
  */
private[implicits] trait LearnImplicits {
  implicit class MappableLayer[T, R, CC[A] <: TraversableLike[A, CC[A]]](
      layer: Layer[CC[T], CC[R]]
  ) extends Layer[CC[T], CC[R]]("Mappable") {
    override val layerType: String = "Mappable"

    override def forwardWithoutContext(input: CC[T])(implicit mode: Mode): CC[R] = {
      layer(input)
    }

    def map[S](
        layer: Layer[CC[T], CC[R]],
        mapLayer: Layer[R, S]
    )(implicit
        cbfRS: CanBuildFrom[CC[R], S, CC[S]]
    ): Map[T, R, S, CC] = {
      Map[T, R, S, CC](layer.name, layer, mapLayer)(cbfRS)
    }
  }

  implicit def unsupervisedTrainableModelToUnsupervisedModelFunction[IT, IO, ID, IS, I](
      model: UnsupervisedTrainableModel[IT, IO, ID, IS, I]
  ): UnsupervisedModelFunction[IT, IO, ID, IS, I] = {
    UnsupervisedModelFunction((_: Configuration) => model)
  }

  implicit def unsupervisedTrainableModelUnitFunctionToUnsupervisedModelFunction[IT, IO, ID, IS, I](
      function: () => UnsupervisedTrainableModel[IT, IO, ID, IS, I]
  ): UnsupervisedModelFunction[IT, IO, ID, IS, I] = {
    UnsupervisedModelFunction((_: Configuration) => function())
  }

  implicit def unsupervisedTrainableModelUnaryRunConfigFunctionToUnsupervisedModelFunction[IT, IO, ID, IS, I](
      function: Configuration => UnsupervisedTrainableModel[IT, IO, ID, IS, I]
  ): UnsupervisedModelFunction[IT, IO, ID, IS, I] = {
    UnsupervisedModelFunction(function)
  }

  implicit def supervisedTrainableModelToModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      model: SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
  ): SupervisedModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    SupervisedModelFunction((_: Configuration) => model)
  }

  implicit def supervisedTrainableModelUnitFunctionToModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      function: () => SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
  ): SupervisedModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    SupervisedModelFunction((_: Configuration) => function())
  }

  implicit def supervisedTrainableModelUnaryRunConfigFunctionToModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T](
      function: Configuration => SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
  ): SupervisedModelFunction[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = {
    SupervisedModelFunction(function)
  }
}
