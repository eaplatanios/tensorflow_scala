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
import org.platanios.tensorflow.api.learn.layers.{Layer, MapSeq}

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
    ): MapSeq[T, R, S, CC] = {
      MapSeq[T, R, S, CC](layer.name, layer, mapLayer)(cbfRS)
    }
  }

  implicit def unsupervisedTrainableModelToUnsupervisedModelFunction[In, Out, Loss](
      model: UnsupervisedTrainableModel[In, Out, Loss]
  ): UnsupervisedModelFunction[In, Out, Loss] = {
    UnsupervisedModelFunction((_: Configuration) => model)
  }

  implicit def unsupervisedTrainableModelUnitFunctionToUnsupervisedModelFunction[In, Out, Loss](
      function: () => UnsupervisedTrainableModel[In, Out, Loss]
  ): UnsupervisedModelFunction[In, Out, Loss] = {
    UnsupervisedModelFunction((_: Configuration) => function())
  }

  implicit def unsupervisedTrainableModelUnaryRunConfigFunctionToUnsupervisedModelFunction[In, Out, Loss](
      function: Configuration => UnsupervisedTrainableModel[In, Out, Loss]
  ): UnsupervisedModelFunction[In, Out, Loss] = {
    UnsupervisedModelFunction(function)
  }

  implicit def supervisedTrainableModelToModelFunction[In, TrainIn, TrainOut, Out, Loss](
      model: SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss]
  ): SupervisedModelFunction[In, TrainIn, TrainOut, Out, Loss] = {
    SupervisedModelFunction((_: Configuration) => model)
  }

  implicit def supervisedTrainableModelUnitFunctionToModelFunction[In, TrainIn, TrainOut, Out, Loss](
      function: () => SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss]
  ): SupervisedModelFunction[In, TrainIn, TrainOut, Out, Loss] = {
    SupervisedModelFunction((_: Configuration) => function())
  }

  implicit def supervisedTrainableModelUnaryRunConfigFunctionToModelFunction[In, TrainIn, TrainOut, Out, Loss](
      function: Configuration => SupervisedTrainableModel[In, TrainIn, TrainOut, Out, Loss]
  ): SupervisedModelFunction[In, TrainIn, TrainOut, Out, Loss] = {
    SupervisedModelFunction(function)
  }
}
