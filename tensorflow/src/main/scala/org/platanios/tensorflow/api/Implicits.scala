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

package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.core.Indexer.{Implicits => IndexerImplicits}
import org.platanios.tensorflow.api.core.Shape.{Implicits => ShapeImplicits}
import org.platanios.tensorflow.api.core.client.FeedMap.{Implicits => FeedMapImplicits}
import org.platanios.tensorflow.api.ops.{Op, OpSpecification, Output}
import org.platanios.tensorflow.api.ops.Output.{Implicits => OutputImplicits}
import org.platanios.tensorflow.api.tensors.Tensor.{Implicits => TensorImplicits}
import org.platanios.tensorflow.api.tensors.TensorFlowNative.{Implicits => TensorNativeImplicits}
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.types.SupportedType.{Implicits => SupportedTypeImplicits}

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends LowPriorityImplicits
        with IndexerImplicits {
  implicit def deviceImplicitConversion(device: String): (OpSpecification => String) = {
    Op.deviceImplicitConversion(device)
  }

  implicit def outputToInitialValueFunction(output: Output[DataType]): () => Output[DataType] = () => output
}

object Implicits extends Implicits

private[api] trait LowPriorityImplicits
    extends LowestPriorityImplicits
        with ShapeImplicits
        with OutputImplicits
        with FeedMapImplicits

private[api] trait LowestPriorityImplicits
    extends TensorImplicits
        with TensorNativeImplicits
        with SupportedTypeImplicits
