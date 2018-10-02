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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.variables.VariableLike
import org.platanios.tensorflow.api.tensors.{TensorConvertible, TensorLike}
import org.platanios.tensorflow.api.types._

import scala.collection.{TraversableLike, breakOut}

/** Type trait representing types that can be converted to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait OutputConvertible[OC] {
  type T

  // TODO: [OPS] Add data type argument.
  /** Converts `value` to a dense tensor. */
  @inline def toOutput(value: OC): Output[T]
}

object OutputConvertible {
  type Aux[OC, TT] = OutputConvertible[OC] {
    type T = TT
  }

  implicit def fromTensorLike[TT, TL[A] <: TensorLike[A]]: OutputConvertible.Aux[TL[TT], TT] = {
    new OutputConvertible[TL[TT]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: TL[TT]): Output[TT] = {
        Basic.constant(value.toTensor)
      }
    }
  }

  implicit def fromTensorConvertible[TT, TC[A]](implicit
      ev: TensorConvertible.Aux[TC[TT], TT]
  ): OutputConvertible.Aux[TC[TT], TT] = {
    new OutputConvertible[TC[TT]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: TC[TT]): Output[TT] = {
        Basic.constant(ev.toTensor(value))
      }
    }
  }

  implicit def fromOutputLike[TT, OL[A] <: OutputLike[A]]: OutputConvertible.Aux[OL[TT], TT] = {
    new OutputConvertible[OL[TT]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: OL[TT]): Output[TT] = {
        value.toOutput
      }
    }
  }

  implicit def fromTensorArray[TT]: OutputConvertible.Aux[TensorArray[TT], TT] = {
    new OutputConvertible[TensorArray[TT]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: TensorArray[TT]): Output[TT] = {
        value.toOutput
      }
    }
  }

  implicit def fromVariableLike[TT, VL[A] <: VariableLike[A]]: OutputConvertible.Aux[VL[TT], TT] = {
    new OutputConvertible[VL[TT]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: VL[TT]): Output[TT] = {
        value.toOutput
      }
    }
  }

  implicit val fromShape: OutputConvertible.Aux[Shape, Long] = {
    new OutputConvertible[Shape] {
      override type T = Long

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: Shape): Output[Long] = {
        value.toOutput(INT64)
      }
    }
  }

  implicit val fromRange: OutputConvertible.Aux[Range, Int] = {
    new OutputConvertible[Range] {
      override type T = Int

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: Range): Output[Int] = {
        Basic.stack(value.map(_.toTensor.toOutput))
      }
    }
  }

  implicit def fromSupportedType[TT](implicit
      evSupported: SupportedType[TT]
  ): OutputConvertible.Aux[TT, TT] = {
    new OutputConvertible[TT] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: TT): Output[TT] = {
        Basic.fill(evSupported.dataType, Shape())(value.toTensor)
      }
    }
  }

  implicit def fromArray[OC, TT](implicit
      ev: OutputConvertible.Aux[OC, TT]
  ): OutputConvertible.Aux[Array[OC], TT] = {
    new OutputConvertible[Array[OC]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: Array[OC]): Output[TT] = {
        Basic.stack(value.map(ev.toOutput))
      }
    }
  }

  implicit def fromTraversable[OC, TT, CC[A] <: TraversableLike[A, CC[A]]](implicit
      ev: OutputConvertible.Aux[OC, TT]
  ): OutputConvertible.Aux[CC[OC], TT] = {
    new OutputConvertible[CC[OC]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toOutput(value: CC[OC]): Output[TT] = {
        Basic.stack(value.map(ev.toOutput)(breakOut))
      }
    }
  }
}
