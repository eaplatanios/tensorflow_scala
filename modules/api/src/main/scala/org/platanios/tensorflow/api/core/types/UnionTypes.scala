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

package org.platanios.tensorflow.api.core.types

import shapeless._

/**
  * @author Emmanouil Antonios Platanios
  */
trait UnionTypes {
  type ![A] = A => Nothing
  type !![A] = ![![A]]

  trait Disjunction { self =>
    type D
    type or[S] = Disjunction {
      type D = self.D with ![S]
    }
  }

  type Union[T] = {
    type or[S] = (Disjunction {type D = ![T]})#or[S]
  }

  type Contains[S, T <: Disjunction] = !![S] <:< ![T#D]
  type ∈[S, T <: Disjunction] = Contains[S, T]
}

object UnionTypes extends UnionTypes
