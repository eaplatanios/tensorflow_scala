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

package org.platanios.tensorflow.api.ops

/**
  * @author Emmanouil Antonios Platanios
  */
trait Text {
  /** Creates an op that joins the strings in the given list of string tensors into one tensor, using the provided
    * separator (which defaults to an empty string).
    *
    * @param  inputs    Sequence of string tensors that will be joined. The tensors must all have the same shape, or be
    *                   scalars. Scalars may be mixed in; these will be broadcast to the shape of the non-scalar inputs.
    * @param  separator Separator string.
    */
  def stringJoin(inputs: Seq[Output], separator: String = "", name: String = "StringJoin"): Output = {
    Op.Builder(opType = "StringJoin", name = name)
        .addInputList(inputs)
        .setAttribute("separator", separator)
        .build().outputs(0)
  }
}

object Text extends Text
