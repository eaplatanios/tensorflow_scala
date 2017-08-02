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

import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}

/**
  * @author Emmanouil Antonios Platanios
  */
trait Logging {
  // TODO: Look into control flow ops for the "Assert" op.
  //  /** Asserts that the provided condition is true.
  //    *
  //    * If `condition` evaluates to `false`, then the op prints all the op outputs in `data`. `summarize` determines how
  //    * many entries of the tensors to print.
  //    *
  //    * @param  condition Condition to assert.
  //    * @param  data      Op outputs whose values are printed if `condition` is `false`.
  //    * @param  summarize Number of tensor entries to print.
  //    * @param  name      Name for the created op.
  //    * @return Created op.
  //    */
  //  def assert(condition: Output, data: Array[Output], summarize: Int = 3, name: String = "Assert"): Output = {
  //    createWith(nameScope = name, values = condition +: data) {
  //      internalAssert(condition = condition, data = data, summarize = summarize)
  //    }
  //  }
  //
  //  private[this] def internalAssert(
  //      condition: Output, data: Array[Output], summarize: Int = 3, name: String = "Assert")
  //      (implicit context: DynamicVariable[OpCreationContext]): Output =
  //    Op.Builder(context = context, opType = "Assert", name = name)
  //        .addInput(condition)
  //        .addInputList(data)
  //        .setAttribute("summarize", summarize)
  //        .build().outputs(0)

  /** Creates an op that prints a list of tensors.
    *
    * The created op returns `input` as its output (i.e., it is effectively an identity op) and prints all the op output
    * values in `data` while evaluating.
    *
    * @param  input     Input op output to pass through this op and return as its output.
    * @param  data      List of tensors whose values to print when the op is evaluated.
    * @param  message   Prefix of the printed values.
    * @param  firstN    Number of times to log. The op will log `data` only the `first_n` times it is evaluated. A value
    *                   of `-1` disables logging.
    * @param  summarize Number of entries to print for each tensor.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def print(
      input: Output, data: Seq[Output], message: String = "", firstN: Int = -1, summarize: Int = 3,
      name: String = "Print"): Output = {
    Op.Builder(opType = "Print", name = name)
        .addInput(input)
        .addInputList(data)
        .setAttribute("message", message)
        .setAttribute("first_n", firstN)
        .setAttribute("summarize", summarize)
        .build().outputs(0)
  }
}

object Logging extends Logging {
  private[api] object Gradients {
    GradientsRegistry.register("Print", printGradient)

    private[this] def printGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      outputGradients ++ Seq.fill(op.inputs.length - 1)(null)
    }
  }
}
