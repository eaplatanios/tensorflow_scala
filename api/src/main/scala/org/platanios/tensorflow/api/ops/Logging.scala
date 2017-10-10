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

/** Contains functions for constructing ops related to logging.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Logging {
  /** $OpDocLoggingPrint
    *
    * @group LoggingOps
    * @param  input     Input op output to pass through this op and return as its output.
    * @param  data      List of tensors whose values to print when the op is evaluated.
    * @param  message   Prefix of the printed values.
    * @param  firstN    Number of times to log. The op will log `data` only the `first_n` times it is evaluated. A value
    *                   of `-1` disables logging.
    * @param  summarize Number of entries to print for each tensor.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def print[T: OutputOps](
      input: T, data: Seq[Output], message: String = "", firstN: Int = -1, summarize: Int = 3,
      name: String = "Print"): T = {
    implicitly[OutputOps[T]].applyUnary(input, i => {
      Op.Builder("Print", name)
          .addInput(i)
          .addInputList(data)
          .setAttribute("message", message)
          .setAttribute("first_n", firstN)
          .setAttribute("summarize", summarize)
          .build().outputs(0)
    })
  }
}

private[api] object Logging extends Logging {
  private[ops] object Gradients {
    GradientsRegistry.register("Print", printGradient)

    private[this] def printGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      outputGradients ++ Seq.fill(op.inputs.length - 1)(null)
    }
  }

  /** @define OpDocLoggingPrint
    *   The `print` op prints a list of tensors.
    *
    *   The created op returns `input` as its output (i.e., it is effectively an identity op) and prints all the op
    *   output values in `data` while evaluating.
    */
  private[ops] trait Documentation
}
