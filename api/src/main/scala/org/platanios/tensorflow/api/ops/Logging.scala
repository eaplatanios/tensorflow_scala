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
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.types.{INT32, STRING}

/** Contains functions for constructing ops related to logging.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Logging {
  /** $OpDocLoggingAssert
    *
    * @group LoggingOps
    * @param  condition Condition to assert.
    * @param  data      Op outputs whose values are printed if `condition` is `false`.
    * @param  summarize Number of tensor entries to print.
    * @param  name      Name for the created op.
    * @return Created op.
    */
  def assert(condition: Output, data: Seq[Output], summarize: Int = 3, name: String = "Assert"): Op = {
    Op.createWithNameScope(name) {
      if (data.forall(d => d.dataType == STRING || d.dataType == INT32)) {
        // As a simple heuristic, we assume that STRING and INT32 tensors are on host memory to avoid the need to use
        // `cond`. If that is not case, we will pay the price copying the tensor to host memory.
        Op.Builder("Assert", name)
            .addInput(condition)
            .addInputList(data)
            .setAttribute("summarize", summarize)
            .build()
      } else {
        ControlFlow.cond(
          condition,
          () => ControlFlow.noOp(),
          () => Op.Builder("Assert", name)
              .addInput(condition)
              .addInputList(data)
              .setAttribute("summarize", summarize)
              .build(),
          name = "AssertGuard")
      }
    }
  }

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
  def print(
      input: Output, data: Seq[Output], message: String = "", firstN: Int = -1, summarize: Int = 3,
      name: String = "Print"): Output = {
    Op.Builder("Print", name)
        .addInput(input)
        .addInputList(data)
        .setAttribute("message", message)
        .setAttribute("first_n", firstN)
        .setAttribute("summarize", summarize)
        .build().outputs(0)
  }
}

private[api] object Logging extends Logging {
  private[ops] object Gradients {
    GradientsRegistry.register("Print", printGradient)

    private[this] def printGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      outputGradients ++ Seq.fill(op.inputs.length - 1)(null)
    }
  }

  /** @define OpDocLoggingAssert
    *   The `assert` op asserts that the provided condition is true.
    *
    *   If `condition` evaluates to `false`, then the op prints all the op outputs in `data`. `summarize` determines how
    *   many entries of the tensors to print.
    *
    *   Note that to ensure that `assert` executes, one usually attaches it as a dependency:
    *   {{{
    *     // Ensure maximum element of x is smaller or equal to 1.
    *     val assertOp = tf.assert(tf.lessEqual(tf.max(x), 1.0), Seq(x))
    *     Op.createWith(controlDependencies = Set(assertOp)) {
    *       ... code using x ...
    *     }
    *   }}}
    *
    * @define OpDocLoggingPrint
    *   The `print` op prints a list of tensors.
    *
    *   The created op returns `input` as its output (i.e., it is effectively an identity op) and prints all the op
    *   output values in `data` while evaluating.
    */
  private[ops] trait Documentation
}
