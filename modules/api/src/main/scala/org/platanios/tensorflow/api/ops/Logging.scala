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

/** Contains functions for constructing ops related to logging.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Logging {
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
    * @return Created op output.
    */
  def print[T, OL[A] <: OutputLike[A]](
      input: OL[T],
      data: Seq[Output[Any]],
      message: String = "",
      firstN: Int = -1,
      summarize: Int = 3,
      name: String = "Print"
  )(implicit
      ev: OutputOps.Aux[OL, T]
  ): OL[T] = {
    ev.applyUnary(input, o =>
      Op.Builder[(Output[T], Seq[Output[Any]]), Output[T]](
        opType = "Print",
        name = name,
        input = (o, data)
      ).setAttribute("message", message)
          .setAttribute("first_n", firstN)
          .setAttribute("summarize", summarize)
          .setGradientFn(printGradient)
          .build().output)
  }

  protected def printGradient[T](
      op: Op[(Output[T], Seq[Output[Any]]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Seq[Output[Any]]) = {
    (outputGradient, Seq.fill(op.input._2.length)(null))
  }

  /** $OpDocLoggingTimestamp
    *
    * @group LoggingOps
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def timestamp(name: String = "Timestamp"): Output[Double] = {
    Op.Builder[Unit, Output[Double]](
      opType = "Timestamp",
      name = name,
      input = ()
    ).build().output
  }
}

object Logging extends Logging {
  /** @define OpDocLoggingPrint
    *   The `print` op prints a list of tensors.
    *
    *   The created op returns `input` as its output (i.e., it is effectively an identity op) and prints all the op
    *   output values in `data` while evaluating.
    *
    * @define OpDocLoggingTimestamp
    *   The `timestamp` op returns a `FLOAT64` tensor that contains the time since the Unix epoch in seconds. Note that
    *   the timestamp is computed when the op is executed, not when it is added to the graph.
    */
  private[ops] trait Documentation
}
