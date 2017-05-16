package org.platanios.tensorflow.api.ops

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
  //  def assert(condition: Op.Output, data: Array[Op.Output], summarize: Int = 3, name: String = "Assert"): Op.Output = {
  //    createWith(nameScope = name, values = condition +: data) {
  //      internalAssert(condition = condition, data = data, summarize = summarize)
  //    }
  //  }
  //
  //  private[this] def internalAssert(
  //      condition: Op.Output, data: Array[Op.Output], summarize: Int = 3, name: String = "Assert")
  //      (implicit context: DynamicVariable[OpCreationContext]): Op.Output =
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
      input: Op.Output, data: Array[Op.Output], message: String = "", firstN: Int = -1, summarize: Int = 3,
      name: String = "Print"): Op.Output = {
    Op.Builder(opType = "Print", name = name)
        .addInput(input)
        .addInputList(data)
        .setAttribute("message", message)
        .setAttribute("first_n", firstN)
        .setAttribute("summarize", summarize)
        .build().outputs(0)
  }
}

object Logging extends Logging
