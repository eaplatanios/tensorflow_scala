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
  def stringJoin(inputs: Seq[Op.Output], separator: String = "", name: String = "StringJoin"): Op.Output = {
    Op.Builder(opType = "StringJoin", name = name)
        .addInputs(inputs)
        .setAttribute("separator", separator)
        .build().outputs(0)
  }
}

object Text extends Text
