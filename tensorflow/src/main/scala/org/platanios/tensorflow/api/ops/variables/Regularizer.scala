package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.ops.Op

/** A variable regularizer is simply a function that takes an `Op.Output` representing the variable value as input, and
  * returns another `Op.Output` representing the regularizer value as output.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Regularizer {
  def apply(value: Op.Output): Op.Output
}
