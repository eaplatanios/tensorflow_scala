package org.platanios.tensorflow.api.ops

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object GradientsSpec {
  /** Gathers and returns all inputs of `destinations` (recursively) that have been reached.
    *
    * @param  destinations Ops whose inputs are being gathered.
    * @param  reached      Reached ops.
    * @return Set of input ops to `destinations` (recursively) that have been reached.
    */
  private[this] def gatherInputs(destinations: Set[Op], reached: mutable.Set[Op]): Set[Op] = {
    val inputs = mutable.Set.empty[Op]
    val queue = mutable.Queue[Op](destinations.toSeq: _*)
    while (queue.nonEmpty) {
      val op = queue.dequeue()
      if (reached.contains(op)) {
        inputs += op
        reached -= op // Done so we don't go through the same ops twice
        op.inputs.foreach(i => queue.enqueue(i.op))
      }
    }
    inputs.toSet
  }
}
