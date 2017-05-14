package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.Shape
import org.platanios.tensorflow.api.tf.DataType

/** A variable partitioner is simply a function that accepts the fully defined `Shape` and the `DataType` of the
  * variable to be created, and returns an array of integers corresponding to the number of partitions for each axis
  * (currently only one axis can be partitioned).
  *
  * @author Emmanouil Antonios Platanios
  */
trait Partitioner {
  def apply(shape: Shape, dataType: DataType): Array[Int]
}
