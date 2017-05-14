package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Basic, Op}
import org.platanios.tensorflow.api.types.DataType

import scala.math.Ordering.Implicits._

/** Partitioned variable wrapper.
  *
  * Variables passed via `wrappedVariables` must contain a non-null save slice information field. Concatenation and
  * iteration is in lexicographic order according to the `variableOffset` property of the save slice information.
  *
  * Accessing this object as an [[Op.Output]] returns the variable parts concatenated along the partition axis.
  *
  * @param  name             Overall name of the variables.
  * @param  shape            Overall shape of the variables.
  * @param  dataType         Data type of the variables.
  * @param  wrappedVariables Variables that comprise this partitioned variable.
  * @param  partitions       Number of partitions for each dimension.
  * @throws IllegalArgumentException If the provided variables sequence is empty, or if their shapes do not match with
  *                                  `shape`, or if their data types do not match `dataType`., or if they have
  *                                  `null`-valued save slice information
  *
  * @author Emmanouil Antonios Platanios
  */
@throws[IllegalArgumentException]
case class PartitionedVariable private[variables] (
    name: String, shape: Shape, dataType: DataType, private val wrappedVariables: Seq[Variable],
    partitions: Array[Int]) extends Op.OutputConvertible with Iterable[Variable] {
  // TODO: !!! Check on the data type.
  if (shape.rank != partitions.length)
    throw new IllegalArgumentException(
      s"The number of partitions provided (${partitions.length}) does not match the shape rank (${shape.rank}).")
  if (partitions.exists(_ <= 0))
    throw new IllegalArgumentException(s"All partitions must be positive: $partitions.")
  if (wrappedVariables.isEmpty)
    throw new IllegalArgumentException("The provided variables list may not be empty.")
  if (wrappedVariables.exists(_.saveSliceInformation == null))
    throw new IllegalArgumentException(s"All variables must have save slice information available.")
  if (wrappedVariables.exists(_.saveSliceInformation.fullShape != shape))
    throw new IllegalArgumentException(
      "All variables' save slice information full shape must match the provided shape.")
  if (wrappedVariables.exists(_.dataType != dataType))
    throw new IllegalArgumentException("All variables' data type must match the provided data type.")
  val variables: Seq[Variable] = wrappedVariables.sortBy(_.saveSliceInformation.variableOffset.toList)

  /** Returns the overall concatenated value as an [[Op.Output]].
    *
    * This is different from using the partitioned variable directly as an op output (through implicit conversion or
    * `toOpOutput`) in that it creates a new set of ops that retain the control dependencies from its scope.
    *
    * @return Concatenated op output.
    * @throws IllegalArgumentException If having more than one partition axes.
    */
  @throws[IllegalArgumentException]
  private[this] def concatenated: Op.Output = {
    val concatenated: Op.Output = {
      if (variables.length == 1) {
        variables.head
      } else {
        if (partitionAxes.length > 1)
          throw new IllegalArgumentException(
            s"Cannot concatenate along more than one dimension: $partitionAxes. " +
                "Multi-axis partition concatenation is not supported.")
        Op.createWithNameScope(s"$name/ConcatenatedPartitions") {
          Basic.concatenate(variables.map(_.toOpOutput).toArray, partitionAxes(0))
        }
      }
    }
    Op.createWithNameScope("") {
      Basic.identity(concatenated, name)
    }
  }

  /** Converts this partitioned variable to an op output.
    *
    * The returned op output will not inherit the control dependencies from the scope where the value is used, which is
    * equivalent behavior to that of getting the value of a variable.
    */
  def toOpOutput: Op.Output = {
    Op.createWith(controlDependencies = Set.empty[Op]) {
      concatenated
    }
  }

  /** Returns an array of integers contining the partition axes of this partitioned variable. */
  private[this] val partitionAxes: Array[Int] = {
    val filteredPartitions = partitions.filter(_ > 1)
    if (filteredPartitions.isEmpty)
      Array[Int](0)
    else
      filteredPartitions
  }

  /** Returns the number of partitions of this partitioned variable.
    *
    * @throws IllegalArgumentException If having more than one partition axes.
    */
  @throws[IllegalArgumentException]
  private[this] def length: Int = {
    if (partitionAxes.length > 1)
      throw new IllegalArgumentException(s"Cannot get a length for ${partitionAxes.length} > 1 partition axes.")
    variables.length
  }

  /** Returns an iterator for accessing the underlying partition variables.
    *
    * This iterator is necessary to control order of access when variables are not partitioned in a standard way along
    * a single axis.
    */
  override def iterator: Iterator[Variable] = new Iterator[Variable] {
    private[this] var index: Int = 0

    override def hasNext: Boolean = index < variables.length

    override def next(): Variable = {
      val nextVariable = variables(index)
      index += 1
      nextVariable
    }
  }
}
