package org.platanios.tensorflow.api

/**
  * @author Emmanouil Antonios Platanios
  */
final case class Shape(shape: Array[Long]) {
  def rank: Int = if (shape == null) -1 else shape.length
  def size(i: Int): Long = shape(i)
  def asArray: Array[Long] = shape

  override def toString: String = if (shape == null) "<unknown>" else shape.mkString(", ").replace("-1", "?")
}

object Shape {
  def apply(shape: Long*): Shape = Shape(shape.toArray)

  def unknown: Shape = Shape(null)
  def scalar: Shape = Shape(Array.empty[Long])
  def ofDim(firstDimensionSize: Long, otherDimensionSizes: Long*): Shape =
    Shape(Array(firstDimensionSize, otherDimensionSizes: _*))
}
