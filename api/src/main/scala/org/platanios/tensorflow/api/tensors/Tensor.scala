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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER
import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.{Index, Indexer, Shape}
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.ops.{Basic, Output, OutputConvertible}
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import spire.math.UShort

import java.nio._
import java.nio.charset.Charset

import scala.language.postfixOps

// TODO: Specialized slices (e.g., contiguous).
// TODO: Is there a need to complicate the flattened index function for the plain tensor?
// TODO: Add casting support.
// TODO: Should we keep assuming that tensor shapes are fully defined here?

/**
  * @author Emmanouil Antonios Platanios
  */
trait Tensor extends TensorLike with OutputConvertible {
  val buffer: ByteBuffer

  val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER

  private[api] def flattenedIndex(indices: Array[Int]): Int = order.index(shape.asArray, indices)
  private[api] def flattenedIndexIterator: Iterator[Int] = order.indexIterator(shape.asArray)

  private[api] def setElementAtFlattenedIndex[T](index: Int, value: T)(implicit evidence: SupportedType[T]): this.type
  private[api] def getElementAtFlattenedIndex(index: Int): dataType.ScalaType

  def entriesIterator: Iterator[dataType.ScalaType] = flattenedIndexIterator.map(getElementAtFlattenedIndex)

  // def update(indices: Array[Int], value: DataType.SupportedScalaType): Unit = {
  //   require(indices.length == rank, "Incomplete set of indices provided.")
  //   val index = order.index(shape.asArray, beginOffsets, strides, indices)
  //   dataType.putElementInBuffer(buffer = buffer, index = index, element = dataType.cast(value))
  // }

  // TODO: Need to improve the syntax here (maybe using implicit conversion to indexer sequences).
  def update(indexers: Seq[Indexer], tensor: Tensor): Unit = {
    val decoded = Indexer.decode(shape, indexers)
    val sliceShape = Shape.fromSeq(decoded._2)
    if (sliceShape != tensor.shape)
      throw ShapeMismatchException(
        s"Tensor slice shape '$sliceShape' does not match assigned tensor shape '${tensor.shape}'.")
    val stridedIndexIterator = order.indexIterator(decoded._1, decoded._3, decoded._4, decoded._5)
    for ((index, stridedIndex) <- tensor.flattenedIndexIterator zip stridedIndexIterator) {
      // TODO: Avoid casting for tensors with the same data type.
      val castedValue = dataType.cast(tensor.getElementAtFlattenedIndex(index))(tensor.dataType.supportedType)
      setElementAtFlattenedIndex(stridedIndex, castedValue)(dataType.supportedType)
    }
  }

  // def update(index: Int, tensor: Tensor): Unit = update(Seq[Indexer](index), tensor)
  //
  // def update(indexers: Seq[Indexer], tensor: Tensor): Unit = slice(indexers: _*).set(tensor)

  def fill[T](value: T)(implicit evidence: SupportedType[T]): this.type

  // TODO: Find a way to add this method for performance benefits.
  // def set(value: SupportedScalaType): Tensor = fill(value)

  def set(tensor: Tensor): this.type = {
    if (shape != tensor.shape && tensor.numElements != 1)
      throw ShapeMismatchException(s"Assigned tensor shape '${tensor.shape}' does not match assignee shape '$shape'")
    for ((index, value) <- flattenedIndexIterator zip tensor.entriesIterator)
      setElementAtFlattenedIndex(index, value)(tensor.dataType.supportedType)
    this
  }

  def scalar: dataType.ScalaType = {
    if (numElements != 1)
      throw new IllegalStateException(s"Cannot obtain a scalar value from a non-scalar tensor with shape '$shape'.")
    getElementAtFlattenedIndex(flattenedIndex(Array.fill[Int](shape.rank)(0))) // TODO: Fix this.
  }

  // TODO: !!! Make this return the sub-class tensor type instead.
  def apply(indexers: Indexer*): Tensor = {
    slice(indexers: _*)
    //    if (dataType.byteSize == -1)
    //      throw new IllegalStateException("Cannot index a tensor whose elements have unknown byte size.")
    //    // TODO: Add checks for whether the indexers provided are within bounds.
    //    if ((indexers.length == rank || (indexers.length == 1 && rank == 0)) && indexers.forall(_.isInstanceOf[Index])) {
    //      val index = flattenedIndex(indexers.map(_.asInstanceOf[Index].index).toArray)
    //      dataType.getElementFromBuffer(buffer = buffer, index = index * dataType.byteSize)
    //    } else {
    //      throw InvalidIndexerException(
    //        "Only sequences of single indices that match in length the rank of a tensor, are supported for obtaining the " +
    //            "value of a tensor element.")
    //    }
  }

  // TODO: Make more efficient for contiguous slices.
  def slice(indexers: Indexer*): Tensor = {
    if (shape.rank == 0 && indexers.length == 1
        && indexers.head.isInstanceOf[Index] && indexers.head.asInstanceOf[Index].index == 0) {
      this
    } else {
      val decoded = Indexer.decode(shape, indexers)
      val tensor = newTensor(Shape.fromSeq(decoded._2))
      stridedAssign(tensor, decoded._1, decoded._3, decoded._4, decoded._5)
    }
  }
  // TODO: Use this for creating slices: Buffer.slice().position(sliceStart).limit(sliceSize)

  private[tensors] def newTensor(shape: Shape): Tensor

  private[tensors] def stridedAssign(
      tensor: Tensor, underlyingTensorDimensions: Array[Int], beginOffsets: Array[Int], endOffsets: Array[Int],
      strides: Array[Int]): Tensor = {
    val stridedIndexIterator = order.indexIterator(underlyingTensorDimensions, beginOffsets, endOffsets, strides)
    for ((newIndex, stridedIndex) <- tensor.flattenedIndexIterator zip stridedIndexIterator)
      tensor.setElementAtFlattenedIndex(newIndex, getElementAtFlattenedIndex(stridedIndex))(dataType.supportedType)
    tensor
  }

  def reshape(shape: Shape, copyData: Boolean = true): Tensor

  /** Returns a summary of the contents of this tensor.
    *
    * @param  maxEntries Maximum number of entries to show for each axis/dimension. If the size of an axis exceeds
    *                    `maxEntries`, the output of that axis will be shortened to the first and last three elements.
    *                    Defaults to `6`. Values below `6` are ignored.
    * @return Tensor summary.
    */
  override def summarize(maxEntries: Int = 6): String = {
    def summarize(tensor: Tensor, maxEntries: Int): String =
      tensor.rank match {
        case 0 => tensor.scalar.toString
        case 1 =>
          val slice =
            if (tensor.numElements <= math.max(maxEntries, 6))
              tensor.entriesIterator
            else
              (tensor(0 :: 3).entriesIterator.toSeq :+ "...") ++ tensor(-3 ::).entriesIterator
          slice.mkString("[", ", ", "]")
        case _ =>
          val innerSummary = {
            def summarizeSlice(index: Int) = summarize(tensor.slice(index).reshape(tensor.shape(1 ::)), maxEntries)

            if (tensor.shape(0) <= math.max(maxEntries, 6))
              for (i <- 0 until tensor.shape(0)) yield summarizeSlice(i)
            else {
              val start = for (i <- 0 until 3) yield summarizeSlice(i)
              val end = for (i <- tensor.shape(0) - 3 until tensor.shape(0)) yield summarizeSlice(i)
              (start :+ "...") ++ end
            }
          }
          val padding = " " * (this.rank - tensor.rank + 1)
          val extraLine = if (tensor.rank >= 3) "\n" else ""
          innerSummary.mkString("[", ",\n" + extraLine + padding, "]")
      }
    toString + "\n" + summarize(this, maxEntries)
  }

  override def toString: String = s"$dataType$shape"

  override def equals(that: Any): Boolean = that match {
    case that: Tensor =>
      this.shape == that.shape &&
          this.dataType == that.dataType &&
          this.entriesIterator.zip(that.entriesIterator).forall(p => p._1 == p._2)
    case _ => false
  }

  override def hashCode(): Int = {
    val prime = 31
    var result = 1
    result = prime * result + dataType.hashCode
    result = prime * result + shape.hashCode
    flattenedIndexIterator.foreach(index => result = prime * result + getElementAtFlattenedIndex(index).hashCode)
    result
  }

  def asNumeric: NumericTensor
  def asRealNumeric: RealNumericTensor

  override def toTensor: Tensor = this
  override def toOutput: Output = Basic.constant(this)
}

object Tensor {
  // TODO: [TENSORS] Add constructor methods for numeric tensors and other specific types of tensors.

  def fromSeq[T](values: T*)(implicit evidence: SupportedType[T]): Tensor = {
    val shape = if (values.length > 1) Shape(values.length) else Shape()
    values.head match {
      case _: String =>
        // TODO: !!! Make more efficient.
        val v = values.asInstanceOf[Seq[String]]
        var size = INT64.byteSize * v.length
        var i = 0
        while (i < v.length) {
          size += NativeTensor.getEncodedStringSize(v(i).getBytes(Charset.forName("UTF-8")).length)
          i += 1
        }
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder)
        val tensor = new StringTensor(shape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
        val baseOffset = INT64.byteSize * tensor.numElements
        var byteIndex = 0
        i = 0
        while (i < v.length) {
          val numEncodedBytes = STRING.putElementInBuffer(buffer, baseOffset + byteIndex, v(i))
          INT64.putElementInBuffer(buffer, INT64.byteSize * i, byteIndex.toLong)
          byteIndex += numEncodedBytes
          i += 1
        }
        tensor
      case _ =>
        val tensor = allocate(values.head.dataType, shape)
        val tensorIndexIterator = tensor.flattenedIndexIterator
        values.foreach(value => tensor.setElementAtFlattenedIndex(tensorIndexIterator.next(), value))
        tensor
    }
  }

  def fromSeq[T](dataType: DataType, values: T*)(implicit evidence: SupportedType[T]): Tensor = {
    val shape = if (values.length > 1) Shape(values.length) else Shape()
    dataType match {
      case STRING =>
        val v = values.map(STRING.cast(_)(evidence))
        var size = INT64.byteSize * v.length
        var i = 0
        while (i < v.length) {
          size += NativeTensor.getEncodedStringSize(v(i).getBytes(Charset.forName("UTF-8")).length)
          i += 1
        }
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder)
        val tensor = new StringTensor(shape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
        val baseOffset = INT64.byteSize * tensor.numElements
        var byteIndex = 0
        i = 0
        while (i < v.length) {
          val numEncodedBytes = STRING.putElementInBuffer(buffer, baseOffset + byteIndex, v(i))
          INT64.putElementInBuffer(buffer, INT64.byteSize * i, byteIndex.toLong)
          byteIndex += numEncodedBytes
          i += 1
        }
        tensor
      case _ =>
        val tensor = allocate(dataType, shape)
        val tensorIndexIterator = tensor.flattenedIndexIterator
        values.foreach(value => {
          val castedValue = dataType.cast(value)
          tensor.setElementAtFlattenedIndex(tensorIndexIterator.next(), castedValue)(dataType.supportedType)
        })
        tensor
    }
  }

  def apply(tensors: Tensor*): Tensor = {
    if (tensors.isEmpty)
      throw new IllegalArgumentException("A data type needs to be provided to construct empty tensors.")
    apply(dataType = tensors.map(_.dataType).maxBy(_.priority), tensors: _*)
  }

  def apply(dataType: DataType, tensors: Tensor*): Tensor = {
    // TODO: What about column-major string tensors?
    val shape = if (tensors.nonEmpty) tensors.head.shape else Shape()
    if (tensors.nonEmpty)
      require(tensors.tail.forall(_.shape == shape), "All provided tensor shapes must match.")
    val newShape = if (tensors.nonEmpty) Shape(tensors.length +: shape.asArray: _*) else Shape(0)
    dataType match {
      case STRING =>
        // TODO: Make this more efficient.
        // val numElements = newShape.numElements.get
        var size = 0
        var t = 0
        while (t < tensors.length) {
          size += tensors(t).buffer.capacity() // TODO: This will not work with slices.
          t += 1
        }
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder)
        val tensor = new StringTensor(newShape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
        val baseOffset = INT64.byteSize * tensor.numElements
        var byteIndex = 0
        var elementIndex = 0
        t = 0
        while (t < tensors.length) {
          val otherBaseOffset = tensors(t).numElements * INT64.byteSize
          var i = 0
          while (i < tensors(t).numElements) {
            val otherOffset = otherBaseOffset +
                INT64.getElementFromBuffer(tensors(t).buffer, i * INT64.byteSize).toInt
            val string = STRING.getElementFromBuffer(tensors(t).buffer, otherOffset)
            val numEncodedBytes = STRING.putElementInBuffer(buffer, baseOffset + byteIndex, string)
            INT64.putElementInBuffer(buffer, elementIndex * INT64.byteSize, byteIndex.toLong)
            byteIndex += numEncodedBytes
            elementIndex += 1
            i += 1
          }
          t += 1
        }
        tensor
      case _ =>
        val tensor = allocate(dataType, newShape)
        val newTensorIndexIterator = tensor.flattenedIndexIterator
        tensors.foreach(t => t.flattenedIndexIterator.foreach(index => {
          tensor.setElementAtFlattenedIndex(
            newTensorIndexIterator.next(), t.getElementAtFlattenedIndex(index))(t.dataType.supportedType)
        }))
        tensor
    }
  }

  def fill[T](dataType: DataType, shape: Shape = null)(value: T)(implicit evidence: SupportedType[T]): Tensor = {
    // TODO: Add downcasting warnings.
    val inferredShape = if (shape == null) Shape() else shape
    inferredShape.assertFullyDefined()
    dataType match {
      case STRING =>
        val numStringBytes = value.toString.getBytes(Charset.forName("UTF-8")).length
        val numEncodedBytes = NativeTensor.getEncodedStringSize(numStringBytes)
        val numBytes = inferredShape.numElements * (INT64.byteSize + numEncodedBytes)
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
        val tensor = new StringTensor(inferredShape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
        val baseOffset = INT64.byteSize * tensor.numElements
        var index = 0
        var i = 0
        while (i < tensor.numElements) {
          STRING.putElementInBuffer(buffer, baseOffset + index, STRING.cast(value))
          INT64.putElementInBuffer(buffer, i * INT64.byteSize, index.toLong)
          index += numEncodedBytes
          i += 1
        }
        tensor
      case _ => allocate(dataType = dataType, shape = inferredShape).fill(value)
    }
  }

  // TODO: [TENSOR] Add checks for direct/non-direct byte buffers.

  def fromBuffer(
      dataType: DataType, shape: Shape, buffer: ByteBuffer, copy: Boolean = false,
      order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): Tensor = dataType match {
    case STRING => ??? // TODO: [TENSORS_STRING]
    case d: RealNumericDataType =>
      val bufferCopy = copyBuffer(dataType, shape, buffer, copy, order)
      new RealNumericTensor(dataType = d, shape = shape, buffer = bufferCopy, order)
    case d: NumericDataType =>
      val bufferCopy = copyBuffer(dataType, shape, buffer, copy, order)
      new NumericTensor(dataType = d, shape = shape, buffer = bufferCopy, order)
    case d: FixedSizeDataType =>
      val bufferCopy = copyBuffer(dataType, shape, buffer, copy, order)
      new FixedSizeTensor(dataType = d, shape = shape, buffer = bufferCopy, order)
    case d => throw InvalidDataTypeException(s"Tensors with data type '$d' are not supported on the Scala side.")
  }

  private[tensors] def copyBuffer(
      dataType: DataType, shape: Shape, buffer: ByteBuffer, copy: Boolean = false,
      order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): ByteBuffer = {
    shape.assertFullyDefined()
    val limit = dataType.byteSize * shape.numElements
    if (!copy && buffer.isDirect) {
      val bufferDuplicate = buffer.duplicate
      bufferDuplicate.limit(limit)
      bufferDuplicate
    } else {
      val bufferCopy = ByteBuffer.allocateDirect(buffer.capacity)
      val readOnlyBufferCopy = buffer.asReadOnlyBuffer
      bufferCopy.put(readOnlyBufferCopy)
      bufferCopy.position(buffer.position)
      bufferCopy.limit(limit)
      bufferCopy.order(buffer.order)
      bufferCopy
    }
  }

  private[api] def allocate(
      dataType: DataType, shape: Shape, order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): Tensor = dataType match {
    case STRING => throw new IllegalArgumentException(
      "Cannot pre-allocate string tensors because their size is not known.")
    case d: RealNumericDataType => RealNumericTensor.allocate(d, shape, order)
    case d: NumericDataType => NumericTensor.allocate(d, shape, order)
    case d: FixedSizeDataType => FixedSizeTensor.allocate(d, shape, order)
    case d => throw InvalidDataTypeException(s"Tensors with data type '$d' are not supported on the Scala side.")
  }

  private[api] def fromTFNativeHandle(nativeHandle: Long): Tensor = {
    DataType.fromCValue(NativeTensor.dataType(nativeHandle)).tensorFromTFNativeHandle(nativeHandle)
  }

  private[api] trait Implicits {
    implicit def scalaValueToTensor(value: Boolean): Tensor = Tensor.fill(dataType = BOOLEAN)(value)
    implicit def scalaValueToTensor(value: String): Tensor = Tensor.fill(dataType = STRING)(value)
    implicit def scalaValueToTensor(value: Float): Tensor = Tensor.fill(dataType = FLOAT32)(value)
    implicit def scalaValueToTensor(value: Double): Tensor = Tensor.fill(dataType = FLOAT64)(value)
    implicit def scalaValueToTensor(value: Byte): Tensor = Tensor.fill(dataType = INT8)(value)
    implicit def scalaValueToTensor(value: Short): Tensor = Tensor.fill(dataType = INT16)(value)
    implicit def scalaValueToTensor(value: Int): Tensor = Tensor.fill(dataType = INT32)(value)
    implicit def scalaValueToTensor(value: Long): Tensor = Tensor.fill(dataType = INT64)(value)
    implicit def scalaValueToTensor(value: UShort): Tensor = Tensor.fill(dataType = UINT16)(value)

    implicit def scalaArrayToTensor(value: Array[Boolean]): Tensor = Tensor.fromSeq(value: _*)(BOOLEAN.supportedType)
    // implicit def scalaArrayToTensor(value: Array[String]): Tensor = Tensor.fromSeq(value: _*)(String.supportedType)
    implicit def scalaArrayToTensor(value: Array[Float]): Tensor = Tensor.fromSeq(value: _*)(FLOAT32.supportedType)
    implicit def scalaArrayToTensor(value: Array[Double]): Tensor = Tensor.fromSeq(value: _*)(FLOAT64.supportedType)
    implicit def scalaArrayToTensor(value: Array[Byte]): Tensor = Tensor.fromSeq(value: _*)(INT8.supportedType)
    implicit def scalaArrayToTensor(value: Array[Short]): Tensor = Tensor.fromSeq(value: _*)(INT16.supportedType)
    implicit def scalaArrayToTensor(value: Array[Int]): Tensor = Tensor.fromSeq(value: _*)(INT32.supportedType)
    implicit def scalaArrayToTensor(value: Array[Long]): Tensor = Tensor.fromSeq(value: _*)(INT64.supportedType)
    implicit def scalaArrayToTensor(value: Array[UShort]): Tensor = Tensor.fromSeq(value: _*)(UINT16.supportedType)

    implicit def tensorToNumeric(tensor: Tensor): NumericTensor = tensor.asNumeric
    implicit def tensorToRealNumeric(tensor: Tensor): RealNumericTensor = tensor.asRealNumeric
  }

  private[api] object Implicits extends Implicits

  //  def apply[T: DataType.SupportedScalaTypes#Member](values: T*): Tensor = {
  //    val valueDataType: DataType = DataType.dataTypeOf(values.head)
  //    val shape: Shape = Shape(values.length)
  //    if (valueDataType != DataType.String) {
  //      null
  //    } else {
  //      // TODO: Support String tensors.
  //      throw new UnsupportedOperationException(
  //        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
  //            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
  //    }
  //  }
  //
  //  /** Creates a [[Tensor]].
  //    *
  //    * The resulting tensor is populated with values of type `dataType`, as specified by the arguments `value` and
  //    * (optionally) `shape` (see examples below).
  //    *
  //    * The argument `value` can be a constant value, or an array (potentially multi-dimensional) with elements of type
  //    * `dataType`. If `value` is a one-dimensional array, then its length should be less than or equal to the number of
  //    * elements implied by the `shape` argument (if specified). In the case where the array length is less than the
  //    * number of elements specified by `shape`, the last element in the array will be used to fill the remaining entries.
  //    *
  //    * The argument `dataType` is optional. If not specified, then its value is inferred from the type of `value`.
  //    *
  //    * The argument `shape` is optional. If present, it specifies the dimensions of the resulting tensor. If not present,
  //    * the shape of `value` is used.
  //    *
  //    * **IMPORTANT NOTE** The data type argument and the shape arguments are not currently being used.
  //    *
  //    * @param  value       A constant value of data type `dataType`.
  //    * @param  dataType    Data type of the resulting tensor. If not provided, its value will be inferred from the type
  //    *                     of `value`.
  //    * @param  shape       Shape of the resulting tensor.
  //    * @param  verifyShape If `true` and `shape` is not `null`, then the shape of `value` will be verified (i.e., checked
  //    *                     to see if it is equal to the provided shape.
  //    * @return Created tensor.
  //    * @throws InvalidShapeException If `shape != null`, `verifyShape == true`, and the shape of values does not match
  //    *                               the provided `shape`.
  //    */
  //  def create(value: Any, dataType: DataType = null, shape: Shape = null, verifyShape: Boolean = false): Tensor = {
  //    val valueDataType: DataType = DataType.dataTypeOf(value)
  //    val inferredDataType: DataType = if (dataType == null) valueDataType else dataType
  //    val inferredShape: Shape = if (shape == null) Tensor.shape(value) else shape
  //    // TODO: !!! Fix this so that it actually does verify the shape and the data type and does appropriate type casts.
  //    if (inferredDataType != DataType.String) {
  //      val numElements = inferredShape.numElements.get
  //      val byteSize = inferredDataType.byteSize * numElements
  //      val nativeHandle = NativeTensor.allocate(inferredDataType.cValue, inferredShape.asArray, byteSize)
  //      if (inferredDataType != valueDataType) {
  //        val tensor: Tensor = allocateForBuffer(dataType, inferredShape, numElements)
  //        castAndWriteTo(tensor.buffer, value, dataType)
  //        tensor
  //      } else {
  //        NativeTensor.setValue(nativeHandle, value)
  //        Tensor(dataType = inferredDataType, shape = inferredShape, nativeHandle = nativeHandle)
  //      }
  //    } else if (inferredShape.rank != 0) {
  //      // TODO: Support String tensors.
  //      throw new UnsupportedOperationException(
  //        s"Non-scalar DataType.String tensors are not supported yet (version ${TensorFlow.version}). Please file a " +
  //            s"feature request at https://github.com/tensorflow/tensorflow/issues/new.")
  //    } else {
  //      val nativeHandle = NativeTensor.allocateScalarBytes(value.asInstanceOf[Array[Byte]])
  //      Tensor(dataType = inferredDataType, shape = inferredShape, nativeHandle = nativeHandle)
  //    }
  //  }
  //
  //  private[this] def castAndWriteTo(buffer: ByteBuffer, value: Any, dataType: DataType): Unit = {
  //    // TODO: May be doable more efficiently.
  //    def writeToHelper(buffer: ByteBuffer, bufferIndex: Int, value: Any, dataType: DataType): Int = {
  //      value match {
  //        case array: Array[_] =>
  //          var bytesWritten = 0
  //          var i = 0
  //          while (i < array.length) {
  //            bytesWritten += writeToHelper(buffer, bufferIndex + bytesWritten, array(i), dataType)
  //            i += 1
  //          }
  //          bytesWritten
  //        case scalar =>
  //          dataType.putElementInBuffer(buffer, bufferIndex, dataType.cast(scalar))
  //          dataType.byteSize
  //      }
  //    }
  //    writeToHelper(buffer, 0, value, dataType)
  //  }
  //
  //  def create(shape: Shape, data: FloatBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Float32, shape, data.remaining())
  //    tensor.buffer.asFloatBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: DoubleBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Float64, shape, data.remaining())
  //    tensor.buffer.asDoubleBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: IntBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Int32, shape, data.remaining())
  //    tensor.buffer.asIntBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(shape: Shape, data: LongBuffer): Tensor = {
  //    val tensor: Tensor = allocateForBuffer(DataType.Int64, shape, data.remaining())
  //    tensor.buffer.asLongBuffer().put(data)
  //    tensor
  //  }
  //
  //  def create(dataType: DataType, shape: Shape, data: ByteBuffer): Tensor = {
  //    val numRemaining: Int = {
  //      if (dataType != DataType.String) {
  //        if (data.remaining() % dataType.byteSize != 0)
  //          throw new IllegalArgumentException(s"A byte buffer with ${data.remaining()} bytes is not compatible with a " +
  //                                                 s"${dataType.toString} Tensor (${dataType.byteSize} bytes/element).")
  //        data.remaining() / dataType.byteSize
  //      } else {
  //        data.remaining()
  //      }
  //    }
  //    val tensor: Tensor = allocateForBuffer(dataType, shape, numRemaining)
  //    tensor.buffer.put(data)
  //    tensor
  //  }
  //  // Helper function to allocate a Tensor for the create() methods that create a Tensor from
  //  // a java.nio.Buffer.
  //  private def allocateForBuffer(dataType: DataType, shape: Shape, numBuffered: Int): Tensor = {
  //    val size: Long = shape.numElements.get
  //    val numBytes: Long = {
  //      if (dataType != DataType.String) {
  //        if (numBuffered != size)
  //          throw incompatibleBufferException(numBuffered, shape)
  //        size * dataType.byteSize
  //      } else {
  //        // DataType.String tensor encoded in a ByteBuffer.
  //        numBuffered
  //      }
  //    }
  //    val nativeHandle: Long = NativeTensor.allocate(dataType.cValue, shape.asArray.clone(), numBytes)
  //    Tensor(dataType = dataType, shape = shape, nativeHandle = nativeHandle)
  //  }
  //
  //  private def incompatibleBufferException(buffer: Buffer, dataType: DataType): IllegalArgumentException = {
  //    new IllegalArgumentException(s"Cannot use ${buffer.getClass.getName} with a Tensor of type $dataType.")
  //  }
  //
  //  private def incompatibleBufferException(numElements: Int, shape: Shape): IllegalArgumentException = {
  //    new IllegalArgumentException(
  //      s"A buffer with $numElements elements is not compatible with a Tensor with shape '$shape'.")
  //  }
  //
  //  private def rank(value: Any): Int = {
  //    value match {
  //      // Array[Byte] is a DataType.STRING scalar.
  //      case _: Array[Byte] => 0
  //      case value: Array[_] => 1 + rank(value(0))
  //      case _ => 0
  //    }
  //  }
  //
  //  private def shape(value: Any): Shape = {
  //    def fillShape(value: Any, axis: Int, shape: Array[Long]): Unit = {
  //      if (shape != null && axis != shape.length) {
  //        if (shape(axis) == 0) {
  //          value match {
  //            case value: Array[_] => shape(axis) = value.length
  //            case _ => shape(axis) = 1
  //          }
  //        } else {
  //          val mismatchedShape = value match {
  //            case value: Array[_] => (shape(axis) != value.length, value.length)
  //            case _ => (shape(axis) != 1, 1)
  //          }
  //          if (mismatchedShape._1)
  //            throw new IllegalArgumentException(
  //              s"Mismatched lengths (${shape(axis)} and ${mismatchedShape._2}) for dimension $axis.")
  //        }
  //        value match {
  //          case value: Array[_] =>
  //            var i = 0
  //            while (i < value.length) {
  //              fillShape(value(i), axis + 1, shape)
  //              i += 1
  //            }
  //        }
  //      }
  //    }
  //
  //    val shapeArray = Array.ofDim[Long](rank(value))
  //    fillShape(value = value, axis = 0, shape = shapeArray)
  //    Shape.fromSeq(shapeArray)
  //  }
}
