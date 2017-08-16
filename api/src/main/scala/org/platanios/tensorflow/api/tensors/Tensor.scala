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

import org.platanios.tensorflow.api.core.{Index, Indexer, Shape}
import org.platanios.tensorflow.api.core.Indexer.Implicits._
import org.platanios.tensorflow.api.core.exception.ShapeMismatchException
import org.platanios.tensorflow.api.ops.{Basic, Output, OutputConvertible}
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
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
case class Tensor private[tensors](
    dataType: DataType, shape: Shape, buffer: ByteBuffer,
    order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER) extends OutputConvertible {
  require(shape.isFullyDefined, s"The shape of a Tensor object must be fully defined. Shape '$shape' is not.")
  // require(shape.numElements > 0, "Empty tensors are not supported in the TensorFlow Scala API.")

  def rank: Int = shape.rank
  def numElements: Int = shape.numElements

  private[api] def flattenedIndex(indices: Array[Int]): Int = order.index(shape.asArray, indices)
  private[api] def flattenedIndexIterator: Iterator[Int] = order.indexIterator(shape.asArray)

  private[api] def setElementAtFlattenedIndex[T](
      index: Int, value: T)(implicit ev: SupportedType[T]): Tensor = dataType match {
    case STRING => ???
    case _ =>
      dataType.putElementInBuffer(buffer, index * dataType.byteSize, dataType.cast(value))
      this
  }

  private[api] def getElementAtFlattenedIndex(index: Int): dataType.ScalaType = dataType match {
    case STRING =>
      val offset = INT64.byteSize * numElements + INT64.getElementFromBuffer(buffer, index * INT64.byteSize).toInt
      dataType.getElementFromBuffer(buffer, offset)
    case _ => dataType.getElementFromBuffer(buffer, index * dataType.byteSize)
  }

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

  def fill[T](value: T)(implicit evidence: SupportedType[T]): Tensor = dataType match {
    case STRING => ???
    case _ =>
      val castedValue = dataType.cast(value)
      for (index <- flattenedIndexIterator)
        dataType.putElementInBuffer(buffer = buffer, index = index * dataType.byteSize, element = castedValue)
      this
  }

  // TODO: Find a way to add this method for performance benefits.
  // def set(value: SupportedScalaType): Tensor = fill(value)

  def set(tensor: Tensor): Tensor = {
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
      dataType match {
        case STRING => ???
        case _ =>
          val decoded = Indexer.decode(shape, indexers)
          val tensor = Tensor.allocate(dataType, Shape.fromSeq(decoded._2), order)
          stridedAssign(tensor, decoded._1, decoded._3, decoded._4, decoded._5)
      }
    }
  }

  // TODO: Use this for creating slices: Buffer.slice().position(sliceStart).limit(sliceSize)

  private[tensors] def stridedAssign(
      tensor: Tensor, underlyingTensorDimensions: Array[Int], beginOffsets: Array[Int], endOffsets: Array[Int],
      strides: Array[Int]): Tensor = {
    val stridedIndexIterator = order.indexIterator(underlyingTensorDimensions, beginOffsets, endOffsets, strides)
    for ((newIndex, stridedIndex) <- tensor.flattenedIndexIterator zip stridedIndexIterator)
      tensor.setElementAtFlattenedIndex(newIndex, getElementAtFlattenedIndex(stridedIndex))(dataType.supportedType)
    tensor
  }

  def reshape(shape: Shape, copyData: Boolean = true): Tensor = {
    val newShape = this.shape.reshape(shape)
    if (copyData) {
      dataType match {
        case STRING => ???
        case _ => Tensor(dataType, newShape, Tensor.copyBuffer(dataType, newShape, buffer, copy = true, order), order)
      }
    } else {
      Tensor(dataType, newShape, buffer, order)
    }
  }

  /** Returns a summary of the contents of this tensor.
    *
    * @param  maxEntries Maximum number of entries to show for each axis/dimension. If the size of an axis exceeds
    *                    `maxEntries`, the output of that axis will be shortened to the first and last three elements.
    *                    Defaults to `6`. Values below `6` are ignored.
    * @return Tensor summary.
    */
  def summarize(maxEntries: Int = 6): String = {
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

  override def toOutput: Output = Basic.constant(this)

  private[api] def nativeView: Tensor.NativeView = {
    if (order != RowMajorOrder)
      throw new IllegalArgumentException("Only row-major tensors can be used in the TensorFlow native library.")
    Tensor.NativeView(NativeTensor.fromBuffer(buffer, dataType.cValue, shape.asArray.map(_.toLong), buffer.capacity()))
  }
}

object Tensor {
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
        val tensor = Tensor(STRING, newShape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
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

  def fill[T](dataType: DataType, shape: Shape = null)(value: T)(implicit ev: SupportedType[T]): Tensor = {
    // TODO: Add downcasting warnings.
    val inferredShape = if (shape == null) Shape() else shape
    inferredShape.assertFullyDefined()
    dataType match {
      case STRING =>
        val numStringBytes = value.toString.getBytes(Charset.forName("UTF-8")).length
        val numEncodedBytes = NativeTensor.getEncodedStringSize(numStringBytes)
        val numBytes = inferredShape.numElements * (INT64.byteSize + numEncodedBytes)
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
        val tensor = Tensor(STRING, inferredShape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
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

  def fromSeq[T](values: T*)(implicit ev: SupportedType[T]): Tensor = {
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
        val tensor = Tensor(STRING, shape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
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
        val tensor = allocate(ev.dataType, shape)
        val tensorIndexIterator = tensor.flattenedIndexIterator
        values.foreach(value => tensor.setElementAtFlattenedIndex(tensorIndexIterator.next(), value))
        tensor
    }
  }

  def fromSeq[T](dataType: DataType, values: T*)(implicit ev: SupportedType[T]): Tensor = {
    val shape = if (values.length > 1) Shape(values.length) else Shape()
    dataType match {
      case STRING =>
        val v = values.map(STRING.cast(_)(ev))
        var size = INT64.byteSize * v.length
        var i = 0
        while (i < v.length) {
          size += NativeTensor.getEncodedStringSize(v(i).getBytes(Charset.forName("UTF-8")).length)
          i += 1
        }
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder)
        val tensor = Tensor(STRING, shape, buffer, DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
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

  private[api] def allocate(
      dataType: DataType, shape: Shape, order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): Tensor = dataType match {
    case STRING => throw new IllegalArgumentException(
      "Cannot pre-allocate string tensors because their size is not known.")
    case _ =>
      shape.assertFullyDefined()
      val numBytes: Int = dataType.byteSize * shape.numElements
      val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
      Tensor(dataType = dataType, shape = shape, buffer = buffer, order = order)
  }

  private[api] def fromTFNativeHandle(nativeHandle: Long): Tensor = {
    val dataType = DataType.fromCValue(NativeTensor.dataType(nativeHandle))
    val tensor = Tensor(
      dataType = dataType,
      shape = Shape.fromSeq(NativeTensor.shape(nativeHandle).map(_.toInt)),
      buffer = NativeTensor.buffer(nativeHandle).order(ByteOrder.nativeOrder),
      order = RowMajorOrder)
    // Keep track of references in the Scala side and notify the native library when the tensor is not referenced
    // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
    // potential memory leak.
    Disposer.add(tensor, () => NativeTensor.delete(nativeHandle))
    tensor
  }

  // TODO: [TENSOR] Add checks for direct/non-direct byte buffers.

  def fromBuffer(
      dataType: DataType, shape: Shape, buffer: ByteBuffer, copy: Boolean = false,
      order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): Tensor = {
    val bufferCopy = copyBuffer(dataType, shape, buffer, copy, order)
    Tensor(dataType = dataType, shape = shape, buffer = bufferCopy, order)
  }

  private[tensors] def copyBuffer(
      dataType: DataType, shape: Shape, buffer: ByteBuffer, copy: Boolean = false,
      order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): ByteBuffer = {
    shape.assertFullyDefined()
    val limit = dataType.byteSize * shape.numElements
    if (!copy && buffer.isDirect) {
      val bufferDuplicate = buffer.duplicate
      bufferDuplicate.limit(limit)
      bufferDuplicate.slice()
      bufferDuplicate
    } else {
      val bufferCopy = ByteBuffer.allocateDirect(limit)
      val readOnlyBufferCopy = buffer.asReadOnlyBuffer
      bufferCopy.put(readOnlyBufferCopy)
      bufferCopy.limit(limit)
      bufferCopy.order(buffer.order)
      bufferCopy
    }
  }

  private[tensors] trait Implicits {
    implicit def scalaValueToTensor(value: Boolean): Tensor = Tensor.fill(dataType = BOOLEAN)(value)
    implicit def scalaValueToTensor(value: String): Tensor = Tensor.fill(dataType = STRING)(value)
    implicit def scalaValueToTensor(value: Float): Tensor = Tensor.fill(dataType = FLOAT32)(value)
    implicit def scalaValueToTensor(value: Double): Tensor = Tensor.fill(dataType = FLOAT64)(value)
    implicit def scalaValueToTensor(value: Byte): Tensor = Tensor.fill(dataType = INT8)(value)
    implicit def scalaValueToTensor(value: Short): Tensor = Tensor.fill(dataType = INT16)(value)
    implicit def scalaValueToTensor(value: Int): Tensor = Tensor.fill(dataType = INT32)(value)
    implicit def scalaValueToTensor(value: Long): Tensor = Tensor.fill(dataType = INT64)(value)
    implicit def scalaValueToTensor(value: UShort): Tensor = Tensor.fill(dataType = UINT16)(value)

    implicit def scalaArrayToTensor(value: Array[Boolean]): Tensor = Tensor.fromSeq(value: _*)
    // implicit def scalaArrayToTensor(value: Array[String]): Tensor = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Float]): Tensor = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Double]): Tensor = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Byte]): Tensor = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Short]): Tensor = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Int]): Tensor = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[Long]): Tensor = Tensor.fromSeq(value: _*)
    implicit def scalaArrayToTensor(value: Array[UShort]): Tensor = Tensor.fromSeq(value: _*)
  }

  private[api] final case class NativeView(private[api] var nativeHandle: Long) extends Closeable {
    override def close(): Unit = {
      if (nativeHandle != 0) {
        NativeTensor.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}
