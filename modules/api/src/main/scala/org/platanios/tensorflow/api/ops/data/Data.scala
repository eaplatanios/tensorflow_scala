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

package org.platanios.tensorflow.api.ops.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.TensorToOutput
import org.platanios.tensorflow.api.io.{CompressionType, NoCompression}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{Variant, INT64, STRING}

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable
import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
trait Data {
  //region Dataset Constructors

  /** Creates a dataset with a single element.
    *
    * @param  data   Data representing the single element.
    * @param  name   Name for this dataset.
    * @param  evData Type class instance for the element type.
    * @tparam T Tensor type of the element.
    * @return Created dataset.
    */
  def datasetFromOutputs[T](
      data: T,
      name: String = "TensorDataset"
  )(implicit evData: SupportedData[T]): Dataset[T] = {
    val datasetName = name
    val ev = evData
    new Dataset[T] {
      override val name  : String           = datasetName
      override val evData: SupportedData[T] = ev

      override def createHandle(): Output[Variant] = {
        val flatOutputs = evData.outputs(data)
        Op.Builder[Seq[Output[Any]], Output[Variant]](
          opType = "TensorDataset",
          name = name,
          input = flatOutputs
        ).setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = evData.dataType(data)
      override def outputShapes: evData.S = evData.shape(data)
    }
  }

  /** Creates a dataset with a single element.
    *
    * @param  data   Data representing the single element.
    * @param  name   Name for this dataset.
    * @param  evData Type class instance for the element type.
    * @tparam T Tensor type of the element.
    * @tparam O Tensor type of the element (symbolic equivalent of `T`).
    * @return Created dataset.
    */
  def datasetFromTensors[T, O](
      data: T,
      name: String = "TensorDataset"
  )(implicit
      evTensorToOutput: TensorToOutput.Aux[T, O],
      evData: SupportedData[O]
  ): Dataset[O] = {
    val outputData = Op.nameScope(s"$name/TensorToOutput") {
      evTensorToOutput.toOutput(data)
    }
    datasetFromOutputs(outputData)
  }

  /** Creates a dataset with slices from the nested structure of tensors (i.e., a [[SupportedData]]-supported type).
    * The slices are taken along the first axis of each tensor in the nested structure.
    *
    * @param  data   Data representing the elements of this dataset.
    * @param  name   Name for this dataset.
    * @param  evData Type class instance for the element type.
    * @tparam T Tensor type of the element.
    * @return Created dataset.
    */
  def datasetFromOutputSlices[T](
      data: T,
      name: String = "TensorSlicesDataset"
  )(implicit evData: SupportedData[T]): Dataset[T] = {
    val datasetName = name
    val ev = evData
    new Dataset[T] {
      override val name  : String           = datasetName
      override val evData: SupportedData[T] = ev

      override def createHandle(): Output[Variant] = {
        val flatOutputs = evData.outputs(data)
        Op.Builder[Seq[Output[Any]], Output[Variant]](
          opType = "TensorSliceDataset",
          name = name,
          input = flatOutputs
        ).setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = {
        evData.dataType(data)
      }

      override def outputShapes: evData.S = {
        val flatShapes = evData.shapes(evData.shape(data))
        evData.decodeShape(
          outputDataTypes,
          flatShapes.map(s => if (s.rank > 1) s(1 ::) else Shape.scalar()))._1
      }
    }
  }

  /** Creates a dataset with slices from the nested structure of tensors (i.e., a [[SupportedData]]-supported type).
    * The slices are taken along the first axis of each tensor in the nested structure.
    *
    * @param  data   Data representing the elements of this dataset.
    * @param  name   Name for this dataset.
    * @param  evData Type class instance for the element type.
    * @tparam T Tensor type of the element.
    * @tparam O Tensor type of the element (symbolic equivalent of `T`).
    * @return Created dataset.
    */
  def datasetFromTensorSlices[T, O](
      data: T,
      name: String = "TensorSlicesDataset"
  )(implicit
      evTensorToOutput: TensorToOutput.Aux[T, O],
      evData: SupportedData[O]
  ): Dataset[O] = {
    val outputData = Op.nameScope(s"$name/TensorToOutput") {
      evTensorToOutput.toOutput(data)
    }
    datasetFromOutputSlices(outputData)
  }

  // TODO: [DATA] Add dynamic version (i.e., passing in `Output`s) for the `fromRange`, `fromFixedLengthRecordFiles`, and `fromTextFiles` datasets.

  /** Creates a new dataset that contains a range of values.
    *
    * @param  start Starting value of the number sequence.
    * @param  limit Ending value (exclusive) of the number sequence.
    * @param  delta Difference between consecutive numbers in the sequence.
    * @param  name  Name for this dataset.
    * @return Created dataset.
    */
  def datasetFromRange(
      start: Long,
      limit: Long,
      delta: Long = 1L,
      name: String = "RangeDataset"
  ): Dataset[Output[Long]] = {
    val datasetName = name
    new Dataset[Output[Long]] {
      override val name  : String                      = datasetName
      override val evData: SupportedData[Output[Long]] = implicitly[SupportedData[Output[Long]]]

      override def createHandle(): Output[Variant] = {
        Op.Builder[(Output[Long], Output[Long], Output[Long]), Output[Variant]](
          opType = "RangeDataset",
          name = name,
          input = (
              Basic.constant(start, name = s"$name/Start"),
              Basic.constant(limit, name = s"$name/Limit"),
              Basic.constant(delta, name = s"$name/Delta"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = INT64.asInstanceOf[evData.D]
      override def outputShapes: evData.S = Shape.scalar().asInstanceOf[evData.S]
    }
  }

  /** Creates a dataset with elements read from binary files.
    *
    * @param  filenames      Names of the files to be read.
    * @param  recordNumBytes Number of bytes in the record.
    * @param  headerNumBytes Number of bytes in the header (i.e., the number of bytes to skip at the start of a file).
    * @param  footerNumBytes Number of bytes in the footer (i.e., the number of bytes to skip at the end of a file).
    * @param  bufferSize     Number of bytes to buffer while reading from files.
    * @param  name           Name for this dataset.
    * @return Created dataset.
    */
  def datasetFromFixedLengthRecordFiles(
      filenames: Seq[String],
      recordNumBytes: Long,
      headerNumBytes: Long,
      footerNumBytes: Long,
      bufferSize: Long = 256 * 1024,
      name: String = "FixedLengthRecordDataset"
  ): Dataset[Output[String]] = {
    val datasetName = name
    new Dataset[Output[String]] {
      override val name  : String                        = datasetName
      override val evData: SupportedData[Output[String]] = implicitly[SupportedData[Output[String]]]

      override def createHandle(): Output[Variant] = {
        Op.Builder[(Output[String], Output[Long], Output[Long], Output[Long], Output[Long]), Output[Variant]](
          opType = "FixedLengthRecordDataset",
          name = name,
          input = (
              Basic.constant(filenames, name = s"$name/FileNames"),
              Basic.constant(recordNumBytes, name = s"$name/RecordNumBytes"),
              Basic.constant(headerNumBytes, name = s"$name/HeaderNumBytes"),
              Basic.constant(footerNumBytes, name = s"$name/FooterNumBytes"),
              Basic.constant(bufferSize, name = s"$name/BufferSize"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = STRING.asInstanceOf[evData.D]
      override def outputShapes: evData.S = Shape.scalar().asInstanceOf[evData.S]
    }
  }

  /** Creates a dataset with elements read from text files (each line in each file corresponds to an element).
    *
    * **Note:** New-line characters are stripped from the output.
    *
    * @param  filenames       Names of the files to be read.
    * @param  compressionType Compression type for the files.
    * @param  bufferSize      Number of bytes to buffer while reading from files.
    * @param  name            Name for this dataset.
    * @return Created dataset.
    */
  def datasetFromTextFiles(
      filenames: Seq[String],
      compressionType: CompressionType = NoCompression,
      bufferSize: Long = 256 * 1024,
      name: String = "TextLinesDataset"
  ): Dataset[Output[String]] = {
    val datasetName = name
    new Dataset[Output[String]] {
      override val name  : String                        = datasetName
      override val evData: SupportedData[Output[String]] = implicitly[SupportedData[Output[String]]]

      override def createHandle(): Output[Variant] = {
        Op.Builder[(Output[String], Output[String], Output[Long]), Output[Variant]](
          opType = "TextLineDataset",
          name = name,
          input = (
              Basic.constant(filenames, name = s"$name/FileNames"),
              Basic.constant(compressionType.name, name = s"$name/CompressionType"),
              Basic.constant(bufferSize, name = s"$name/BufferSize"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = STRING.asInstanceOf[evData.D]
      override def outputShapes: evData.S = Shape.scalar().asInstanceOf[evData.S]
    }
  }

  /** Creates a dataset with elements read from files that contain TensorFlow records.
    *
    * @param  filenames       Names of the files to be read.
    * @param  compressionType Compression type for the files.
    * @param  bufferSize      Number of bytes to buffer while reading from files.
    * @param  name            Name for this dataset.
    * @return Created dataset.
    */
  def datasetFromTFRecordFiles(
      filenames: Seq[String],
      compressionType: CompressionType = NoCompression,
      bufferSize: Long = 256 * 1024,
      name: String = "TFRecordsDataset"
  ): Dataset[Output[String]] = {
    val datasetName = name
    new Dataset[Output[String]] {
      override val name  : String                        = datasetName
      override val evData: SupportedData[Output[String]] = implicitly[SupportedData[Output[String]]]

      override def createHandle(): Output[Variant] = {
        Op.Builder[(Output[String], Output[String], Output[Long]), Output[Variant]](
          opType = "TFRecordDataset",
          name = name,
          input = (
              Basic.constant(filenames, name = s"$name/FileNames"),
              Basic.constant(compressionType.name, name = s"$name/CompressionType"),
              Basic.constant(bufferSize, name = s"$name/BufferSize"))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = STRING.asInstanceOf[evData.D]
      override def outputShapes: evData.S = Shape.scalar().asInstanceOf[evData.S]
    }
  }

  /** Stores outstanding iterators created from a Scala iterable.
    *
    * This class keeps track of potentially multiple iterators that may have been created from an iterable, e.g., in the
    * case that the dataset is repeated, or nested within a parallel computation.
    *
    * @param  generator Function that generates an iterable containing dataset elements.
    */
  private case class GeneratorState[T](generator: () => Iterable[T]) {
    private val _nextId   = new AtomicLong(0)
    private val iterators = mutable.Map.empty[Long, scala.Iterator[T]]

    private[Data] def nextId: Long = {
      _nextId.getAndIncrement()
    }

    private[Data] def getIterator(id: Long): scala.Iterator[T] = {
      iterators.getOrElseUpdate(id, generator().iterator)
    }

    private[Data] def deleteIterator(id: Long): Unit = {
      iterators.remove(id)
    }
  }

  /** Creates a dataset whose elements are generated by Scala iterables over dataset elements.
    *
    * The `generator` argument must be a function that takes no arguments and returns an iterable over dataset
    * elements. The elements contained in that iterable must be compatible with the provided `outputDataType` and
    * `outputShape` arguments.
    *
    * For example:
    * {{{
    *   // TODO: [DATA] !!! Improve this example with variable shapes -- like in the Python API.
    *   val generator = () => Range(0, 10).map(Tensor(_))
    *   val dataset = Dataset.fromGenerator(generator, INT32, Shape.scalar())
    *   val value = dataset.createOneShotIterator().next()
    *   session.run(value) ==> 0
    *   session.run(value) ==> 1
    * }}}
    *
    * @param  generator      Function that takes no arguments and returns an [[Iterable]] over dataset elements.
    * @param  outputDataType Output data type structure for the tensor structure of the generated [[Iterable]] elements.
    * @param  outputShape    Output shape structure for the tensor structure of the generated [[Iterable]] elements.
    * @return Constructed dataset.
    */
  def datasetFromGenerator[T, O, D, S](
      generator: () => Iterable[T],
      outputDataType: D,
      outputShape: S = null
  )(implicit
      evTensorToOutput: TensorToOutput.Aux[T, O],
      evData: SupportedData.Aux[O, D, S],
      evFunctionOutput: Function.ArgType[O]
  ): Dataset[O] = {
    val outputShapeWithDefault: S = {
      if (outputShape != null) {
        outputShape
      } else {
        evData.decodeShape(
          outputDataType,
          Seq.fill(evData.size(outputDataType))(Shape.unknown()))._1
      }
    }

    val flatDataTypes = evData.dataTypes(outputDataType)
    val flatShapes = evData.shapes(outputShapeWithDefault)
    val generatorState = GeneratorState(generator)

    /** Creates an op that generates the next element from iterator with ID, `iteratorId`.
      *
      * We map this function across an infinite repetition of the `iteratorId`, and throw an `OutOfRange` to terminate
      * the iteration.
      *
      * @param  iteratorId Scalar tensor whose value uniquely identifies the iterator in the internal
      *                    generator state, from which to generate an element.
      * @return Created op outputs structured according to the output data type of this dataset.
      */
    def generatorMapFn(iteratorId: Output[Long]): O = {
      /** Scala callback function that will be called to invoke the iterator. */
      @throws[OutOfRangeException]
      def generatorScalaCallback(iteratorId: Tensor[Long]): Seq[Tensor[Any]] = {
        val iterator = generatorState.getIterator(iteratorId.scalar)
        val element = {
          if (iterator.hasNext)
            iterator.next()
          else
            throw OutOfRangeException("The iterator does not contain any more elements.")
        }
        val flatTensors = evTensorToOutput.tensors(element)
        // Additional type and shape checking to ensure that the components of the generated element match the
        // output data types and output shapes arguments.
        (flatTensors, flatDataTypes, flatShapes).zipped.foreach((tensor, dataType, shape) => {
          if (tensor.dataType != dataType)
            throw InvalidDataTypeException(
              s"The generator yielded an element of type ${tensor.dataType} " +
                  s"where an element of type $dataType was expected.")
          if (!tensor.shape.isCompatibleWith(shape))
            throw InvalidShapeException(
              s"The generator yielded an element with shape ${tensor.shape} " +
                  s"where an element with shape $shape was expected.")
        })
        flatTensors
      }

      val flatValues = Callback.callback(generatorScalaCallback, iteratorId, flatDataTypes, stateful = true)
      // The Scala callback op drops the inferred shapes, so we add them back in here.
      if (outputShape != null) {
        flatValues.zip(flatShapes).foreach(p => {
          if (!p._1.shape.isCompatibleWith(p._2)) {
            throw new IllegalArgumentException(
              s"Generator output shape ${p._1.shape} is not compatible with provided shape ${p._2}.")
          } else {
            p._1.setShape(p._2)
          }
        })
      }
      evData.decodeOutput(outputDataType, flatValues)._1
    }

    /** Associates each traversal of the provided `generator` with a unique iterator ID. */
    def flatMapFn(iteratorId: Output[Long]): Dataset[O] = {
      // First, generate an infinite dataset containing the iterator ID repeated forever. Then, map using the
      // `generatorMapFn`, which gets the next element from the iterator with the relevant ID, and throws an
      // IndexOutOfBoundsException when that iterator contains no more elements.
      datasetFromOutputs(iteratorId).repeat().map(generatorMapFn)
    }

    // A single-element dataset that, each time it is evaluated, contains a freshly-generated and unique (for the
    // returned dataset) INT64 ID that will be used to identify the appropriate Scala state, which is encapsulated in
    // the internal generator state, and captured in the provided callback function. The ID disambiguates between
    // multiple concurrently existing iterators.
    val idDataset = datasetFromTensors(Tensor(0L)).map((_: Output[Long]) => {
      Callback.callback((_: Unit) => {
        Tensor(generatorState.nextId)
      }, (), INT64, stateful = true)
    })

    // A dataset that contains all of the elements generated by a single iterator created from the provided generator,
    // identified by the iterator ID contained in `idDataset`. Lifting the iteration into a `flatMap` here enables
    // multiple repetitions and/or nested versions of the returned dataset to be created, because it forces the
    // generation of a new ID for each version.
    idDataset.flatMap(flatMapFn)
  }

  //endregion Dataset Constructors
}

object Data extends Data
