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

package org.platanios.tensorflow.tpu

import org.platanios.tensorflow.api._

/** Helper object that contains TPU specific ops.
  *
  * @author Emmanouil Antonios Platanios
  */
private[tpu] object Ops {
  /** Creates an op that sums inputs across replicated TPU instances. Each instance supplies its own input, and the
    * output of each is the sum of all the inputs.
    *
    * @param  input Local input to the sum.
    * @param  name  Name for the created op.
    * @return Created op output, which is the sum of all the distributed inputs.
    */
  def crossReplicaSum(input: Output, name: String = "CrossReplicaSum"): Output = {
    if (!Ops.SUPPORTED_CROSS_REPLICA_SUM_DATA_TYPES.contains(input.dataType))
      throw tf.InvalidDataTypeException(
        s"${input.dataType} is not a supported TPU cross-replica-sum data type. " +
            s"The supported data types are: ${Ops.SUPPORTED_CROSS_REPLICA_SUM_DATA_TYPES.mkString(", ")}.")
    Op.Builder("CrossReplicaSum", name)
        .addInput(input)
        .build().outputs(0)
  }

  /** Creates an op which feeds a single tensor value into the computation.
    *
    * @param  value         Tensor to feed into the computation.
    * @param  deviceOrdinal Specified the TPU device to use. This should be `-1` when the op is running on a TPU device,
    *                       and `>= 0` when the op is running on the CPU device.
    * @param  name          Name for the created op.
    * @return Created op.
    * @throws tf.InvalidDataTypeException If the data type of `value` is not a supported TPU data type.
    */
  @throws[tf.InvalidDataTypeException]
  def infeedEnqueue(
      value: Output,
      deviceOrdinal: Int = -1,
      name: String = "InfeedEnqueue"
  ): Op = {
    if (!Ops.SUPPORTED_INFEED_DATA_TYPES.contains(value.dataType))
      throw tf.InvalidDataTypeException(
        s"${value.dataType} is not a supported TPU infeed data type. " +
            s"The supported data types are: ${Ops.SUPPORTED_INFEED_DATA_TYPES.mkString(", ")}.")
    Op.Builder("InfeedEnqueue", name)
        .addInput(value)
        .setAttribute("dtype", value.dataType)
        .setAttribute("shape", value.shape)
        .setAttribute("device_ordinal", deviceOrdinal)
        .build()
  }

  /** Creates an op which feeds an array of tensor values into the computation.
    *
    * @param  values        Tensors to feed into the computation.
    * @param  deviceOrdinal Specified the TPU device to use. This should be `-1` when the op is running on a TPU device,
    *                       and `>= 0` when the op is running on the CPU device.
    * @param  name          Name for the created op.
    * @return Created op.
    * @throws tf.InvalidDataTypeException If the data type of any of `values` is not a supported TPU data type.
    */
  @throws[tf.InvalidDataTypeException]
  def infeedEnqueueTuple(
      values: Array[Output],
      deviceOrdinal: Int = -1,
      name: String = "InfeedEnqueueTuple"
  ): Op = {
    for (dataType <- values.map(_.dataType)) {
      if (!Ops.SUPPORTED_INFEED_DATA_TYPES.contains(dataType))
        throw tf.InvalidDataTypeException(
          s"$dataType is not a supported TPU infeed data type. " +
              s"The supported data types are: ${Ops.SUPPORTED_INFEED_DATA_TYPES.mkString(", ")}.")
    }
    Op.Builder("InfeedEnqueueTuple", name)
        .addInputList(values)
        .setAttribute("dtypes", values.map(_.dataType))
        .setAttribute("shapes", values.map(_.shape))
        .setAttribute("device_ordinal", deviceOrdinal)
        .build()
  }

  /** Creates a placeholder op for a value that will be fed into the computation.
    *
    * @param  dataType Data type of the tensor that will be fed into the computation.
    * @param  shape    Shape of the tensor that will be fed into the computation.
    * @param  name     Name for the created op.
    * @return Created op output.
    * @throws tf.InvalidDataTypeException If `dataType` is not a supported TPU data type.
    */
  @throws[tf.InvalidDataTypeException]
  def infeedDequeue(
      dataType: DataType,
      shape: Shape,
      name: String = "InfeedDequeue"
  ): Output = {
    if (!Ops.SUPPORTED_INFEED_DATA_TYPES.contains(dataType))
      throw tf.InvalidDataTypeException(
        s"$dataType is not a supported TPU infeed data type. " +
            s"The supported data types are: ${Ops.SUPPORTED_INFEED_DATA_TYPES.mkString(", ")}.")
    Op.Builder("InfeedDequeue", name)
        .setAttribute("dtype", dataType)
        .setAttribute("shape", shape)
        .build().outputs(0)
  }

  /** Creates a placeholder op for an array of values that will be fed into the computation.
    *
    * @param  dataTypes Data types of the tensors that will be fed into the computation.
    * @param  shapes    Shapes of the tensors that will be fed into the computation.
    * @param  name      Name for the created op.
    * @return Created op outputs.
    * @throws tf.InvalidDataTypeException If any of `dataTypes` is not a supported TPU data type.
    */
  @throws[tf.InvalidDataTypeException]
  def infeedDequeueTuple(
      dataTypes: Array[DataType],
      shapes: Array[Shape],
      name: String = "InfeedDequeueTuple"
  ): Array[Output] = {
    for (dataType <- dataTypes) {
      if (!Ops.SUPPORTED_INFEED_DATA_TYPES.contains(dataType))
        throw tf.InvalidDataTypeException(
          s"$dataType is not a supported TPU infeed data type. " +
              s"The supported data types are: ${Ops.SUPPORTED_INFEED_DATA_TYPES.mkString(", ")}.")
    }
    Op.Builder("InfeedDequeueTuple", name)
        .setAttribute("dtypes", dataTypes)
        .setAttribute("shapes", shapes)
        .build().outputs
  }

  protected[tpu] val SUPPORTED_CROSS_REPLICA_SUM_DATA_TYPES: Set[DataType] = {
    Set[DataType](BFLOAT16, FLOAT32)
  }

  protected[tpu] val SUPPORTED_INFEED_DATA_TYPES: Set[DataType] = {
    Set[DataType](BOOLEAN, INT32, INT64, BFLOAT16, FLOAT32, COMPLEX64)
  }

  private[tpu] object Gradients {
    tf.gradientsRegistry.registerNonDifferentiable("InfeedEnqueue")
    tf.gradientsRegistry.registerNonDifferentiable("InfeedEnqueueTuple")
    tf.gradientsRegistry.registerNonDifferentiable("InfeedDequeue")
    tf.gradientsRegistry.registerNonDifferentiable("InfeedDequeueTuple")

    tf.gradientsRegistry.register("CrossReplicaSum", crossReplicaSumGradient)

    private[this] def crossReplicaSumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(crossReplicaSum(outputGradients.head))
    }
  }
}
