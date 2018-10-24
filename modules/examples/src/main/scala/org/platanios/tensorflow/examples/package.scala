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

package org.platanios.tensorflow

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.DataType
import org.platanios.tensorflow.api.implicits.helpers.{DataTypeToShape, OutputStructure, OutputToDataType, OutputToShape}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.rnn.cell.LSTMState

/**
  * @author Emmanouil Antonios Platanios
  */
package object examples {
  // Implicit helpers for Scala 2.11.
  val evOutputStructureFloatLong     : OutputStructure[(Output[Float], Output[Long])] = OutputStructure[(Output[Float], Output[Long])]
  val evOutputStructureIntInt        : OutputStructure[(Output[Int], Output[Int])]    = OutputStructure[(Output[Int], Output[Int])]
  val evOutputStructureLSTMStateFloat: OutputStructure[LSTMState[Float]]              = OutputStructure[LSTMState[Float]]

  val evOutputToDataTypeFloatLong     : OutputToDataType[(Output[Float], Output[Long])] = OutputToDataType[(Output[Float], Output[Long])]
  val evOutputToDataTypeIntInt        : OutputToDataType[(Output[Int], Output[Int])]    = OutputToDataType[(Output[Int], Output[Int])]
  val evOutputToDataTypeLSTMStateFloat: OutputToDataType[LSTMState[Float]]              = OutputToDataType[LSTMState[Float]]

  val evOutputToShapeFloatLong     : OutputToShape[(Output[Float], Output[Long])] = OutputToShape[(Output[Float], Output[Long])]
  val evOutputToShapeIntInt        : OutputToShape[(Output[Int], Output[Int])]    = OutputToShape[(Output[Int], Output[Int])]
  val evOutputToShapeLSTMStateFloat: OutputToShape[LSTMState[Float]]              = OutputToShape[LSTMState[Float]]

  val evDataTypeToShapeFloatLong     : DataTypeToShape.Aux[(DataType[Float], DataType[Long]), (Shape, Shape)]  = DataTypeToShape[(DataType[Float], DataType[Long])]
  val evDataTypeToShapeIntInt        : DataTypeToShape.Aux[(DataType[Int], DataType[Int]), (Shape, Shape)]     = DataTypeToShape[(DataType[Int], DataType[Int])]
  val evDataTypeToShapeLSTMStateFloat: DataTypeToShape.Aux[(DataType[Float], DataType[Float]), (Shape, Shape)] = DataTypeToShape[(DataType[Float], DataType[Float])]
}
