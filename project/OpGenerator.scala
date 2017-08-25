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

import com.google.protobuf.TextFormat
import org.tensorflow.framework.{AttrValue, OpDef, OpList}
import org.tensorflow.framework.OpDef.AttrDef

import java.io.StringWriter
import java.nio.file.{Files, Path, Paths}
import java.util.Locale

import scala.collection.mutable
import scala.collection.JavaConverters._

/**
  * @author Emmanouil Antonios Platanios
  */
case class OpGenerator(opDef: OpDef) {
  /** Name of this op used to name the generated functions. */
  val name: String = OpGenerator.processName(opDef.getName)

  /** Returns a boolean value indicating whether this op is supported in the Scala API. */
  def isSupported: Boolean = _isSupported

  def jniNative(className: String): (String, String, String) = {
    if (!_isSupported)
      throw new UnsupportedOperationException(s"The '$name' is not supported in the Scala API.")

    val implementationBuilder = StringBuilder.newBuilder
    val deallocationBuilder = StringBuilder.newBuilder

    val nullReturnValuePlaceholder = "##NULL_RETURN##"

    val scalaArgs = params.map(_._1).map(p => s"${p._1}: ${p._2}").mkString(", ") // TODO: [SCALA] Defaults.
    val headArgs = params.map(_._1).map(p => s"${OpGenerator.scalaTypeToJni(p._2)._2}").mkString(", ")
    val headSignArgs = params.map(_._1).map(p => s"${OpGenerator.scalaTypeToJni(p._2)._1}").mkString("")
    val implArgs = params.map(_._1).map(p => s"${OpGenerator.scalaTypeToJni(p._2)._2} ${p._1}").mkString(", ")

    implementationBuilder.append(
      s"""  REQUIRE_HANDLE(context, TFE_Context, context_handle, $nullReturnValuePlaceholder);
         |  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
         |
         |  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
         |      TFE_NewOp(context, "${opDef.getName}", status.get()), TFE_DeleteOp);
         |  CHECK_STATUS(env, status.get(), $nullReturnValuePlaceholder);""".stripMargin)

    addInputs(implementationBuilder, nullReturnValuePlaceholder)
    addInferredAttributes(implementationBuilder, nullReturnValuePlaceholder)
    addAttributes(implementationBuilder, deallocationBuilder, nullReturnValuePlaceholder)

    val numOutputs = numOutputsExpression()
    val (scalaReturnType, returnType, nullValue) = numOutputs match {
      case "0" => ("Unit", ("V", "void"), "")
      case "1" => ("Long", ("J", "jlong"), "0")
      case _ => ("Array[Long]", ("[J", "jlongArray"), "nullptr")
    }

    addExecute(implementationBuilder, deallocationBuilder, numOutputs, nullReturnValuePlaceholder)

    val scalaFunction =
      s"""  @native def $name(contextHandle: Long, $scalaArgs): $scalaReturnType""".stripMargin

    val header =
      s"""/*
         | * Class:     ${className}__
         | * Method:    $name
         | * Signature: (J$headSignArgs)${returnType._1}
         | */
         |JNIEXPORT ${returnType._2} JNICALL Java_${className}_00024_$name
         |  (JNIEnv *, jobject, jlong, $headArgs);""".stripMargin

    val implementation =
      s"""JNIEXPORT ${returnType._2} JNICALL Java_${className}_00024_$name(
         |    JNIEnv* env, jobject object, jlong context_handle, $implArgs) {
         |${implementationBuilder.mkString}
         |}""".stripMargin.replace(nullReturnValuePlaceholder, nullValue)

    (scalaFunction, header, implementation)
  }

  private[this] val numInputs = opDef.getInputArgCount

  /** All parameters, including inputs and non-inferred attributes, required, and those with defaults, except for the
    * `"name"` attribute, paired with their Scala types and their default values. */
  private[this] val params = mutable.ListBuffer.empty[((String, String), Option[String])]

  /** Map from attribute name to the first input argument it is inferred from. */
  private[this] val inferredAttrs = mutable.HashMap.empty[String, String]

  /** Names and Scala types of the non-inferred attributes, in parameter order. */
  private[this] val attrs = mutable.ListBuffer.empty[(String, String)]

  /** Map from attribute name to a list of the argument indices to which it corresponds. */
  private[this] val attrToArgs = mutable.HashMap.empty[String, mutable.ListBuffer[Int]]

  private[this] val attrExpressions = mutable.HashMap.empty[String, String]

  /** Boolean value indicating whether this op is supported in the Scala API. */
  private[this] var _isSupported = true

  /** Marks this op as being unsupported in the Scala API. */
  private[this] def setUnsupported(): Unit = {
    _isSupported = false
    params.clear()
    inferredAttrs.clear()
    attrs.clear()
    attrToArgs.clear()
    attrExpressions.clear()
  }

  // Check if this op is supported for eager execution.
  if (opDef.getInputArgList.asScala.exists(_.getIsRef))
    setUnsupported()
  if (opDef.getInputArgList.asScala.exists(a => a.getTypeAttr.isEmpty && a.getTypeListAttr.isEmpty))
    setUnsupported()
  if (opDef.getOutputArgList.asScala.exists(_.getIsRef))
    setUnsupported()

  if (_isSupported)
    processArgs()

  private[this] def addAttributeForArgument(attribute: String, argumentIndex: Int): Unit = {
    inferredAttrs.getOrElseUpdate(attribute, opDef.getInputArg(argumentIndex).getName)
    attrToArgs.getOrElseUpdate(attribute, mutable.ListBuffer.empty[Int]).append(argumentIndex)
  }

  private[this] def processArgs(): Unit = {
    // All the input arguments followed by those attributes that don't have defaults.
    val argsWithoutDefaults = mutable.ListBuffer.empty[(String, String, String)]

    // Parameters with default values (these have to be listed after those without). No input arguments are included
    // here, just attributes.
    val argsWithDefaults = mutable.ListBuffer.empty[((String, String, String), String)]

    // Process input arguments.
    opDef.getInputArgList.asScala.zipWithIndex.foreach { case (arg, index) =>
      var scalaType = "Long"
      if (!arg.getTypeAttr.isEmpty)
        addAttributeForArgument(arg.getTypeAttr, index)
      else
        addAttributeForArgument(arg.getTypeListAttr, index)
      if (!arg.getNumberAttr.isEmpty) {
        scalaType = "Array[Long]"
        addAttributeForArgument(arg.getNumberAttr, index)
      }
      argsWithoutDefaults.append((arg.getName, scalaType, "TFE_TensorHandle*"))
    }

    // Process attributes that have not been inferred. We do not want add inferred attributes to the Scala function
    // signatures.
    opDef.getAttrList.asScala.filter(a => !inferredAttrs.contains(a.getName)).foreach { attr =>
      val scalaType = OpGenerator.attrTypeToScala(attr.getType)
      if (scalaType.isEmpty)
        setUnsupported()
      if (_isSupported) {
        if (attr.hasDefaultValue) {
          val scalaValue = OpGenerator.attrValueToScala(attr, attr.getDefaultValue)
          if (scalaValue.isDefined)
            argsWithDefaults.append(((attr.getName, scalaType.get, attr.getType), scalaValue.get))
          else
            setUnsupported()
        } else {
          argsWithoutDefaults.append((attr.getName, scalaType.get, attr.getType))
        }
      }
    }

    if (_isSupported) {
      // Save the list of attribute parameters (i.e., attributes that won't be inferred). Those with defaults go at
      // the end. Get the attributes in the order we want by taking the attributes without defaults from the end of
      // argsWithoutDefaults, and then adding argsWithDefaults.
      attrs.appendAll(argsWithoutDefaults.drop(numInputs).map(a => (a._1, a._3)))
      attrs.appendAll(argsWithDefaults.map(_._1).map(a => (a._1, a._3)))
      argsWithoutDefaults.foreach(
        nameAndType => params.append(((OpGenerator.processName(nameAndType._1), nameAndType._2), None)))
      argsWithDefaults.foreach(
        nameTypeAndDefault => params.append(
          ((OpGenerator.processName(nameTypeAndDefault._1._1), nameTypeAndDefault._1._2),
              Some(nameTypeAndDefault._2))))
    }
  }

  private[this] def numOutputsExpression(): String = {
    // If output i is list output, outputSizes[i] will be set to a string with the Scala expression that will evaluate
    // to its length. outputSizes[i] is empty for non-list outputs.
    var numFixedOutputs = 0
    val numOutputsExpression = StringBuilder.newBuilder
    opDef.getOutputArgList.asScala.foreach(outputArg => {
      if (outputArg.getNumberAttr.nonEmpty) {
        if (numOutputsExpression.nonEmpty) numOutputsExpression.append(" + ")
        numOutputsExpression.append(attrExpressions(outputArg.getNumberAttr))
      } else if (outputArg.getTypeListAttr.nonEmpty) {
        if (numOutputsExpression.nonEmpty) numOutputsExpression.append(" + ")
        // Have to be careful to use an expression that works in both the graph and the eager paths here.
        val inferred = inferredAttrs.get(outputArg.getTypeListAttr)
        if (inferred.isDefined)
          numOutputsExpression.append(inferred.get)
        else
          numOutputsExpression.append(s"env->GetArrayLength(${attrExpressions(outputArg.getTypeListAttr)})")
      } else {
        numFixedOutputs += 1
      }
    })
    if (numFixedOutputs > 0) {
      if (numOutputsExpression.nonEmpty) numOutputsExpression.append(" + ")
      numOutputsExpression.append(numFixedOutputs)
    } else if (numOutputsExpression.isEmpty) {
      numOutputsExpression.append("0")
    }
    numOutputsExpression.mkString
  }

  private[this] def addInputs(implementationBuilder: mutable.StringBuilder, nullReturnValue: String): Unit = {
    // TODO: !!! Can manage tensor handles better.
    params.take(numInputs).foreach(param => {
      val inputName = param._1._1
      val inputScalaType = param._1._2
      inputScalaType match {
        case "Long" =>
          implementationBuilder.append(
            s"""
               |
               |  REQUIRE_HANDLE(${inputName}_tensor_handle, TFE_TensorHandle, $inputName, $nullReturnValue);
               |  TFE_OpAddInput(op.get(), ${inputName}_tensor_handle, status.get());
               |  CHECK_STATUS(env, status.get(), $nullReturnValue);""".stripMargin)
        case "Array[Long]" =>
          val numTensors = s"${inputName}_num_tensors"
          val tensorElems = s"${inputName}_elems"
          implementationBuilder.append(
            s"""
               |
               |  const int $numTensors = env->GetArrayLength($inputName);
               |  jlong *$tensorElems = env->GetLongArrayElements($inputName, nullptr);
               |  for (int i = 0; i < $numTensors; ++i) {
               |    REQUIRE_HANDLE(tensor_handle, TFE_TensorHandle, $tensorElems[i], 0);
               |    TFE_OpAddInput(op.get(), tensor_handle, status.get());
               |    CHECK_STATUS(env, status.get(), $nullReturnValue);
               |  }
               |  env->ReleaseLongArrayElements($inputName, $tensorElems, JNI_ABORT);""".stripMargin)
        case _ => throw new IllegalArgumentException(s"Invalid input argument Scala type '$inputScalaType'.")
      }
    })
  }

  private[this] def addInferredAttributes(
      implementationBuilder: mutable.StringBuilder, nullReturnValue: String): Unit = {
    // Validate list inputs and infer length attributes.
    opDef.getAttrList.asScala.filter(
      a => (a.getType == "int" || a.getType == "type") && attrToArgs.contains(a.getName)).foreach(attr => {
      val attrName = attr.getName
      attr.getType match {
        case "int" =>
          // Inferred int attributes are the lengths of input lists.
          // Check that those inputs are lists of the same length.
          attrToArgs(attrName).foreach(arg => {
            val inputName = params(arg)._1._1
            if (!attrExpressions.contains(attrName)) {
              val attrValueName = attrExpressions.getOrElseUpdate(attrName, s"${inputName}_attr_$attrName")
              implementationBuilder.append(
                s"""
                   |
                   |  const int $attrValueName = env->GetArrayLength($inputName);
                   |  TFE_OpSetAttrInt(op.get(), "$attrName", static_cast<int64_t>($attrValueName));""".stripMargin)
            } else {
              val attrValueName = attrExpressions(attrName)
              implementationBuilder.append(
                s"""
                   |
                   |  const int ${inputName}_attr_$attrName = env->GetArrayLength($inputName);
                   |  if ($attrValueName != ${inputName}_attr_$attrName) {
                   |      std::stringstream error_msg;
                   |      error_msg
                   |          << "List argument '$inputName' of '$name' op with length '"
                   |          << ${inputName}_attr_$attrName
                   |          << "' must match length '"
                   |          << $attrValueName
                   |          << "' of argument '${inferredAttrs(attrName)}'";
                   |      throw_exception(env, jvm_illegal_argument_exception, error_msg.str().c_str());
                   |  }""".stripMargin)
            }
          })
        case "type" =>
          attrToArgs(attrName).foreach(arg => {
            val inputName = params(arg)._1._1
            val inputScalaType = params(arg)._1._2
            inputScalaType match {
              case "Long" =>
                if (!attrExpressions.contains(attrName)) {
                  val attrValueName = attrExpressions.getOrElseUpdate(attrName, s"${inputName}_attr_$attrName")
                  val tensorHandle = s"${attrValueName}_${inputName}_tensor_h"
                  implementationBuilder.append(
                    s"""
                       |
                       |  REQUIRE_HANDLE($tensorHandle, TFE_TensorHandle, $inputName, $nullReturnValue);
                       |  const TF_DataType $attrValueName = TFE_TensorHandleDataType($tensorHandle);
                       |  TFE_OpSetAttrType(op.get(), "$attrName", $attrValueName);""".stripMargin)
                } else {
                  val attrValueName = attrExpressions(attrName)
                  val tensorHandle = s"${attrValueName}_${inputName}_tensor_h"
                  implementationBuilder.append(
                    s"""
                       |
                       |  REQUIRE_HANDLE($tensorHandle, TFE_TensorHandle, $inputName, $nullReturnValue);
                       |  const TF_DataType ${inputName}_attr_$attrName = TFE_TensorHandleDataType($tensorHandle);
                       |  if ($attrValueName != ${inputName}_attr_$attrName) {
                       |      std::stringstream error_msg;
                       |      error_msg
                       |          << "Argument '$inputName' of '$name' op with data type '"
                       |          << ${inputName}_attr_$attrName
                       |          << "' must match data type '"
                       |          << $attrValueName
                       |          << "' of argument '${inferredAttrs(attrName)}'";
                       |      throw_exception(env, jvm_illegal_argument_exception, error_msg.str().c_str());
                       |  }""".stripMargin)
                }
              case "Array[Long]" =>
                val numTensors = s"${inputName}_attr_${attrName}_num_tensors"
                val tensorElems = s"${inputName}_attr_${attrName}_elems"
                implementationBuilder.append(
                  s"""
                     |
                     |  const int $numTensors = env->GetArrayLength($inputName);
                     |  jlong *$tensorElems = env->GetLongArrayElements($inputName, nullptr);""".stripMargin)
                if (!attrExpressions.contains(attrName)) {
                  val attrValueName = attrExpressions.getOrElseUpdate(attrName, s"${inputName}_attr_$attrName")
                  implementationBuilder.append(
                    s"""
                       |
                       |  REQUIRE_HANDLE(${tensorElems}_head, TFE_TensorHandle, $tensorElems[0], $nullReturnValue);
                       |  const TF_DataType $attrValueName = TFE_TensorHandleDataType(${tensorElems}_head);
                       |  TFE_OpSetAttrType(op.get(), "$attrName", $attrValueName);""".stripMargin)
                }
                val attrValueName = attrExpressions(attrName)
                implementationBuilder.append(
                  s"""
                     |
                     |  for (int i = 0; i < $numTensors; ++i) {
                     |    REQUIRE_HANDLE(tensor, TFE_TensorHandle, $tensorElems[i], 0);
                     |    const TF_DataType data_type = TFE_TensorHandleDataType(tensor);
                     |    if ($attrValueName != data_type) {
                     |      std::stringstream error_msg;
                     |      error_msg
                     |          << "Argument '$inputName' of '$name' op with data type '"
                     |          << data_type
                     |          << "' must match data type '"
                     |          << $attrValueName
                     |          << "' of argument '${inferredAttrs(attrName)}'";
                     |      throw_exception(env, jvm_illegal_argument_exception, error_msg.str().c_str());
                     |    }
                     |  }
                     |  env->ReleaseLongArrayElements($inputName, $tensorElems, JNI_ABORT);""".stripMargin)
              case _ => throw new IllegalArgumentException(s"Invalid input argument Scala type '$inputScalaType'.")
            }
          })
      }
    })
  }

  @throws[IllegalArgumentException]
  private[this] def addAttributes(
      implementationBuilder: mutable.StringBuilder, deallocationBuilder: mutable.StringBuilder,
      nullReturnValue: String): Unit = {
    attrs.zipWithIndex.foreach({ case (attr, index) =>
      val attrName = attr._1
      val attrType = attr._2
      val value = attrExpressions.getOrElseUpdate(attrName, params(index + numInputs)._1._1)
      attrType match {
        case "string" =>
          implementationBuilder.append(
            s"""
               |
               |  jbyte *${attrName}_c_value = env->GetByteArrayElements($value, nullptr);
               |  TFE_OpSetAttrString(op.get(), "$attrName", reinterpret_cast<const char *>(${attrName}_c_value));
               |  env->ReleaseByteArrayElements($value, ${attrName}_c_value, JNI_ABORT);""".stripMargin)
        case "int" =>
          implementationBuilder.append(
            s"""
               |
               |  TFE_OpSetAttrInt(op.get(), "$attrName", static_cast<int64_t>($value));""".stripMargin)
        case "float" =>
          implementationBuilder.append(
            s"""
               |
               |  TFE_OpSetAttrFloat(op.get(), "$attrName", static_cast<float>($value));""".stripMargin)
        case "bool" =>
          implementationBuilder.append(
            s"""
               |
               |  TFE_OpSetAttrBool(op.get(), "$attrName", static_cast<unsigned char>($value));""".stripMargin)
        case "type" =>
          implementationBuilder.append(
            s"""
               |
               |  TFE_OpSetAttrType(op.get(), "$attrName", static_cast<TF_DataType>($value));""".stripMargin)
        case "shape" =>
          implementationBuilder.append(
            s"""
               |
               |  std::unique_ptr<int64_t[]> ${attrName}_c_value;
               |  int ${attrName}_num_dims = -1;
               |  if ($value != nullptr) {
               |    ${attrName}_num_dims = env->GetArrayLength($value);
               |    ${attrName}_c_value.reset(new int64_t[${attrName}_num_dims]);
               |    jlong *${attrName}_elems = env->GetLongArrayElements($value, nullptr);
               |    for (int i = 0; i < ${attrName}_num_dims; ++i) {
               |      ${attrName}_c_value[i] = static_cast<int64_t>(${attrName}_elems[i]);
               |    }
               |    env->ReleaseLongArrayElements($value, ${attrName}_elems, JNI_ABORT);
               |  }
               |  TFE_OpSetAttrShape(
               |      op.get(), "$attrName", ${attrName}_c_value.get(), static_cast<int>(${attrName}_num_dims),
               |      status.get());
               |  CHECK_STATUS(env, status.get(), $nullReturnValue);""".stripMargin)
        case "tensor" => throw new UnsupportedOperationException(s"Unsupported attribute type '$attrType'.")
        case "func" => throw new UnsupportedOperationException(s"Unsupported attribute type '$attrType'.")
        case "list(string)" =>
          implementationBuilder.append(
            s"""
               |
               |  int ${attrName}_num_strings = env->GetArrayLength($value);
               |  jbyteArray **${attrName}_arrays = new jbyteArray *[${attrName}_num_strings];
               |  jbyte **${attrName}_strings = new jbyte *[${attrName}_num_strings];
               |  for (int i = 0; i < ${attrName}_num_strings; i++) {
               |    ${attrName}_arrays[i] = (jbyteArray) env->GetObjectArrayElement($value, i);
               |    ${attrName}_strings[i] = env->GetByteArrayElements(${attrName}_arrays[i], nullptr);
               |  }
               |  TFE_OpSetAttrStringList(
               |    op.get(), "$attrName", const_cast<const char **>(reinterpret_cast<char **>(${attrName}_strings)),
               |    ${attrName}_num_strings);""".stripMargin)
          deallocationBuilder.append(
            s"""
               |
               |  for (int i = 0; i < ${attrName}_num_strings; i++) {
               |    env->ReleaseByteArrayElements(${attrName}_arrays[i], ${attrName}_strings[i], JNI_ABORT);
               |  }""".stripMargin)
        case "list(int)" =>
          // Make a copy of the array to paper over any differences in byte representations of the JVM type and the
          // corresponding C type. For example, jint vs TF_DataType. If this copy turns out to be a problem in
          // practice, we can avoid it for many types.
          implementationBuilder.append(
            s"""
               |
               |  const int ${attrName}_n = env->GetArrayLength($value);
               |  std::unique_ptr<int64_t[]> ${attrName}_c_value(new int64_t[${attrName}_n]);
               |  jlong* ${attrName}_elems = env->GetLongArrayElements($value, nullptr);
               |  for (int i = 0; i < ${attrName}_n; ++i) {
               |    ${attrName}_c_value[i] = static_cast<int64_t>(${attrName}_elems[i]);
               |  }
               |  TFE_OpSetAttrIntList(op.get(), "$attrName", ${attrName}_c_value.get(), ${attrName}_n);
               |  env->ReleaseLongArrayElements($value, ${attrName}_elems, JNI_ABORT);""".stripMargin)
        case "list(float)" =>
          // Make a copy of the array to paper over any differences in byte representations of the JVM type and the
          // corresponding C type. For example, jint vs TF_DataType. If this copy turns out to be a problem in
          // practice, we can avoid it for many types.
          implementationBuilder.append(
            s"""
               |
               |  const int ${attrName}_n = env->GetArrayLength($value);
               |  std::unique_ptr<float[]> ${attrName}_c_value(new float[${attrName}_n]);
               |  jfloat* ${attrName}_elems = env->GetFloatArrayElements($value, nullptr);
               |  for (int i = 0; i < ${attrName}_n; ++i) {
               |    ${attrName}_c_value[i] = static_cast<float>(${attrName}_elems[i]);
               |  }
               |  TFE_OpSetAttrFloatList(op.get(), "$attrName", ${attrName}_c_value.get(), ${attrName}_n);
               |  env->ReleaseFloatArrayElements($value, ${attrName}_elems, JNI_ABORT);""".stripMargin)
        case "list(bool)" =>
          // Make a copy of the array to paper over any differences in byte representations of the JVM type and the
          // corresponding C type. For example, jint vs TF_DataType. If this copy turns out to be a problem in
          // practice, we can avoid it for many types.
          implementationBuilder.append(
            s"""
               |
               |  const int ${attrName}_n = env->GetArrayLength($value);
               |  std::unique_ptr<unsigned char[]> ${attrName}_c_value(new unsigned char[${attrName}_n]);
               |  jboolean* ${attrName}_elems = env->GetBooleanArrayElements($value, nullptr);
               |  for (int i = 0; i < ${attrName}_n; ++i) {
               |    ${attrName}_c_value[i] = static_cast<unsigned char>(${attrName}_elems[i]);
               |  }
               |  TFE_OpSetAttrBoolList(op.get(), "$attrName", ${attrName}_c_value.get(), ${attrName}_n);
               |  env->ReleaseBooleanArrayElements($value, ${attrName}_elems, JNI_ABORT);""".stripMargin)
        case "list(type)" =>
          // Make a copy of the array to paper over any differences in byte representations of the JVM type and the
          // corresponding C type. For example, jint vs TF_DataType. If this copy turns out to be a problem in
          // practice, we can avoid it for many types.
          implementationBuilder.append(
            s"""
               |
               |  const int ${attrName}_n = env->GetArrayLength($value);
               |  std::unique_ptr<TF_DataType[]> ${attrName}_c_value(new TF_DataType[${attrName}_n]);
               |  jint* ${attrName}_elems = env->GetIntArrayElements($value, nullptr);
               |  for (int i = 0; i < ${attrName}_n; ++i) {
               |    ${attrName}_c_value[i] = static_cast<TF_DataType>(${attrName}_elems[i]);
               |  }
               |  TFE_OpSetAttrTypeList(op.get(), "$attrName", ${attrName}_c_value.get(), ${attrName}_n);
               |  env->ReleaseIntArrayElements($value, ${attrName}_elems, JNI_ABORT);""".stripMargin)
        case "list(shape)" =>
          implementationBuilder.append(
            s"""
               |
               |  std::unique_ptr<int[]> ${attrName}_c_num_dims;
               |  std::unique_ptr<int64_t*[]> ${attrName}_c_shapes;
               |  const int ${attrName}_c_num_shapes = env->GetArrayLength($value);
               |  if (${attrName}_c_num_shapes > 0) {
               |    ${attrName}_c_num_dims.reset(new int[${attrName}_c_num_shapes]);
               |    ${attrName}_c_shapes.reset(new int64_t*[${attrName}_c_num_shapes]);
               |    for (int j = 0; j < ${attrName}_c_num_shapes; ++j) {
               |      jlongArray shape = (jlongArray) env->GetObjectArrayElement($value, j);
               |      ${attrName}_c_num_dims[j] = -1;
               |      if (shape != nullptr) {
               |        ${attrName}_c_num_dims[j] = env->GetArrayLength(shape);
               |        ${attrName}_c_shapes[j] = new int64_t[${attrName}_c_num_dims[j]];
               |        jlong *shape_elems = env->GetLongArrayElements(shape, nullptr);
               |        for (int i = 0; i < ${attrName}_c_num_dims[j]; ++i) {
               |          ${attrName}_c_shapes[j][i] = static_cast<int64_t>(shape_elems[i]);
               |        }
               |        env->ReleaseLongArrayElements(shape, shape_elems, JNI_ABORT);
               |      } else {
               |        ${attrName}_c_shapes[j] = new int64_t[0];
               |      }
               |    }
               |  }
               |  TFE_OpSetAttrShapeList(
               |      op.get(), "$attrName", const_cast<const int64_t **>(${attrName}_c_shapes.get()),
               |      ${attrName}_c_num_dims.get(), ${attrName}_c_num_shapes, status.get());
               |  CHECK_STATUS(env, status.get(), $nullReturnValue);""".stripMargin)
        case "list(tensor)" => throw new UnsupportedOperationException(s"Unsupported attribute type '$attrType'.")
        case "list(func)" => throw new UnsupportedOperationException(s"Unsupported attribute type '$attrType'.")
        case _ => throw new IllegalArgumentException(s"Invalid attribute type '$attrType'.")
      }
    })
  }

  private[this] def addExecute(
      implementationBuilder: mutable.StringBuilder, deallocationBuilder: mutable.StringBuilder, numOutputs: String,
      nullReturnValue: String): Unit = {
    implementationBuilder.append(
      s"""
         |
         |  const int num_outputs = $numOutputs;
         |  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [num_outputs]);
         |  std::unique_ptr<int[]> actual_num_outputs(new int[1] {1});
         |  TFE_Execute(op.get(), outputs.get(), actual_num_outputs.get(), status.get());
         |  CHECK_STATUS(env, status.get(), $nullReturnValue);
         |${if (deallocationBuilder.nonEmpty) "\n" + deallocationBuilder.mkString + "\n" else ""}""".stripMargin)
    numOutputs match {
      case "0" => implementationBuilder.append(
        s"""
           |  return;""".stripMargin)
      case "1" =>
        implementationBuilder.append(
          s"""
             |  return reinterpret_cast<jlong>(outputs[0]);""".stripMargin)
      case _ =>
        implementationBuilder.append(
          s"""
             |  jlongArray outputs_array = env->NewLongArray(static_cast<jsize>(num_outputs));
             |  jlong* output_elems = env->GetLongArrayElements(outputs_array, nullptr);
             |  for (int i = 0; i < num_outputs; ++i) {
             |    output_elems[i] = reinterpret_cast<jlong>(outputs[i]);
             |  }
             |  env->ReleaseLongArrayElements(outputs_array, output_elems, JNI_COMMIT);
             |  return outputs_array;""".stripMargin)
    }
  }
}

object OpGenerator {
  def generateFiles(path: Path, ops: Map[String, Seq[String]], scalaPackage: String): Unit = {
    val opList = OpList.newBuilder()
    TextFormat.merge(
      Files.readAllLines(path.resolve(Paths.get("resources", "ops.pbtxt"))).toArray.mkString("\n"), opList)
    val opDefsMap = opList.getOpList.asScala.map(o => o.getName -> o).toMap
    ops.foreach(o => generateFiles(path, o._1, o._2.map(opDefsMap), scalaPackage))
  }

  private[this] def generateFiles(path: Path, group: String, opDefs: Seq[OpDef], scalaPackage: String): Unit = {
    val headerName = s"tensor_${group.toLowerCase}_ops.h"
    val (scalaObject, nativeHeader, nativeImplementation) = code(scalaPackage, opDefs, headerName, group)
    val scalaPath = path.resolve(Paths.get("scala", scalaPackage.split('.') :+ s"$group.scala": _*))
    val nativePath = path.resolve(Paths.get("native", "generated"))
    val headerPath = nativePath.resolve(headerName)
    val implementationPath = nativePath.resolve(s"tensor_${group.toLowerCase}_ops.cc")
    Files.createDirectories(scalaPath.getParent)
    Files.createDirectories(nativePath)
    Files.write(scalaPath, scalaObject.getBytes())
    Files.write(headerPath, nativeHeader.getBytes())
    Files.write(implementationPath, nativeImplementation.getBytes())
  }

  private[this] def code(
      scalaPackage: String, opDefs: Seq[OpDef], headerName: String, objectName: String): (String, String, String) = {
    val jniObjectName = s"$scalaPackage.$objectName".replace(".", "_")
    val opCode = opDefs.map(OpGenerator(_).jniNative(jniObjectName))
    val scalaObject =
      s"""/* DO NOT EDIT THIS FILE - it is machine generated */
         |
         |/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
         | *
         | * Licensed under the Apache License, Version 2.0 (the "License"); you may not
         | * use this file except in compliance with the License. You may obtain a copy of
         | * the License at
         | *
         | *     http://www.apache.org/licenses/LICENSE-2.0
         | *
         | * Unless required by applicable law or agreed to in writing, software
         | * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
         | * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
         | * License for the specific language governing permissions and limitations under
         | * the License.
         | */
         |
         |package $scalaPackage
         |
         |import org.platanios.tensorflow.jni.TensorFlow
         |
         |object $objectName {
         |  TensorFlow.load()
         |
         |${opCode.map(_._1).mkString("\n")}
         |}
         |""".stripMargin
    val nativeHeader =
      s"""/* DO NOT EDIT THIS FILE - it is machine generated */
         |#include <jni.h>
         |/* Header for class ${jniObjectName}__ */
         |
         |#ifndef _Included_${jniObjectName}__
         |#define _Included_${jniObjectName}__
         |#ifdef __cplusplus
         |extern "C" {
         |#endif
         |${opCode.map(_._2).mkString("\n\n")}
         |
         |#ifdef __cplusplus
         |}
         |#endif
         |#endif
         |""".stripMargin
    val nativeImplementation =
      s"""/* DO NOT EDIT THIS FILE - it is machine generated */
         |
         |/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
         | *
         | * Licensed under the Apache License, Version 2.0 (the "License"); you may not
         | * use this file except in compliance with the License. You may obtain a copy of
         | * the License at
         | *
         | *     http://www.apache.org/licenses/LICENSE-2.0
         | *
         | * Unless required by applicable law or agreed to in writing, software
         | * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
         | * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
         | * License for the specific language governing permissions and limitations under
         | * the License.
         | */
         |
         |#include "$headerName"
         |
         |#include <algorithm>
         |#include <cstring>
         |#include <memory>
         |#include <sstream>
         |
         |#include "include/c_api.h"
         |#include "include/c_eager_api.h"
         |#include "include/exception_jni.h"
         |#include "include/utilities.h"
         |
         |${opCode.map(_._3).mkString("\n\n")}
         |""".stripMargin
    (scalaObject, nativeHeader, nativeImplementation)
  }

  def processName(name: String): String = {
    // TODO: [PYTHON -> SCALA] Avoid reserved keywords.
    val c = name.toCharArray.toBuffer[Char]
    c(0) = Character.toLowerCase(c(0))
    new String(c.toArray)
  }

  @throws[IllegalArgumentException]
  def scalaTypeToJni(scalaType: String): (String, String) = scalaType match {
    case "Array[Byte]" => ("[B", "jbyteArray")
    case "Int" => ("I", "jint")
    case "Long" => ("J", "jlong")
    case "Float" => ("F", "jfloat")
    case "Boolean" => ("Z", "jboolean")
    case "Array[Array[Byte]]" => ("[L", "jobjectArray")
    case "Array[Int]" => ("[I", "jintArray")
    case "Array[Long]" => ("[J", "jlongArray")
    case "Array[Float]" => ("[F", "jfloatArray")
    case "Array[Boolean]" => ("[Z", "jbooleanArray")
    case _ => throw new IllegalArgumentException(s"Unsupported Scala type '$scalaType'.")
  }

  @throws[IllegalArgumentException]
  def jniNullReturnValue(jniType: String): String = jniType match {
    case "jlong" => "0"
    case "jlongArray" => "nullptr"
    case _ => throw new IllegalArgumentException(s"Invalid JNI return type '$jniType'.")
  }

  def attrTypeToScala(attrType: String): Option[String] = attrType match {
    case "string" => Some("Array[Byte]")
    case "int" => Some("Long")
    case "float" => Some("Float")
    case "bool" => Some("Boolean")
    case "type" => Some("Int")
    case "shape" => Some("Array[Long]")
    case "tensor" => None // Some("String")
    case "func" => None
    case "list(string)" => Some("Array[Array[Byte]]")
    case "list(int)" => Some("Array[Long]")
    case "list(float)" => Some("Array[Float]")
    case "list(bool)" => Some("Array[Boolean]")
    case "list(type)" => Some("Array[Int]")
    case "list(shape)" => Some("Array[Array[Long]]")
    case "list(tensor)" => None // Some("Array[String]")
    case "list(func)" => None
    case _ => None
  }

  def attrValueToScala(attr: AttrDef, value: AttrValue): Option[String] = {
    // Note that the return value of this function may contain spaces (for example, it could be a string "foo bar"
    // with an embedded space), and thus is not safe to pass it to wordWrap().
    val scalaValue = attr.getType match {
      case "string" => s"${'"'}${OpGenerator.escapeString(value.getS.toStringUtf8)}${'"'}"
      case "int" => s"${value.getI}L"
      case "float" =>
        val f = value.getF
        if (f == Float.NaN) "Float.NaN"
        else if (f == Float.NegativeInfinity) "Float.NegativeInfinity"
        else if (f == Float.PositiveInfinity) "Float.PositiveInfinity"
        else s"${f}f"
      case "bool" => value.getB.toString
      case "type" => s"${value.getType.getNumber.toString}"
      case "shape" =>
        val s = value.getShape
        if (s.getUnknownRank) null else s"Array[Long](${s.getDimList.asScala.mkString(", ")})"
      case "tensor" =>
        // Note that this gets used in the argument list and so it must survive naive word wrapping.
        s"${"\"\"\""}${TextFormat.shortDebugString(value.getTensor)}${"\"\"\""}"
      case "func" => s"${'"'}${OpGenerator.escapeString(value.getFunc.getName)}${'"'}"
      case t if t.startsWith("list(") =>
        val content = {
          if (value.getList.getSCount > 0) {
            value.getList.getSList.asScala.map(
              v => s"${'"'}${OpGenerator.escapeString(v.toStringUtf8)}${'"'}").mkString(", ")
          } else if (value.getList.getICount > 0) {
            value.getList.getIList.asScala.map(v => s"${v}L").mkString(", ")
          } else if (value.getList.getFCount > 0) {
            value.getList.getFList.asScala.map(v => {
              if (v == Float.NaN) "Float.NaN"
              else if (v == Float.NegativeInfinity) "Float.NegativeInfinity"
              else if (v == Float.PositiveInfinity) "Float.PositiveInfinity"
              else s"${v}f"
            }).mkString(", ")
          } else if (value.getList.getBCount > 0) {
            value.getList.getBList.asScala.map(_.toString).mkString(", ")
          } else if (value.getList.getTypeCount > 0) {
            value.getList.getTypeList.asScala.map(v => s"${v.getNumber.toString}").mkString(", ")
          } else if (value.getList.getShapeCount > 0) {
            value.getList.getShapeList.asScala.map(v => {
              if (v.getUnknownRank)
                null
              else
                s"Array[Long](${v.getDimList.asScala.mkString(", ")})"
            }).mkString(", ")
          } else if (value.getList.getTensorCount > 0) {
            val values = value.getList.getTensorList.asScala.map(v => {
              // Note that this gets used in the argument list and so it must survive naive word wrapping.
              s"${"\"\"\""}${TextFormat.shortDebugString(v)}${"\"\"\""}"
            })
            s"Array[String](${values.mkString(", ")})"
          } else if (value.getList.getFuncCount > 0) {
            value.getList.getFuncList.asScala.map(
              v => s"${'"'}${OpGenerator.escapeString(v.getName)}${'"'}").mkString(", ")
          }
        }
        s"Array($content)"
      case _ => null
    }
    Option(scalaValue)
  }

  def escapeString(string: String): String = {
    def hex(char: Char): String = Integer.toHexString(char).toUpperCase(Locale.ENGLISH)

    if (string == null) {
      null
    } else {
      val writer: StringWriter = new StringWriter(string.length() * 2)
      val size: Int = string.length
      var i = 0
      while (i < size) {
        string.charAt(i) match {
          case '\b' => writer.write('\\'); writer.write('b')
          case '\n' => writer.write('\\'); writer.write('n')
          case '\t' => writer.write('\\'); writer.write('t')
          case '\f' => writer.write('\\'); writer.write('f')
          case '\r' => writer.write('\\'); writer.write('r')
          case '\'' => writer.write('\\'); writer.write('\'')
          case '\"' => writer.write('\\'); writer.write('\"')
          case '\\' => writer.write('\\'); writer.write('\\')
          case c if c > 0xfff => writer.write("\\u" + hex(c))
          case c if c > 0xff => writer.write("\\u0" + hex(c))
          case c if c > 0x7f => writer.write("\\u00" + hex(c))
          case c if c < 32 && c > 0xf => writer.write("\\u00" + hex(c))
          case c if c < 32 => writer.write("\\u000" + hex(c))
          case c => writer.write(c)
        }
        i += 1
      }
      writer.toString
    }
  }
}
