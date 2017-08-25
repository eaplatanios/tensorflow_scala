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

// TODO: [OP_GEN] Handle default values.

/**
  * @author Emmanouil Antonios Platanios
  */
case class OpGenerator(opDef: OpDef) {
  import OpGenerator._

  /** Name of this op used to name the generated functions. */
  val name: String = OpGenerator.processName(opDef.getName)

  private[this] val cNullValuePlaceholder = "##NULL_RETURN##"

  def generateCode(className: String): (String, String, String) = {
    val codeBuilder = StringBuilder.newBuilder
    val deallocationBuilder = StringBuilder.newBuilder

    val arguments = inputs ++ parameters
    val scalaArgs = arguments.map(p => s"${p._2}: ${typeToScalaType(argumentTypes(p._1))}").mkString(", ")
    val headArgs = arguments.map(_._1).map(p => s"${typeToJni(argumentTypes(p))}").mkString(", ")
    val headSignArgs = arguments.map(_._1).map(p => s"${typeToShortJni(argumentTypes(p))}").mkString("")
    val implArgs = arguments.map(p => s"${typeToJni(argumentTypes(p._1))} ${p._2}").mkString(", ")

    codeBuilder.append(
      s"""  REQUIRE_HANDLE(context, TFE_Context, context_handle, $cNullValuePlaceholder);
         |  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
         |
         |  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
         |      TFE_NewOp(context, "${opDef.getName}", status.get()), TFE_DeleteOp);
         |  CHECK_STATUS(env, status.get(), $cNullValuePlaceholder);""".stripMargin)

    addInputs(codeBuilder)
    addInferredAttributes(codeBuilder)
    addAttributes(codeBuilder, deallocationBuilder)

    val numOutputs = numOutputsExpression()
    val (scalaReturnType, returnType, nullValue) = numOutputs match {
      case "0" => ("Unit", ("V", "void"), "")
      case "1" => ("Long", ("J", "jlong"), "0")
      case _ => ("Array[Long]", ("[J", "jlongArray"), "nullptr")
    }

    addExecute(codeBuilder, deallocationBuilder, numOutputs)

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
         |${codeBuilder.mkString}
         |}""".stripMargin.replace(cNullValuePlaceholder, nullValue)

    (scalaFunction, header, implementation)
  }

  // Check if this op is supported for eager execution.
  if (opDef.getInputArgList.asScala.exists(_.getIsRef))
    throwUnsupportedException(name, "Op inputs cannot be reference types.")
  if (opDef.getOutputArgList.asScala.exists(_.getIsRef))
    throwUnsupportedException(name, "Op outputs cannot be reference types.")

  private[this] val initializationOutputs = initialize()

  /** Map from argument (i.e., input or attribute) names to corresponding TensorFlow type. */
  val argumentTypes: Map[String, String] = initializationOutputs._1

  /** Names of the inputs, paired with their processed (i.e., C-safe and Scala-safe) version. */
  val inputs: Seq[(String, String)] = initializationOutputs._2

  /** All non-inferrable attributes, including those with defaults, except for the `"name"` attribute, paired with their
    * processed (i.e., C-safe and Scala-safe) names. */
  val parameters: Seq[(String, String)] = initializationOutputs._3

  /** Map from parameter name (original -- not processed) to their default values. */
  val parameterDefaults: Map[String, AttrValue] = initializationOutputs._4

  /** Map from inferrable attribute name to the first input argument it is inferred from. */
  val inferrableAttributes: Map[String, String] = initializationOutputs._5

  /** Map from inferrable attribute name to a list of the parameter indices to which it corresponds. */
  val inferrableAttributeToInputs: Map[String, Seq[Int]] = initializationOutputs._6

  /** C code expressions that create and initialize all variables related to inferring attribute values. */
  val inferredAttributeExpressions: Seq[String] = initializationOutputs._7

  /** Map from attribute name to the C expression that computes its value (often simply set to the parameter name). */
  val attributeExpressions: Map[String, String] = initializationOutputs._8

  private[this] def initialize(): (
      Map[String, String],        // argumentTypes
          Seq[(String, String)],  // inputs
          Seq[(String, String)],  // parameters
          Map[String, AttrValue], // parameterDefaults
          Map[String, String],    // inferrableAttributes
          Map[String, Seq[Int]],  // attributeToParameters
          Seq[String],            // inferredAttributeExpressions
          Map[String, String]     // attributeExpressions
      ) = {
    val argumentTypes = mutable.HashMap.empty[String, String]
    val inputs = mutable.ListBuffer.empty[(String, String)]
    val parameters = mutable.ListBuffer.empty[(String, String)]
    val parameterDefaults = mutable.HashMap.empty[String, AttrValue]
    val inferrableAttributes = mutable.HashMap.empty[String, String]
    val inferrableAttributeToInputs = mutable.HashMap.empty[String, mutable.ListBuffer[Int]]
    val inferredAttributeExpressions = mutable.ListBuffer.empty[String]
    val attributeExpressions = mutable.HashMap.empty[String, String]

    // Process input arguments.
    opDef.getInputArgList.asScala.zipWithIndex.foreach { case (arg, index) =>
      argumentTypes.update(arg.getName, if (!arg.getNumberAttr.isEmpty) "list(tensor)" else "tensor")
      val inferrableAttrs = mutable.ListBuffer.empty[(String, String)]
      if (!arg.getTypeAttr.isEmpty)
        inferrableAttrs.append((arg.getTypeAttr, "type"))
      else
        inferrableAttrs.append((arg.getTypeListAttr, "list(type)"))
      if (!arg.getNumberAttr.isEmpty)
        inferrableAttrs.append((arg.getNumberAttr, "int"))
      inferrableAttrs.foreach(a => {
        argumentTypes.update(a._1, a._2)
        inferrableAttributes.getOrElseUpdate(a._1, opDef.getInputArg(index).getName)
        inferrableAttributeToInputs.getOrElseUpdate(a._1, mutable.ListBuffer.empty[Int]).append(index)
      })
      inputs.append((arg.getName, OpGenerator.processName(arg.getName)))
    }

    // Process attributes that have not been inferred. We do not want add inferred attributes to the Scala function
    // signatures.
    val attrsWithoutDefaults = mutable.ListBuffer.empty[String]
    val attrsWithDefaults = mutable.ListBuffer.empty[String]
    opDef.getAttrList.asScala.filter(a => !inferrableAttributes.contains(a.getName)).foreach { attr =>
      argumentTypes.update(attr.getName, attr.getType)
      if (attr.hasDefaultValue) {
        parameterDefaults.update(attr.getName, attr.getDefaultValue)
        attrsWithDefaults.append(attr.getName)
      } else {
        attrsWithoutDefaults.append(attr.getName)
      }
    }
    // Save the list of attribute parameters (i.e., attributes that won't be inferred). Those with defaults go at the
    // end. Get the attributes in the order we want by taking the attributes without defaults from the end of
    // argsWithoutDefaults, and then adding argsWithDefaults.
    attrsWithoutDefaults.foreach(a => parameters.append((a, OpGenerator.processName(a))))
    attrsWithDefaults.foreach(a => parameters.append((a, OpGenerator.processName(a))))

    // Infer attribute values and create the appropriate attribute expressions.
    inferrableAttributeToInputs.foreach({ case (attrName, inputIndices) =>
      argumentTypes(attrName) match {
        case "int" =>
          // Inferred int attributes are the lengths of input lists.
          if (!attributeExpressions.contains(attrName)) {
            val inputName = inputs(inputIndices.head)._2
            val attrValueName = s"${inputName}_attr_$attrName"
            val attrValueExpression =
              s"""
                 |
                 |  const int $attrValueName = env->GetArrayLength($inputName);
                 |  TFE_OpSetAttrInt(op.get(), "$attrName", static_cast<int64_t>($attrValueName));""".stripMargin
            inferredAttributeExpressions.append(attrValueExpression)
            attributeExpressions.update(attrName, attrValueName)
          }
        case "type" =>
          if (!attributeExpressions.contains(attrName)) {
            val inputName = inputs(inputIndices.head)._2
            val inputType = argumentTypes(inputs(inputIndices.head)._1)
            val attrValueName = s"${inputName}_attr_$attrName"
            inputType match {
              case "tensor" =>
                val tensorHandle = s"${attrValueName}_${inputName}_handle"
                val attrValueExpression =
                  s"""
                     |
                     |  REQUIRE_HANDLE($tensorHandle, TFE_TensorHandle, $inputName, $cNullValuePlaceholder);
                     |  const TF_DataType $attrValueName = TFE_TensorHandleDataType($tensorHandle);
                     |  TFE_OpSetAttrType(op.get(), "$attrName", $attrValueName);""".stripMargin
                inferredAttributeExpressions.append(attrValueExpression)
                attributeExpressions.update(attrName, attrValueName)
              case "list(tensor)" =>
                val tensorElems = s"${inputName}_attr_${attrName}_elems"
                val attrValueExpression =
                  s"""
                     |
                     |  jlong *$tensorElems = env->GetLongArrayElements($inputName, nullptr);
                     |  REQUIRE_HANDLE(${tensorElems}_head, TFE_TensorHandle, $tensorElems[0], $cNullValuePlaceholder);
                     |  const TF_DataType $attrValueName = TFE_TensorHandleDataType(${tensorElems}_head);
                     |  TFE_OpSetAttrType(op.get(), "$attrName", $attrValueName);
                     |  env->ReleaseLongArrayElements($inputName, $tensorElems, JNI_ABORT);""".stripMargin
                inferredAttributeExpressions.append(attrValueExpression)
                attributeExpressions.update(attrName, attrValueName)
            }
          }
        case "list(type)" => // TODO: Manage "list(type)" attributes.
      }
    })

    // Add attribute expressions for the non-inferrable attributes (i.e., the parameters).
    parameters.foreach(p => attributeExpressions.update(p._1, p._2))

    (argumentTypes.toMap,
        inputs.toList,
        parameters.toList,
        parameterDefaults.toMap,
        inferrableAttributes.toMap,
        inferrableAttributeToInputs.mapValues(_.toList).toMap,
        inferredAttributeExpressions.toList,
        attributeExpressions.toMap)
  }

  private[this] def numOutputsExpression(): String = {
    // If output i is list output, outputSizes[i] will be set to a string with the Scala expression that will evaluate
    // to its length. outputSizes[i] is empty for non-list outputs.
    var numFixedOutputs = 0
    val numOutputsExpression = StringBuilder.newBuilder
    opDef.getOutputArgList.asScala.foreach(outputArg => {
      if (outputArg.getNumberAttr.nonEmpty) {
        if (numOutputsExpression.nonEmpty) numOutputsExpression.append(" + ")
        numOutputsExpression.append(attributeExpressions(outputArg.getNumberAttr))
      } else if (outputArg.getTypeListAttr.nonEmpty) {
        if (numOutputsExpression.nonEmpty) numOutputsExpression.append(" + ")
        val inferred = inferrableAttributes.getOrElse(
          outputArg.getTypeListAttr, attributeExpressions(outputArg.getTypeListAttr))
        if (argumentTypes(inferred).startsWith("list("))
          numOutputsExpression.append(s"env->GetArrayLength($inferred)")
        else
          numOutputsExpression.append(inferred)
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

  private[this] def addInputs(codeBuilder: mutable.StringBuilder): Unit = {
    inputs.foreach(param => {
      val inputName = param._2
      val inputType = argumentTypes(param._1)
      inputType match {
        case "tensor" =>
          codeBuilder.append(
            s"""
               |
               |  REQUIRE_HANDLE(${inputName}_handle, TFE_TensorHandle, $inputName, $cNullValuePlaceholder);
               |  TFE_OpAddInput(op.get(), ${inputName}_handle, status.get());
               |  CHECK_STATUS(env, status.get(), $cNullValuePlaceholder);""".stripMargin)
        case "list(tensor)" =>
          val numTensors = s"${inputName}_num_tensors"
          val tensorElems = s"${inputName}_elems"
          codeBuilder.append(
            s"""
               |
               |  const int $numTensors = env->GetArrayLength($inputName);
               |  jlong *$tensorElems = env->GetLongArrayElements($inputName, nullptr);
               |  for (int i = 0; i < $numTensors; ++i) {
               |    REQUIRE_HANDLE(tensor_handle, TFE_TensorHandle, $tensorElems[i], 0);
               |    TFE_OpAddInput(op.get(), tensor_handle, status.get());
               |    CHECK_STATUS(env, status.get(), $cNullValuePlaceholder);
               |  }
               |  env->ReleaseLongArrayElements($inputName, $tensorElems, JNI_ABORT);""".stripMargin)
        case _ => throw new IllegalArgumentException(s"Invalid input argument type '$inputType'.")
      }
    })
  }

  private[this] def addInferredAttributes(codeBuilder: mutable.StringBuilder): Unit = {
    // Add all attribute value inference code and validate consistence of inferred attribute values with the inputs.
    inferredAttributeExpressions.foreach(codeBuilder.append)
    inferrableAttributeToInputs.foreach({ case (attrName, inputIndices) =>
      argumentTypes(attrName) match {
        case "int" =>
          // Inferred int attributes are the lengths of input lists.
          // Check that those inputs are lists of the same length.
          inputIndices.tail.foreach(inputIndex => {
            val inputName = inputs(inputIndex)._2
            val attrValueName = attributeExpressions(attrName)
            codeBuilder.append(
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
                 |          << "' of argument '${inferrableAttributes(attrName)}'";
                 |      throw_exception(env, jvm_illegal_argument_exception, error_msg.str().c_str());
                 |  }""".stripMargin)
          })
        case "type" =>
          inputIndices.tail.foreach(input => {
            val inputName = inputs(input)._2
            val inputType = argumentTypes(inputs(input)._1)
            inputType match {
              case "tensor" =>
                val attrValueName = attributeExpressions(attrName)
                val tensorHandle = s"${attrValueName}_${inputName}_handle"
                codeBuilder.append(
                  s"""
                     |
                     |  REQUIRE_HANDLE($tensorHandle, TFE_TensorHandle, $inputName, $cNullValuePlaceholder);
                     |  const TF_DataType ${inputName}_attr_$attrName = TFE_TensorHandleDataType($tensorHandle);
                     |  if ($attrValueName != ${inputName}_attr_$attrName) {
                     |      std::stringstream error_msg;
                     |      error_msg
                     |          << "Argument '$inputName' of '$name' op with data type '"
                     |          << ${inputName}_attr_$attrName
                     |          << "' must match data type '"
                     |          << $attrValueName
                     |          << "' of argument '${inferrableAttributes(attrName)}'";
                     |      throw_exception(env, jvm_illegal_argument_exception, error_msg.str().c_str());
                     |  }""".stripMargin)
              case "list(tensor)" =>
                val attrValueName = attributeExpressions(attrName)
                val numTensors = s"${inputName}_attr_${attrName}_num_tensors"
                val tensorElems = s"${inputName}_attr_${attrName}_elems"
                codeBuilder.append(
                  s"""
                     |
                     |  const int $numTensors = env->GetArrayLength($inputName);
                     |  $tensorElems = env->GetLongArrayElements($inputName, nullptr);
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
                     |          << "' of argument '${inferrableAttributes(attrName)}'";
                     |      throw_exception(env, jvm_illegal_argument_exception, error_msg.str().c_str());
                     |    }
                     |  }
                     |  env->ReleaseLongArrayElements($inputName, $tensorElems, JNI_ABORT);""".stripMargin)
              case _ => throw new IllegalArgumentException(s"Invalid input argument type '$inputType'.")
            }
          })
        case "list(type)" => // TODO: Manage "list(type)" attributes.
      }
    })
  }

  @throws[IllegalArgumentException]
  private[this] def addAttributes(
      codeBuilder: mutable.StringBuilder, deallocationBuilder: mutable.StringBuilder): Unit = {
    parameters.foreach(parameter => {
      val attrName = parameter._1
      val attrType = argumentTypes(attrName)
      val value = attributeExpressions(attrName)
      attrType match {
        case "string" =>
          codeBuilder.append(
            s"""
               |
               |  jbyte *${attrName}_c_value = env->GetByteArrayElements($value, nullptr);
               |  TFE_OpSetAttrString(op.get(), "$attrName", reinterpret_cast<const char *>(${attrName}_c_value));
               |  env->ReleaseByteArrayElements($value, ${attrName}_c_value, JNI_ABORT);""".stripMargin)
        case "int" =>
          codeBuilder.append(
            s"""
               |
               |  TFE_OpSetAttrInt(op.get(), "$attrName", static_cast<int64_t>($value));""".stripMargin)
        case "float" =>
          codeBuilder.append(
            s"""
               |
               |  TFE_OpSetAttrFloat(op.get(), "$attrName", static_cast<float>($value));""".stripMargin)
        case "bool" =>
          codeBuilder.append(
            s"""
               |
               |  TFE_OpSetAttrBool(op.get(), "$attrName", static_cast<unsigned char>($value));""".stripMargin)
        case "type" =>
          codeBuilder.append(
            s"""
               |
               |  TFE_OpSetAttrType(op.get(), "$attrName", static_cast<TF_DataType>($value));""".stripMargin)
        case "shape" =>
          codeBuilder.append(
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
               |  CHECK_STATUS(env, status.get(), $cNullValuePlaceholder);""".stripMargin)
        case "tensor" => throw new UnsupportedOperationException(s"Unsupported attribute type '$attrType'.")
        case "func" => throw new UnsupportedOperationException(s"Unsupported attribute type '$attrType'.")
        case "list(string)" =>
          codeBuilder.append(
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
          codeBuilder.append(
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
          codeBuilder.append(
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
          codeBuilder.append(
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
          codeBuilder.append(
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
          codeBuilder.append(
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
               |  CHECK_STATUS(env, status.get(), $cNullValuePlaceholder);""".stripMargin)
        case "list(tensor)" => throw new UnsupportedOperationException(s"Unsupported attribute type '$attrType'.")
        case "list(func)" => throw new UnsupportedOperationException(s"Unsupported attribute type '$attrType'.")
        case _ => throw new IllegalArgumentException(s"Invalid attribute type '$attrType'.")
      }
    })
  }

  private[this] def addExecute(
      codeBuilder: mutable.StringBuilder, deallocationBuilder: mutable.StringBuilder, numOutputs: String): Unit = {
    codeBuilder.append(
      s"""
         |
         |  const int num_outputs = $numOutputs;
         |  std::unique_ptr<TFE_TensorHandle* []> outputs(new TFE_TensorHandle* [num_outputs]);
         |  std::unique_ptr<int[]> actual_num_outputs(new int[1] {1});
         |  TFE_Execute(op.get(), outputs.get(), actual_num_outputs.get(), status.get());
         |  CHECK_STATUS(env, status.get(), $cNullValuePlaceholder);
         |${if (deallocationBuilder.nonEmpty) "\n" + deallocationBuilder.mkString + "\n" else ""}""".stripMargin)
    numOutputs match {
      case "0" => codeBuilder.append(
        s"""
           |  return;""".stripMargin)
      case "1" =>
        codeBuilder.append(
          s"""
             |  return reinterpret_cast<jlong>(outputs[0]);""".stripMargin)
      case _ =>
        codeBuilder.append(
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
    val opCode = opDefs.map(OpGenerator(_).generateCode(jniObjectName))
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

  private[this] val reservedKeywords = Set(
    "abstract", "case", "catch", "class", "def", "do", "else", "extends", "false", "final", "finally", "for", "forSome",
    "if", "implicit", "import", "lazy", "macro", "match", "new", "null", "object", "override", "package", "private",
    "protected", "return", "sealed", "super", "this", "throw", "trait", "try", "true", "type", "val", "var", "while",
    "with", "yield")

  private[OpGenerator] def processName(name: String): String = {
    val c = name.toCharArray.toBuffer[Char]
    c(0) = Character.toLowerCase(c(0))
    var processedName = new String(c.toArray)
    if (reservedKeywords.contains(processedName)) processedName = s"_$processedName"
    processedName
  }

  private[OpGenerator] val typeToScalaType: Map[String, String] = Map(
    // "func"
    // "list(func)"
    "string" -> "Array[Byte]",
    "int" -> "Long",
    "float" -> "Float",
    "bool" -> "Boolean",
    "type" -> "Int",
    "shape" -> "Array[Long]",
    "tensor" -> "Long",
    "list(string)" -> "Array[Array[Byte]]",
    "list(int)" -> "Array[Long]",
    "list(float)" -> "Array[Float]",
    "list(bool)" -> "Array[Boolean]",
    "list(type)" -> "Array[Int]",
    "list(shape)" -> "Array[Array[Long]]",
    "list(tensor)" -> "Array[Long]")

  private[OpGenerator] val scalaTypeToShortJni: Map[String, String] = Map(
    "Array[Byte]" -> "[B",
    "Int" -> "I",
    "Long" -> "J",
    "Float" -> "F",
    "Boolean" -> "Z",
    "Array[Array[Byte]]" -> "[L",
    "Array[Int]" -> "[I",
    "Array[Long]" -> "[J",
    "Array[Float]" -> "[F",
    "Array[Boolean]" -> "[Z")

  private[OpGenerator] val typeToShortJni: Map[String, String] = typeToScalaType.mapValues(scalaTypeToShortJni)

  private[OpGenerator] val scalaTypeToJni: Map[String, String] = Map(
    "Array[Byte]" -> "jbyteArray",
    "Int" -> "jint",
    "Long" -> "jlong",
    "Float" -> "jfloat",
    "Boolean" -> "jboolean",
    "Array[Array[Byte]]" -> "jobjectArray",
    "Array[Int]" -> "jintArray",
    "Array[Long]" -> "jlongArray",
    "Array[Float]" -> "jfloatArray",
    "Array[Boolean]" -> "jbooleanArray")

  private[OpGenerator] val typeToJni: Map[String, String] = typeToScalaType.mapValues(scalaTypeToJni)

  private[OpGenerator] def attrValueToScala(attr: AttrDef, value: AttrValue): Option[String] = {
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

  private[OpGenerator] def escapeString(string: String): String = {
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

  private[OpGenerator] def throwUnsupportedException(name: String, message: String): Unit = {
    throw new UnsupportedOperationException(s"Op '$name' is not supported on the Scala API. Error message: $message")
  }
}
