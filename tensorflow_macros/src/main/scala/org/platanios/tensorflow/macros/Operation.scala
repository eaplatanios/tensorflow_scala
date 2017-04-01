//package org.platanios.tensorflow.macros
//
//import org.platanios.tensorflow.jni.{DataType, TensorFlow, Operation => NativeOperation}
//import org.tensorflow.framework.AttrValue.ListValue
//import org.tensorflow.framework.OpDef.{ArgDef, AttrDef}
//import org.tensorflow.framework._
//import com.typesafe.scalalogging.Logger
//import org.slf4j.LoggerFactory
//
//import scala.annotation.StaticAnnotation
//import scala.collection.JavaConverters._
//import scala.collection.mutable.ArrayBuffer
//import scala.meta._
//
///**
//  * @author Emmanouil Antonios Platanios
//  */
//object Operation {
//  private val logger: Logger = Logger(LoggerFactory.getLogger("TensorFlow / Macros / Operation"))
//
//  private def allOpDefs: List[OpDef] = {
//    val opList: List[OpDef] = OpList.parseFrom(NativeOperation.allOps).getOpList.asScala.toList
//    opList.filter(opDef =>
//                    opDef.getDeprecation.getVersion == 0
//                        && !opDef.getName.startsWith("_")
//                        && !opDef.getName.endsWith("Grad"))
//  }
//
//  private def convertName(name: String): String = name(0).toLower + name.tail
//
//  private def convertDataTypeNumber(dataTypeNumber: Int): Either[Exception, String] = dataTypeNumber match {
//    case DataType.float.cValue => Right("DataType.float")
//    case DataType.double.cValue => Right("DataType.double")
//    case DataType.int32.cValue => Right("DataType.int32")
//    case DataType.uint8.cValue => Right("DataType.uint8")
//    case DataType.string.cValue => Right("DataType.string")
//    case DataType.int64.cValue => Right("DataType.int64")
//    case DataType.boolean.cValue => Right("DataType.boolean")
//    case value => Left(new IllegalArgumentException(
//      s"Data type $value is not recognized in Scala (TensorFlow version ${TensorFlow.version})."))
//  }
//
//  private def convertTensorShapeProto(tensorShapeProto: TensorShapeProto): String = {
//    if (tensorShapeProto.getUnknownRank)
//      "Shape(Array.empty[Long])"
//    else
//      s"Shape(Array[Long](${tensorShapeProto.getDimList.asScala.map(_.getSize).mkString(", ")}))"
//  }
//
//  private case class Op(
//      name: String,
//      summary: Option[String],
//      description: Option[String],
//      optimizations: Option[String],
//      inputArgs: Seq[Arg],
//      outputArgs: Seq[Arg])
//
//  private case class Arg(
//      name: String,
//      description: Option[String])
//
//  private case class Attr(
//      name: String,
//      attrType: String,
//      description: Option[String],
//      defaultValue: Option[String],
//      additionalInformation: Option[String])
//
//  private case class Deprecation(version: Int, explanation: Option[String])
//
//  private def parseOpDef(opDef: OpDef): Op = {
//    val name: String = convertName(opDef.getName)
//    val summary: Option[String] = Option(opDef.getSummary)
//    val description: Option[String] = Option(opDef.getDescription)
//    val optimizations: Option[String] = {
//      val optimizations: ArrayBuffer[String] = ArrayBuffer[String]()
//      if (opDef.getIsCommutative)
//        optimizations += "- Commutative: This operation is commutative (i.e., \"op(a,b) == op(b,a)\" for all " +
//            "inputs a and b)."
//      if (opDef.getIsAggregate)
//        optimizations += "- Aggregate: This operation accepts N >= 2 inputs and produces 1 output, all of " +
//            "the same type. The operation is be associative and commutative, and produces output with the " +
//            "same shape as its input. The optimizer may replace an aggregate op taking its input from " +
//            "multiple devices with a tree of aggregate ops that aggregate locally within each device (and " +
//            "possibly within groups of nearby devices) before communicating."
//      if (opDef.getIsStateful)
//        optimizations += "- Stateful: By default Ops may be moved between devices. This is a stateful op, " +
//            "meaning that it should either not be moved, or it should only be moved if its state can also be " +
//            "moved (e.g. via some sort of save / restore). Stateful ops are guaranteed to never be optimized " +
//            "away by Common Subexpression Elimination (CSE)."
//      if (opDef.getAllowsUninitializedInput)
//        optimizations += "- Allows Uninitialized Input: By default, all inputs to an Op must be initialized " +
//            "tensors. This op may initialize tensors for the first time and thus may take an uninitialized " +
//            "tensor as input."
//      if (optimizations.nonEmpty)
//        Some(s"The following optimizations apply to this op:\n\t${optimizations.mkString("\n\t")}")
//      else
//        None
//    }
//    val attributes: Seq[Attr] = opDef.getAttrList.asScala.map(parseAttr).filter(_.isRight).map(_.right.get)
//    val inputArgs: Seq[Arg] = opDef.getInputArgList.asScala.map(parseArgDef(_, attributes))
//    val outputArgs: Seq[Arg] = opDef.getOutputArgList.asScala.map(parseArgDef(_, attributes))
//    val deprecation: Deprecation = {
//      val deprecation: OpDeprecation = opDef.getDeprecation
//      Deprecation(version = deprecation.getVersion, explanation = Option(deprecation.getExplanation))
//    }
//    null
//  }
//
//  private def parseAttr(attrDef: AttrDef): Either[Exception, Attr] = {
//    val name: String = convertName(attrDef.getName)
//
//    def parseAttrType(attrType: String, isList: Boolean = false): Either[Exception, String] = {
//      attrType match {
//        case "string" => Right("String")
//        case "int" => Right("Int")
//        case "float" => Right("Float")
//        case "bool" => Right("Boolean")
//        case "type" => Right("DataType[_]")
//        case "shape" => Right("Shape")
//        case "tensor" => Right("Tensor[_]")
//        case list if list.startsWith("list") =>
//          if (isList)
//            Left(new IllegalArgumentException(
//              s"Error while parsing the type for attribute $name: Nested lists are not supported as attribute types " +
//                  s"(TensorFlow version ${TensorFlow.version})."))
//          else
//            Right(s"Array[${parseAttrType(attrType.substring(5, attrType.length - 1), isList = true)}]")
//        case "func" => Right("Func")
//        case "placeholder" =>
//          if (isList)
//            Left(new IllegalArgumentException(
//              s"Error while parsing the type for attribute $name: Lists of placeholders are not supported as " +
//                  s"attribute types (TensorFlow version ${TensorFlow.version})."))
//          else
//            Right("Placeholder")
//        case attributeType => Left(new IllegalArgumentException(
//          s"Error while parsing the type for attribute $name: Attribute type '$attributeType' is not supported " +
//              s"(TensorFlow version ${TensorFlow.version})."))
//      }
//    }
//
//    def parseAttrValue(
//        attrValue: AttrValue, attrType: Either[Exception, String]): Either[Exception, Option[String]] = {
//      if (attrValue.getValueCase.getNumber == 0) {
//        Right(None)
//      } else {
//        attrType match {
//          case Right("String") => Right(Some(attrValue.getS.toStringUtf8))
//          case Right("Int") => Right(Some(attrValue.getI.toString))
//          case Right("Float") => Right(Some(attrValue.getF.toString))
//          case Right("Boolean") => Right(Some(attrValue.getB.toString))
//          case Right("DataType[_]") => convertDataTypeNumber(attrValue.getType.getNumber)
//          case Right("Shape") => Right(Some(convertTensorShapeProto(attrValue.getShape)))
//          case Right("Tensor[_]") => Left(new IllegalArgumentException(
//            s"Error while parsing the default value for attribute $name: Attribute type 'Tensor[_]' is not supported " +
//                s"in Scala (TensorFlow version ${TensorFlow.version})."))
//          case Right(list) if list.startsWith("Array") =>
//            val valueList: ListValue = attrValue.getList
//            list.substring(5, list.length - 1) match {
//              case "String" => Right(Some(
//                s"Array[String](${valueList.getSList.asScala.map(_.toStringUtf8).mkString(", ")})"))
//              case "Int" => Right(Some(s"Array[Int](${valueList.getIList.asScala.map(_.toString).mkString(", ")})"))
//              case "Float" => Right(Some(s"Array[Float](${valueList.getFList.asScala.map(_.toString).mkString(", ")})"))
//              case "Boolean" => Right(Some(
//                s"Array[Boolean](${valueList.getBList.asScala.map(_.toString).mkString(", ")})"))
//              case "DataType[_]" =>
//                val dataTypes: Seq[Either[Exception, Option[String]]] =
//                  valueList.getTypeList.asScala.map(t => convertDataTypeNumber(t.getNumber))
//                // TODO: Somehow group all exceptions instead of only returning the first one.
//                dataTypes
//                    .find(_.isLeft)
//                    .getOrElse(Right(Some(s"Array[DataType[_]](${dataTypes.map(_.right).mkString(", ")})")))
//              case "Shape" => Right(Some(
//                s"Array[Shape](${valueList.getShapeList.asScala.map(s => convertTensorShapeProto(s)).mkString(", ")})"))
//              case "Tensor[_]" => Left(new IllegalArgumentException(
//                s"Error while parsing the default value for attribute $name: List type 'Tensor[_]' is not supported " +
//                    s"in Scala (TensorFlow version ${TensorFlow.version})."))
//              case "Func" => Left(new IllegalArgumentException(
//                s"Error while parsing the default value for attribute $name: List type 'Func' is not supported in " +
//                    s"Scala (TensorFlow version ${TensorFlow.version})."))
//              case listType => Left(new IllegalArgumentException(
//                s"Error while parsing the default value for attribute $name: List type '$listType' is not supported " +
//                    s"in Scala (TensorFlow version ${TensorFlow.version})."))
//            }
//          case Right("Func") => Left(new IllegalArgumentException(
//            s"Error while parsing the default value for attribute $name: Attribute type 'Func' is not supported in " +
//                s"Scala (TensorFlow version ${TensorFlow.version})."))
//          case Right("Placeholder") => Left(new IllegalArgumentException(
//            s"Error while parsing the default value for attribute $name: Attribute type 'Placeholder' is not " +
//                s"supported in Scala (TensorFlow version ${TensorFlow.version})."))
//          case Right(attributeType) => Left(new IllegalArgumentException(
//            s"Error while parsing the default value for attribute $name: Attribute type '$attributeType' is not " +
//                s"supported in Scala (TensorFlow version ${TensorFlow.version})."))
//          case Left(exception) => Left(exception)
//        }
//      }
//    }
//
//    val attrType: Either[Exception, String] = parseAttrType(attrDef.getType)
//    val description: Option[String] = Option(attrDef.getDescription)
//    val defaultValue: Either[Exception, Option[String]] = parseAttrValue(attrDef.getDefaultValue, attrType)
//    val additionalInformation: Option[String] = {
//      var info: String = ""
//      if (attrDef.getHasMinimum) {
//        if (attrType.getOrElse("Exception").startsWith("Array"))
//          info += s"The minimum allowed length for this list is ${attrDef.getMinimum}."
//        else
//          info += s"The minimum allowed value for this argument is ${attrDef.getMinimum}."
//      }
//      val allowedValues: Either[Exception, Option[String]] =
//        parseAttrValue(attrDef.getAllowedValues, attrType = attrType.map(t => s"Array[$t]"))
//      if (allowedValues.isRight) {
//        attrType match {
//          case Right(typeName) => allowedValues.map(_.map(t => t.substring(7 + typeName.length, t.length - 1)))
//        }
//        if (info.length > 0)
//          info += " "
//        info += s"The set of allowed values for this argument is [${allowedValues.right.get.get}]."
//      }
//      if (info.length > 0)
//        Some(info)
//      else
//        None
//    }
//    // TODO: Combine exceptions somehow.
//    if (attrType.isLeft)
//      Left(attrType.left.get)
//    else if (defaultValue.isLeft)
//      Left(defaultValue.left.get)
//    else
//      Right(Attr(
//        name = name, attrType = attrType.right.get, description = description, defaultValue = defaultValue.right.get,
//        additionalInformation = additionalInformation))
//  }
//
//  private def parseArgDef(argDef: ArgDef, attributes: Seq[Attr]): Arg = {
//    val name: String = convertName(argDef.getName)
//    val description: Option[String] = Option(argDef.getDescription)
//    val dataType: String = {
//      val dataTypeNumber: Int = argDef.getType.getNumber
//      if (dataTypeNumber == 0) {
//        var dataTypeName: String = ""
//        if (argDef.getTypeAttr != "") {
//          dataTypeName = attributes.find(_.name == argDef.getTypeAttr).get.name
//          if (argDef.getNumberAttr != "")
//            dataTypeName = s"Array[$dataTypeName]"
//        } else if (argDef.getTypeListAttr != "") {
//          dataTypeName = s"Array[${attributes.find(_.name == argDef.getTypeListAttr).get.name}]"
//        }
//        dataTypeName
//      } else {
//        convertDataTypeNumber(dataTypeNumber)
//      }
//    }
//
//    Arg(name = name, description = description)
//  }
//
//  class addOps extends StaticAnnotation {
//    inline def apply(defn: Any): Any = meta {
//      defn match {
//        case q"object $name { ..$stats }" =>
//          val ops: Seq[Op] = allOpDefs.map(parseOpDef)
//
//          println(stats)
//
//
//          q"object $name { ..$stats }"
//        //        val main = q"def main(args: Array[String]): Unit = { ..$stats }"
//        //        q"object $name { $main }"
//        case _ =>
//          abort("@addOps must annotate an object.")
//      }
//    }
//  }
//}
