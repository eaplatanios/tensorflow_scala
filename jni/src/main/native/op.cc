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

#include "exception.h"
#include "op.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string.h>
#include <stdint.h>
#include <iostream>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

namespace {
template <class T>
T* requireHandleImpl(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(T*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throw_exception(
        env, tf_invalid_argument_exception,
        "close() has been called on the Graph this Operation was a part of");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

TF_Operation* require_operation_handle(JNIEnv *env, jlong handle) {
  return requireHandleImpl<TF_Operation>(env, handle);
}

TF_Graph* require_graph_handle(JNIEnv *env, jlong handle) {
  return requireHandleImpl<TF_Graph>(env, handle);
}

TF_OperationDescription* requireOperationDescriptionHandle(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throw_exception(env, tf_invalid_argument_exception,
                    "Operation has already been built");
    return 0;
  }
  return reinterpret_cast<TF_OperationDescription*>(handle);
}

bool resolveOutput(JNIEnv* env, jlong op_handle, jint index, TF_Output* out) {
  if (op_handle == 0) {
    throw_exception(env, tf_invalid_argument_exception,
                    "close() was called on the Graph");
    return false;
  }
  out->oper = reinterpret_cast<TF_Operation*>(op_handle);
  out->index = static_cast<int>(index);
  return true;
}

TF_Tensor* requireTensor(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throw_exception(env, tf_invalid_argument_exception,
                    "close() has been called on the Tensor");
    return nullptr;
  }
  return reinterpret_cast<TF_Tensor*>(handle);
}

const char* attrTypeToString(TF_AttrType type, char is_list) {
  std::string typeName;
  switch(type) {
    case TF_ATTR_STRING:
      typeName = "String";
      break;
    case TF_ATTR_INT:
      typeName = "Int";
      break;
    case TF_ATTR_FLOAT:
      typeName = "Float";
      break;
    case TF_ATTR_BOOL:
      typeName = "Boolean";
      break;
    case TF_ATTR_TYPE:
      typeName = "DataType";
      break;
    case TF_ATTR_SHAPE:
      typeName = "Shape";
      break;
    case TF_ATTR_TENSOR:
      typeName = "Tensor";
      break;
    case TF_ATTR_PLACEHOLDER:
      typeName = "Placeholder";
      break;
    case TF_ATTR_FUNC:
      typeName = "Function";
      break;
    default:
      typeName = "Unknown";
      break;
  }
  if (is_list == 1)
    return ("List[" + typeName + "]").c_str();
  return typeName.c_str();
}
}  // namespace

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Op_00024_name(JNIEnv* env,
                                                                             jobject object,
                                                                             jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationName(op));
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Op_00024_opType(JNIEnv* env,
                                                                                 jobject object,
                                                                                 jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationOpType(op));
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Op_00024_device(JNIEnv* env,
                                                                                 jobject object,
                                                                                 jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationDevice(op));
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Op_00024_numInputs(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumInputs(op);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Op_00024_numControlInputs(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumControlInputs(op);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Op_00024_numOutputs(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumOutputs(op);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Op_00024_numControlOutputs(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumControlOutputs(op);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Op_00024_numConsumers(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle,
                                                                                jint output_index) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return 0;
  TF_Output output{op, output_index};
  return TF_OperationOutputNumConsumers(output);
}

JNIEXPORT jobject JNICALL Java_org_platanios_tensorflow_jni_Op_00024_input(JNIEnv* env,
                                                                                  jobject object,
                                                                                  jlong handle,
                                                                                  jint input_index) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;

  int num_inputs = TF_OperationNumInputs(op);
  if (input_index < 0 || input_index >= num_inputs) {
    throw_exception(
        env, tf_invalid_argument_exception,
        "invalid input index (%d) for an operation that has %d inputs",
        input_index, num_inputs);
    return 0;
  }

  TF_Output output = TF_OperationInput(TF_Input{op, input_index});
  jclass outputClass = env->FindClass("org/platanios/tensorflow/jni/Output");
  jmethodID outputClassConstructor = env->GetStaticMethodID(
    outputClass, "apply", "(JI)Lorg/platanios/tensorflow/jni/Output;");
  return env->CallStaticObjectMethod(
    outputClass, outputClassConstructor, reinterpret_cast<jlong>(output.oper), output.index);
}

JNIEXPORT jobject JNICALL Java_org_platanios_tensorflow_jni_Op_00024_inputs(
    JNIEnv* env, jobject object, jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;

  int num_inputs = TF_OperationNumInputs(op);
  jclass output_class = env->FindClass("org/platanios/tensorflow/jni/Output");
  jmethodID output_class_constructor = env->GetStaticMethodID(
      output_class, "apply", "(JI)Lorg/platanios/tensorflow/jni/Output;");
  jobjectArray array = env->NewObjectArray(num_inputs, output_class, nullptr);
  for (int i = 0; i < num_inputs; ++i) {
    TF_Output output = TF_OperationInput(TF_Input{op, i});
    jobject joutput = env->CallStaticObjectMethod(
      output_class, output_class_constructor, reinterpret_cast<jlong>(output.oper), output.index);
    env->SetObjectArrayElement(array, i, joutput);
  }
  return array;
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_controlInputs(JNIEnv* env,
                                                                                             jobject object,
                                                                                             jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;

  int numControlInputs = TF_OperationNumControlInputs(op);
  std::unique_ptr<TF_Operation*[]> controlInputs(new TF_Operation*[numControlInputs]);
  TF_OperationGetControlInputs(op, controlInputs.get(), numControlInputs);
  jlongArray ret = env->NewLongArray(numControlInputs);
  jlong* ops = env->GetLongArrayElements(ret, nullptr);
  for (int i = 0; i < numControlInputs; ++i) {
    ops[i] = reinterpret_cast<jlong>(controlInputs[i]);
  }
  env->ReleaseLongArrayElements(ret, ops, 0);
  return ret;
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_controlOutputs(JNIEnv* env,
                                                                                             jobject object,
                                                                                             jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;

  int numControlOutputs = TF_OperationNumControlOutputs(op);
  std::unique_ptr<TF_Operation*[]> controlOutputs(new TF_Operation*[numControlOutputs]);
  TF_OperationGetControlOutputs(op, controlOutputs.get(), numControlOutputs);
  jlongArray ret = env->NewLongArray(numControlOutputs);
  jlong* ops = env->GetLongArrayElements(ret, nullptr);
  for (int i = 0; i < numControlOutputs; ++i) {
    ops[i] = reinterpret_cast<jlong>(controlOutputs[i]);
  }
  env->ReleaseLongArrayElements(ret, ops, 0);
  return ret;
}

JNIEXPORT jobjectArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_consumers(JNIEnv* env,
                                                                                  jobject object,
                                                                                  jlong handle,
                                                                                  jint output_index) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;

  TF_Output output{op, output_index};
  int numConsumers = TF_OperationOutputNumConsumers(output);
  std::unique_ptr<TF_Input[]> consumers(new TF_Input[numConsumers]);
  TF_OperationOutputConsumers(output, consumers.get(), numConsumers);
  jclass outputClass = env->FindClass("org/platanios/tensorflow/jni/Output");
  jmethodID outputClassConstructor = env->GetStaticMethodID(
    outputClass, "apply", "(JI)Lorg/platanios/tensorflow/jni/Output;");
  jobjectArray ret = env->NewObjectArray(numConsumers, outputClass, NULL);
  for (int i = 0; i < numConsumers; ++i) {
    env->SetObjectArrayElement(
      ret, i, env->CallStaticObjectMethod(
        outputClass, outputClassConstructor, reinterpret_cast<jlong>(consumers[i].oper), consumers[i].index));
  }
  return ret;
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Op_00024_inputDataType(
    JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jint input_index) {
  TF_Graph* graph = require_graph_handle(env, graph_handle);
  if (graph == nullptr) return 0;
  TF_Operation* op = require_operation_handle(env, op_handle);
  if (op == nullptr) return 0;

  int num_inputs = TF_OperationNumInputs(op);
  if (input_index < 0 || input_index >= num_inputs) {
    throw_exception(
        env, tf_invalid_argument_exception,
        "invalid input index (%d) for an operation that has %d inputs",
        input_index, num_inputs);
    return 0;
  }

  return static_cast<jint>(TF_OperationInputType(TF_Input{op, input_index}));
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Op_00024_outputDataType(
    JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jint output_index) {
  TF_Graph* graph = require_graph_handle(env, graph_handle);
  if (graph == nullptr) return 0;
  TF_Operation* op = require_operation_handle(env, op_handle);
  if (op == nullptr) return 0;

  int num_outputs = TF_OperationNumOutputs(op);
  if (output_index < 0 || output_index >= num_outputs) {
    throw_exception(
        env, tf_invalid_argument_exception,
        "invalid output index (%d) for an operation that has %d outputs",
        output_index, num_outputs);
    return 0;
  }

  return static_cast<jint>(TF_OperationOutputType(TF_Output{op, output_index}));
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_shape(
    JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jint output_index) {
  TF_Graph *graph = require_graph_handle(env, graph_handle);
  if (graph == nullptr) return nullptr;
  TF_Operation *op = require_operation_handle(env, op_handle);
  if (op == nullptr) return nullptr;

  int num_outputs = TF_OperationNumOutputs(op);
  if (output_index < 0 || output_index >= num_outputs) {
    throw_exception(
        env, tf_invalid_argument_exception,
        "invalid output index (%d) for an operation that has %d outputs",
        output_index, num_outputs);
    return nullptr;
  }

  TF_Output output{op, output_index};
  TF_Status *status = TF_NewStatus();
  int num_dims = TF_GraphGetTensorNumDims(graph, output, status);
  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  if (num_dims < 0) return nullptr;
  static_assert(sizeof(jlong) == sizeof(int64_t), "Java long is not compatible with the TensorFlow C API");
  // One might have trivially wanted to do:
  // TF_GraphGetTensorShape(graph, output, static_cast<int64_t*>(dims), ...)
  // but on some platforms this fails with:
  // static_cast from 'jlong *' (aka 'long *') to 'int64_t *' (aka 'long long *') is not allowed
  // For now, we do the expensive but safe thing of copying.
  std::unique_ptr<int64_t[]> cdims(new int64_t[num_dims]);
  TF_GraphGetTensorShape(graph, output, cdims.get(), num_dims, status);
  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  jlongArray ret = env->NewLongArray(num_dims);
  jlong *dims = env->GetLongArrayElements(ret, nullptr);
  for (int i = 0; i < num_dims; ++i)
    dims[i] = static_cast<jlong>(cdims[i]);
  env->ReleaseLongArrayElements(ret, dims, 0);
  return ret;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setShape(
        JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jint output_index, jlongArray shape,
        jint num_dims) {
  TF_Graph *graph = require_graph_handle(env, graph_handle);
  if (graph == nullptr) return;
  TF_Operation *op = require_operation_handle(env, op_handle);
  if (op == nullptr) return;
  TF_Output output{op, output_index};
  std::unique_ptr<int64_t[]> dims;
  if (num_dims > 0) {
    dims.reset(new int64_t[num_dims]);
    jlong *elems = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i) {
      dims[i] = static_cast<int64_t>(elems[i]);
    }
    env->ReleaseLongArrayElements(shape, elems, JNI_ABORT);
  }
  TF_Status *status = TF_NewStatus();
  TF_GraphSetTensorShape(graph, output, dims.get(), num_dims, status);
  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return;
  }
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Op_00024_getAttrString(
        JNIEnv* env, jobject object, jlong opHandle, jstring attrName) {
  TF_Operation* op = require_operation_handle(env, opHandle);
  if (op == nullptr) return nullptr;
  const char* attrNameString = env->GetStringUTFChars(attrName, nullptr);
  TF_Status* status = TF_NewStatus();
  TF_AttrMetadata attr_metadata = TF_OperationGetAttrMetadata(op, attrNameString, status);

  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  if (attr_metadata.total_size < 0) return nullptr;
  if (attr_metadata.type != TF_ATTR_STRING || attr_metadata.is_list == 1)
    throw_exception(
        env, tf_invalid_argument_exception, "Attribute '%s' is not a string. It is a '%s', instead.",
        attrNameString, attrTypeToString(attr_metadata.type, attr_metadata.is_list));
  if (attr_metadata.total_size < 0) return nullptr;
  size_t string_size = static_cast<size_t>(attr_metadata.total_size);
  char* attrValue = new char[string_size + 1];
  attrValue[string_size] = '\0';
  status = TF_NewStatus();
  TF_OperationGetAttrString(op, attrNameString, attrValue, string_size, status);

  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  env->ReleaseStringUTFChars(attrName, attrNameString);
  return env->NewStringUTF(attrValue);
}

JNIEXPORT jobjectArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_getAttrStringList(
        JNIEnv* env, jobject object, jlong opHandle, jstring attrName) {
  TF_Operation *op = require_operation_handle(env, opHandle);
  if (op == nullptr) return nullptr;
  const char *attrNameString = env->GetStringUTFChars(attrName, nullptr);
  TF_Status *status = TF_NewStatus();
  TF_AttrMetadata attr_metadata = TF_OperationGetAttrMetadata(op, attrNameString, status);

  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  if (attr_metadata.total_size < 0) return nullptr;
  if (attr_metadata.type != TF_ATTR_STRING || attr_metadata.is_list == 0)
    throw_exception(
        env, tf_invalid_argument_exception,
        "Attribute '%s' is not a string list. It is a '%s', instead.",
        attrNameString, attrTypeToString(attr_metadata.type, attr_metadata.is_list));
  size_t storageSize = static_cast<size_t>(attr_metadata.total_size);
  int list_size = static_cast<int>(attr_metadata.list_size);
  if (list_size <= 0) return nullptr;
  void **attrValuePointers = new void *[list_size];
  size_t *attrValueLengths = new size_t[list_size];
  void *storage = new char[storageSize];
  status = TF_NewStatus();
  TF_OperationGetAttrStringList(
          op, attrNameString, attrValuePointers, attrValueLengths, list_size, storage, storageSize, status);

  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  jobjectArray ret;
  ret = env->NewObjectArray(list_size, env->FindClass("java/lang/String"), env->NewStringUTF(""));
  for (int i = 0; i < list_size; i++) {
    char *value = new char[attrValueLengths[i] + 1];
    strncpy(value, reinterpret_cast<const char *>(attrValuePointers[i]), attrValueLengths[i]);
    value[attrValueLengths[i]] = '\0';
    env->SetObjectArrayElement(ret, i, env->NewStringUTF(value));
  }
  env->ReleaseStringUTFChars(attrName, attrNameString);
  return ret;
}

#define DEFINE_GET_ATTR_SCALAR(name, jname, jtype, ctype, tf_type)                           \
  JNIEXPORT jtype JNICALL Java_org_platanios_tensorflow_jni_Op_00024_getAttr##name(          \
      JNIEnv* env, jobject object, jlong handle, jstring name) {                             \
    static_assert(                                                                           \
        sizeof(ctype) >= sizeof(jtype),                                                      \
        "Information loss when converting between Java and C types.");                       \
    TF_Operation *op = require_operation_handle(env, handle);                                \
    if (op == nullptr) return -1;                                                            \
    const char *attr_name = env->GetStringUTFChars(name, nullptr);                           \
    TF_Status *status = TF_NewStatus();                                                      \
    TF_AttrMetadata attr_metadata = TF_OperationGetAttrMetadata(op, attr_name, status);      \
                                                                                             \
    if (!throw_exception_if_not_ok(env, status)) {                                           \
      TF_DeleteStatus(status);                                                               \
      return -1;                                                                             \
    }                                                                                        \
    TF_DeleteStatus(status);                                                                 \
                                                                                             \
    if (attr_metadata.type != tf_type || attr_metadata.is_list == 1)                         \
      throw_exception(                                                                       \
          env, tf_invalid_argument_exception,                                                \
          "Attribute '%s' is not a %s. It is a '%s', instead.",                              \
          name, attr_name, attrTypeToString(attr_metadata.type, attr_metadata.is_list));     \
                                                                                             \
    ctype *value = new ctype;                                                                \
    status = TF_NewStatus();                                                                 \
    TF_OperationGetAttr##name(op, attr_name, value, status);                                 \
    env->ReleaseStringUTFChars(name, attr_name);                                             \
                                                                                             \
    if (!throw_exception_if_not_ok(env, status)) {                                           \
      TF_DeleteStatus(status);                                                               \
      return -1;                                                                             \
    }                                                                                        \
    TF_DeleteStatus(status);                                                                 \
                                                                                             \
    return static_cast<jtype>(*value);                                                       \
  }

#define DEFINE_GET_ATTR_LIST(name, jname, jtype, ctype, tf_type)                                  \
  JNIEXPORT jtype##Array JNICALL Java_org_platanios_tensorflow_jni_Op_00024_getAttr##name##List(  \
      JNIEnv* env, jobject object, jlong handle, jstring name) {                                  \
    static_assert(                                                                                \
        sizeof(ctype) >= sizeof(jtype),                                                           \
        "Information loss when converting between Java and C types.");                            \
    TF_Operation *op = require_operation_handle(env, handle);                                     \
    if (op == nullptr) return nullptr;                                                            \
    const char *attr_name = env->GetStringUTFChars(name, nullptr);                                \
    TF_Status *status = TF_NewStatus();                                                           \
    TF_AttrMetadata attr_metadata = TF_OperationGetAttrMetadata(op, attr_name, status);           \
                                                                                                  \
    if (!throw_exception_if_not_ok(env, status)) {                                                \
      TF_DeleteStatus(status);                                                                    \
      return nullptr;                                                                             \
    }                                                                                             \
                                                                                                  \
    if (attr_metadata.type != tf_type || attr_metadata.is_list == 0)                              \
      throw_exception(                                                                            \
          env, tf_invalid_argument_exception,                                                     \
          "Attribute '%s' is not a %s list. It is a '%s', instead.",                              \
          name, attr_name, attrTypeToString(attr_metadata.type, attr_metadata.is_list));          \
                                                                                                  \
    int list_size = static_cast<int>(attr_metadata.list_size);                                    \
    if (list_size < 0) return nullptr;                                                            \
    ctype *attr_values = new ctype[list_size];                                                    \
    TF_OperationGetAttr##name##List(op, attr_name, attr_values, list_size, status);               \
    env->ReleaseStringUTFChars(name, attr_name);                                                  \
                                                                                                  \
    if (!throw_exception_if_not_ok(env, status)) {                                                \
      TF_DeleteStatus(status);                                                                    \
      return nullptr;                                                                             \
    }                                                                                             \
    TF_DeleteStatus(status);                                                                      \
                                                                                                  \
    jtype##Array ret = env->New##jname##Array(list_size);                                         \
    jtype *ret_elements = env->Get##jname##ArrayElements(ret, NULL);                              \
    for (int i = 0; i < list_size; i++) {                                                         \
      ret_elements[i] = static_cast<jtype>(attr_values[i]);                                       \
    }                                                                                             \
    env->Release##jname##ArrayElements(ret, ret_elements, NULL);                                  \
    return ret;                                                                                   \
  }

DEFINE_GET_ATTR_SCALAR(Int, Long, jlong, int64_t, TF_ATTR_INT);
DEFINE_GET_ATTR_LIST(Int, Long, jlong, int64_t, TF_ATTR_INT);
DEFINE_GET_ATTR_SCALAR(Float, Float, jfloat, float, TF_ATTR_FLOAT);
DEFINE_GET_ATTR_LIST(Float, Float, jfloat, float, TF_ATTR_FLOAT);
DEFINE_GET_ATTR_SCALAR(Bool, Boolean, jboolean, unsigned char, TF_ATTR_BOOL);
DEFINE_GET_ATTR_LIST(Bool, Boolean, jboolean, unsigned char, TF_ATTR_BOOL);
DEFINE_GET_ATTR_SCALAR(Type, Int, jint, TF_DataType, TF_ATTR_TYPE);
DEFINE_GET_ATTR_LIST(Type, Int, jint, TF_DataType, TF_ATTR_TYPE);

#undef DEFINE_GET_ATTR_SCALAR
#undef DEFINE_GET_ATTR_LIST

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Op_00024_getAttrTensor(
        JNIEnv* env, jobject object, jlong opHandle, jstring attrName) {
  TF_Operation *op = require_operation_handle(env, opHandle);
  if (op == nullptr) return 0;
  const char *attr_name = env->GetStringUTFChars(attrName, nullptr);
  TF_Status *status = TF_NewStatus();
  TF_AttrMetadata attr_metadata = TF_OperationGetAttrMetadata(op, attr_name, status);

  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);

  if (attr_metadata.type != TF_ATTR_TENSOR || attr_metadata.is_list == 1)
    throw_exception(
        env, tf_invalid_argument_exception, "Attribute '%s' is not a tensor. It is a '%s', instead.",
        attr_name, attrTypeToString(attr_metadata.type, attr_metadata.is_list));
  TF_Tensor* value;
  status = TF_NewStatus();
  TF_OperationGetAttrTensor(op, attr_name, &value, status);
  jlong tensor_handle = reinterpret_cast<jlong>(value);

  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);

  return tensor_handle;
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_getAttrShape(
        JNIEnv* env, jobject object, jlong opHandle, jstring attrName) {
  TF_Operation *op = require_operation_handle(env, opHandle);
  if (op == nullptr) return nullptr;
  const char *attr_name = env->GetStringUTFChars(attrName, nullptr);
  TF_Status *status = TF_NewStatus();
  TF_AttrMetadata attr_metadata = TF_OperationGetAttrMetadata(op, attr_name, status);

  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  if (attr_metadata.total_size < 0) return nullptr;
  if (attr_metadata.type != TF_ATTR_SHAPE || attr_metadata.is_list == 1)
    throw_exception(
        env, tf_invalid_argument_exception, "Attribute '%s' is not a shape. It is a '%s', instead.",
        attr_name, attrTypeToString(attr_metadata.type, attr_metadata.is_list));
  int num_dims = static_cast<int>(attr_metadata.total_size);
  static_assert(sizeof(jlong) == sizeof(int64_t), "Java long is not compatible with the TensorFlow C API");
  // One might have trivially wanted to do:
  // TF_OperationGetAttrShape(op, attr_name, static_cast<int64_t*>(dims), ...)
  // but on some platforms this fails with:
  // static_cast from 'jlong *' (aka 'long *') to 'int64_t *' (aka 'long long *') is not allowed
  // For now, we do the expensive but safe thing of copying.
  std::unique_ptr<int64_t[]> cdims(new int64_t[num_dims]);
  status = TF_NewStatus();
  TF_OperationGetAttrShape(op, attr_name, cdims.get(), num_dims, status);

  if (!throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  jlongArray ret = env->NewLongArray(num_dims);
  jlong *dims = env->GetLongArrayElements(ret, nullptr);
  for (int i = 0; i < num_dims; ++i)
    dims[i] = static_cast<jlong>(cdims[i]);
  env->ReleaseLongArrayElements(ret, dims, 0);
  return ret;
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_allOps(JNIEnv* env, jobject object) {
  TF_Buffer* opListBuffer = TF_GetAllOpList();
  jbyteArray ret = env->NewByteArray(opListBuffer->length);
  jbyte* cpy = env->GetByteArrayElements(ret, nullptr);
  std::memcpy(cpy, opListBuffer->data, opListBuffer->length);
  env->ReleaseByteArrayElements(ret, cpy, 0);
  return ret;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Op_00024_allocate(
    JNIEnv* env, jobject object, jlong graph_handle, jstring type, jstring name) {
  if (graph_handle == 0) {
    throw_exception(env, tf_invalid_argument_exception,
                    "close() has been called on the Graph");
    return 0;
  }
  TF_Graph* graph = reinterpret_cast<TF_Graph*>(graph_handle);
  const char* op_type = env->GetStringUTFChars(type, nullptr);
  const char* op_name = env->GetStringUTFChars(name, nullptr);
  TF_OperationDescription* d = TF_NewOperation(graph, op_type, op_name);
  env->ReleaseStringUTFChars(name, op_name);
  env->ReleaseStringUTFChars(type, op_type);
  static_assert(sizeof(jlong) >= sizeof(TF_OperationDescription*),
                "Cannot represent a C TF_OperationDescription as a Java long");
  return reinterpret_cast<jlong>(d);
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Op_00024_finish(
    JNIEnv* env, jobject object, jlong handle) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  TF_Operation* op = TF_FinishOperation(d, status);
  if (throw_exception_if_not_ok(env, status)) {
    TF_DeleteStatus(status);
    return reinterpret_cast<jlong>(op);
  }
  TF_DeleteStatus(status);
  return 0;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_addInput(
    JNIEnv* env, jobject object, jlong handle, jlong op_handle, jint index) {
  TF_Output out;
  if (!resolveOutput(env, op_handle, index, &out)) return;
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  TF_AddInput(d, out);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_addInputList(
    JNIEnv* env, jobject object, jlong handle, jlongArray op_handles,
    jintArray indices) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const size_t n = static_cast<size_t>(env->GetArrayLength(op_handles));
  if (env->GetArrayLength(indices) != n) {
    throw_exception(env, tf_invalid_argument_exception,
                    "mismatch in number of Operations (%d) and output indices "
                        "(%d) provided",
                    n, env->GetArrayLength(indices));
    return;
  }
  std::unique_ptr<TF_Output[]> o(new TF_Output[n]);
  jlong* oph = env->GetLongArrayElements(op_handles, nullptr);
  jint* idx = env->GetIntArrayElements(indices, nullptr);
  bool ok = true;
  for (int i = 0; i < n && ok; ++i) {
    ok = resolveOutput(env, oph[i], idx[i], &o[i]);
  }
  env->ReleaseIntArrayElements(indices, idx, JNI_ABORT);
  env->ReleaseLongArrayElements(op_handles, oph, JNI_ABORT);
  if (!ok) return;
  TF_AddInputList(d, o.get(), n);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_addControlInput
  (JNIEnv* env, jobject object, jlong handle, jlong inputOpHandle) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  TF_Operation* op = require_operation_handle(env, inputOpHandle);
  if (op == nullptr) return;
  TF_AddControlInput(d, op);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setDevice(
    JNIEnv* env, jobject object, jlong handle, jstring device) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const char* c_device = env->GetStringUTFChars(device, nullptr);
  TF_SetDevice(d, c_device);
  env->ReleaseStringUTFChars(device, c_device);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_colocateWith(
    JNIEnv* env, jobject object, jlong handle, jlong colocationOpHandle) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  TF_Operation* colocationOp = require_operation_handle(env, colocationOpHandle);
  if (colocationOp == nullptr) return;
  TF_ColocateWith(d, colocationOp);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttrString(
    JNIEnv* env, jobject object, jlong handle, jstring name, jbyteArray value) {
  static_assert(
          sizeof(jbyte) == 1, "Require Java byte to be represented as a single byte.");
  TF_OperationDescription *d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  jbyte *c_value = env->GetByteArrayElements(value, nullptr);
  TF_SetAttrString(d, c_name, c_value, static_cast<size_t>(env->GetArrayLength(value)));
  env->ReleaseByteArrayElements(value, c_value, JNI_ABORT);
  env->ReleaseStringUTFChars(name, c_name);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttrStringList(
        JNIEnv* env, jobject object, jlong handle, jstring name, jobjectArray values) {
  static_assert(
          sizeof(jbyte) == 1, "Require Java byte to be represented as a single byte.");
  TF_OperationDescription *d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  int num_strings = env->GetArrayLength(values);
  size_t *lengths = new size_t[num_strings];
  jbyte **strings = new jbyte *[num_strings];
  for (int i = 0; i < num_strings; i++) {
    jbyteArray value = (jbyteArray) env->GetObjectArrayElement(values, i);
    lengths[i] = static_cast<size_t>(env->GetArrayLength(value));
    strings[i] = env->GetByteArrayElements(value, nullptr);
    // TODO: We do not release the array elements because we assume the arrays will be small enough.
  }
  TF_SetAttrStringList(d, c_name, reinterpret_cast<const void *const *>(strings), lengths, num_strings);
  env->ReleaseStringUTFChars(name, c_name);
}

#define DEFINE_SET_ATTR_SCALAR(atype, jtype, ctype)                                          \
  JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttr##atype(          \
      JNIEnv* env, jobject object, jlong handle, jstring name, jtype value) {                \
    static_assert(                                                                           \
        sizeof(ctype) >= sizeof(jtype),                                                      \
        "Information loss when converting between Java and C types");                        \
    TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);             \
    if (d == nullptr) return;                                                                \
    const char* c_name = env->GetStringUTFChars(name, nullptr);                              \
    TF_SetAttr##atype(d, c_name, static_cast<ctype>(value));                                 \
    env->ReleaseStringUTFChars(name, c_name);                                                \
  }

#define DEFINE_SET_ATTR_LIST(atype, jname, jtype, ctype)                           \
  JNIEXPORT void JNICALL                                                           \
      Java_org_platanios_tensorflow_jni_Op_00024_setAttr##atype##List(             \
          JNIEnv* env, jobject object, jlong handle, jstring name,                 \
          jtype##Array value) {                                                    \
    TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);   \
    if (d == nullptr) return;                                                      \
    const char* c_name = env->GetStringUTFChars(name, nullptr);                    \
    /* Make a copy of the array to paper over any differences */                   \
    /* in byte representations of the jtype and ctype         */                   \
    /* For example, jint vs TF_DataType.                      */                   \
    /* If this copy turns out to be a problem in practice     */                   \
    /* can avoid it for many types.                           */                   \
    const int n = env->GetArrayLength(value);                                      \
    std::unique_ptr<ctype[]> c_value(new ctype[n]);                                \
    jtype* elems = env->Get##jname##ArrayElements(value, nullptr);                 \
    for (int i = 0; i < n; ++i) {                                                  \
      c_value[i] = static_cast<ctype>(elems[i]);                                   \
    }                                                                              \
    TF_SetAttr##atype##List(d, c_name, c_value.get(), n);                          \
    env->Release##jname##ArrayElements(value, elems, JNI_ABORT);                   \
    env->ReleaseStringUTFChars(name, c_name);                                      \
  }

#define DEFINE_SET_ATTR(atype, jname, jtype, ctype) \
  DEFINE_SET_ATTR_SCALAR(atype, jtype, ctype)       \
  DEFINE_SET_ATTR_LIST(atype, jname, jtype, ctype)

DEFINE_SET_ATTR(Int, Long, jlong, int64_t);
DEFINE_SET_ATTR(Float, Float, jfloat, float);
DEFINE_SET_ATTR(Bool, Boolean, jboolean, unsigned char);
DEFINE_SET_ATTR(Type, Int, jint, TF_DataType);
#undef DEFINE_SET_ATTR
#undef DEFINE_SET_ATTR_LIST
#undef DEFINE_SET_ATTR_SCALAR

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttrTensor(
    JNIEnv* env, jobject object, jlong handle, jstring name, jlong tensor_handle) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  TF_Tensor* t = requireTensor(env, tensor_handle);
  if (t == nullptr) return;
  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_Status* status = TF_NewStatus();
  TF_SetAttrTensor(d, cname, t, status);
  throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttrTensorList(
    JNIEnv* env, jobject object, jlong handle, jstring name, jlongArray tensor_handles) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const int n = env->GetArrayLength(tensor_handles);
  std::unique_ptr<TF_Tensor* []> tensors(new TF_Tensor*[n]);
  jlong* jhandles = env->GetLongArrayElements(tensor_handles, nullptr);
  bool ok = true;
  for (int i = 0; i < n && ok; ++i) {
    tensors[i] = requireTensor(env, jhandles[i]);
    ok = !env->ExceptionCheck();
  }
  env->ReleaseLongArrayElements(tensor_handles, jhandles, JNI_ABORT);
  if (!ok) return;

  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_Status* status = TF_NewStatus();
  TF_SetAttrTensorList(d, cname, tensors.get(), n, status);
  throw_exception_if_not_ok(env, status);
  TF_DeleteStatus(status);
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttrShape(
    JNIEnv* env, jobject object, jlong handle, jstring name, jlongArray shape,
    jint num_dims) {
  TF_OperationDescription *d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  std::unique_ptr<int64_t[]> cvalue;
  // num_dims and env->GetArrayLength(shape) are assumed to be consistent.
  // i.e., either num_dims < 0 or num_dims == env->GetArrayLength(shape).
  if (num_dims > 0) {
    cvalue.reset(new int64_t[num_dims]);
    jlong *elems = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i) {
      cvalue[i] = static_cast<int64_t>(elems[i]);
    }
    env->ReleaseLongArrayElements(shape, elems, JNI_ABORT);
  }
  const char *cname = env->GetStringUTFChars(name, nullptr);
  TF_SetAttrShape(d, cname, cvalue.get(), static_cast<int>(num_dims));
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttrShapeList(
    JNIEnv* env, jobject object, jlong handle, jstring name, jobjectArray shapes, jintArray num_dims, jint num_shapes) {
  TF_OperationDescription *d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  std::unique_ptr<int[]> c_num_dims;
  std::unique_ptr<int64_t*[]> c_shapes;
  int c_num_shapes = static_cast<int>(num_shapes);
  // num_dims[i] and env->GetArrayLength(shapes[i]) are assumed to be consistent.
  // i.e., either num_dims[i] < 0 or num_dims[i] == env->GetArrayLength(shapes[i]).
  if (c_num_shapes > 0) {
    c_num_dims.reset(new int[c_num_shapes]);
    c_shapes.reset(new int64_t*[c_num_shapes]);
    jint *num_dims_elems = env->GetIntArrayElements(num_dims, nullptr);
    for (int j = 0; j < c_num_shapes; ++j) {
      c_num_dims[j] = static_cast<int>(num_dims_elems[j]);
      if (c_num_dims[j] > -1) {
        c_shapes[j] = new int64_t[c_num_dims[j]];
        jlongArray shapes_elems = (jlongArray) env->GetObjectArrayElement(shapes, j);
        jlong *shape_elems = env->GetLongArrayElements(shapes_elems, nullptr);
        for (int i = 0; i < c_num_dims[j]; ++i) {
          c_shapes[j][i] = static_cast<int64_t>(shape_elems[i]);
        }
        env->ReleaseLongArrayElements(shapes_elems, shape_elems, JNI_ABORT);
      } else {
        c_shapes[j] = new int64_t[0];
      }
    }
    env->ReleaseIntArrayElements(num_dims, num_dims_elems, JNI_ABORT);
  }
  const char *cname = env->GetStringUTFChars(name, nullptr);
  TF_SetAttrShapeList(d, cname, c_shapes.get(), c_num_dims.get(), c_num_shapes);
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttrFuncName(
    JNIEnv* env, jobject object, jlong handle, jstring name, jbyteArray value) {
  static_assert(
          sizeof(jbyte) == 1, "Require Java byte to be represented as a single byte.");
  TF_OperationDescription *d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  jbyte *c_value = env->GetByteArrayElements(value, nullptr);
  TF_SetAttrFuncName(d, c_name, (char *) c_value, static_cast<size_t>(env->GetArrayLength(value)));
  env->ReleaseByteArrayElements(value, c_value, JNI_ABORT);
  env->ReleaseStringUTFChars(name, c_name);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Op_00024_setAttrProto(
    JNIEnv* env, jobject object, jlong handle, jstring name, jbyteArray value) {
  static_assert(sizeof(jbyte) == 1, "We require the Java byte to be represented as a single byte.");
  TF_OperationDescription *d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const char *c_name = env->GetStringUTFChars(name, nullptr);
  jbyte *c_value = env->GetByteArrayElements(value, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_SetAttrValueProto(d, c_name, c_value, static_cast<size_t>(env->GetArrayLength(value)), status.get());
  env->ReleaseByteArrayElements(value, c_value, JNI_ABORT);
  env->ReleaseStringUTFChars(name, c_name);
  CHECK_STATUS(env, status.get(), void());
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_toOpDef(
    JNIEnv* env, jobject object, jlong graph_handle, jstring op_type) {
  TF_Graph *g = require_graph_handle(env, graph_handle);
  if (g == nullptr) return nullptr;
  const char *c_op_type = env->GetStringUTFChars(op_type, nullptr);

  // Call the C API "TF_GraphGetOpDef" function and throw an exception if an error occurs
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  jbyteArray return_array = nullptr;
  TF_Buffer *buf = TF_NewBuffer();
  TF_GraphGetOpDef(g, c_op_type, buf, status.get());
  env->ReleaseStringUTFChars(op_type, c_op_type);
  CHECK_STATUS(env, status.get(), nullptr);
  // sizeof(jsize) is less than sizeof(size_t) on some platforms.
  if (buf->length > std::numeric_limits<jint>::max()) {
    throw_exception(env, tf_invalid_argument_exception,
                    "OpDef is too large to serialize into a Java byte array.");
  } else {
    static_assert(sizeof(jbyte) == 1, "Unexpected size of the Java byte type.");
    jint return_array_length = static_cast<jint>(buf->length);
    return_array = env->NewByteArray(return_array_length);
    env->SetByteArrayRegion(return_array, 0, return_array_length, static_cast<const jbyte *>(buf->data));
  }

  // Clean up and return the byte array
  TF_DeleteBuffer(buf);
  return return_array;
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Op_00024_toNodeDef(
    JNIEnv* env, jobject object, jlong handle) {
  TF_Operation* op = require_operation_handle(env, handle);
  if (op == nullptr) return nullptr;

  // Call the C API "TF_OperationToNodeDef" function and throw an exception if an error occurs
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  jbyteArray return_array = nullptr;
  TF_Buffer *buf = TF_NewBuffer();
  TF_OperationToNodeDef(op, buf, status.get());
  CHECK_STATUS(env, status.get(), nullptr);
  // sizeof(jsize) is less than sizeof(size_t) on some platforms.
  if (buf->length > std::numeric_limits<jint>::max()) {
    throw_exception(env, tf_invalid_argument_exception,
                    "NodeDef is too large to serialize into a Java byte array.");
  } else {
    static_assert(sizeof(jbyte) == 1, "Unexpected size of the Java byte type.");
    jint return_array_length = static_cast<jint>(buf->length);
    return_array = env->NewByteArray(return_array_length);
    env->SetByteArrayRegion(return_array, 0, return_array_length, static_cast<const jbyte *>(buf->data));
  }

  // Clean up and return the byte array
  TF_DeleteBuffer(buf);
  return return_array;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Op_00024_tryEvaluateConstant(
    JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jint output_index) {
  TF_Graph *graph = require_graph_handle(env, graph_handle);
  if (graph == nullptr) return 0;
  TF_Operation* op = require_operation_handle(env, op_handle);
  if (op == nullptr) return 0;
  TF_Output output{op, output_index};

  TF_Tensor* result_tensor;
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  bool evaluated = TF_TryEvaluateConstant(graph, output, &result_tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);
  if (!evaluated) return 0;
  TFE_TensorHandle* result_eager_tensor = TFE_NewTensorHandle(result_tensor, status.get());
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(result_eager_tensor);
}
