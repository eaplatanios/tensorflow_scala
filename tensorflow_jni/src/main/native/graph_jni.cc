#include "include/graph_jni.h"

#include <limits>
#include <vector>
#include "include/c_api.h"
#include "include/exception_jni.h"

namespace {
  template<class T>
  T *require_handle_helper(JNIEnv *env, jlong handle, const char *closed_error_msg) {
    static_assert(sizeof(jlong) >= sizeof(T *),
                  "Cannot package C object pointers as a Java long");
    if (handle == 0) {
      throwException(env, kNullPointerException, closed_error_msg);
      return nullptr;
    }
    return reinterpret_cast<T *>(handle);
  }

  TF_Operation *require_operation_handle(JNIEnv *env, jlong handle) {
    return require_handle_helper<TF_Operation>(env, handle, "close() has been called on this operation.");
  }

  TF_Graph *require_graph_handle(JNIEnv *env, jlong handle) {
    return require_handle_helper<TF_Graph>(env, handle, "close() has been called on this graph.");
  }

  std::unique_ptr<TF_Output[]> to_tf_output_array(
      JNIEnv *env, jobjectArray java_array, jfieldID output_op_handle_field_id,
      jfieldID output_op_index_field_id) {
    if (java_array == nullptr) return nullptr;
    int array_length = env->GetArrayLength(java_array);
    std::unique_ptr<TF_Output[]> array(new TF_Output[array_length]);
    for (int i = 0; i < array_length; ++i) {
      jobject object = env->GetObjectArrayElement(java_array, i);
      jlong op_handle = env->GetLongField(object, output_op_handle_field_id);
      jint output_index = env->GetIntField(object, output_op_index_field_id);
      TF_Operation *op = require_operation_handle(env, op_handle);
      if (op == nullptr) return nullptr;
      array[i] = {op, output_index};
    }
    return array;
  }
}  // namespace

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_allocate(JNIEnv* env, jobject object) {
  return reinterpret_cast<jlong>(TF_NewGraph());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_delete(
    JNIEnv* env, jobject object, jlong graph_handle) {
  if (graph_handle == 0) return;
  TF_DeleteGraph(reinterpret_cast<TF_Graph *>(graph_handle));
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_findOp(
    JNIEnv* env, jobject object, jlong graph_handle, jstring op_name) {
  TF_Graph *g = require_graph_handle(env, graph_handle);
  if (g == nullptr) return 0;
  const char *op_name_c_string = env->GetStringUTFChars(op_name, nullptr);
  TF_Operation *op = TF_GraphOperationByName(g, op_name_c_string);
  env->ReleaseStringUTFChars(op_name, op_name_c_string);
  return reinterpret_cast<jlong>(op);
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_ops(
    JNIEnv* env, jobject object, jlong graph_handle) {
  TF_Graph *g = require_graph_handle(env, graph_handle);
  if (g == nullptr) return 0;

  // Call the C API "TF_GraphNextOperation" repeatedly to obtain all ops in the graph
  std::vector<TF_Operation *> ops;
  size_t pos = 0;
  TF_Operation *op;
  while ((op = TF_GraphNextOperation(g, &pos)) != nullptr)
    ops.push_back(op);

  // Construct the return array
  jlongArray return_array = env->NewLongArray(static_cast<jsize>(ops.size()));
  jlong *op_handles_array = env->GetLongArrayElements(return_array, nullptr);
  for (int i = 0; i < ops.size(); ++i)
    op_handles_array[i] = reinterpret_cast<jlong>(ops[i]);
  env->ReleaseLongArrayElements(return_array, op_handles_array, 0);
  return return_array;
}

JNIEXPORT jobjectArray JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_addGradients(
        JNIEnv* env, jobject object, jlong graph_handle, jobjectArray y_array, jobjectArray x_array,
        jobjectArray dx_array) {
  TF_Graph *g = require_graph_handle(env, graph_handle);
  if (g == nullptr) return nullptr;

  // Convert the inputs to their C API equivalent data structures
  jclass output_class = env->FindClass("org/platanios/tensorflow/jni/OpOutput");
  jfieldID output_op_handle_field_id = env->GetFieldID(output_class, "opHandle", "J");
  jfieldID output_op_index_field_id = env->GetFieldID(output_class, "outputIndex", "I");

  int ny = env->GetArrayLength(y_array);
  std::unique_ptr<TF_Output[]> y = to_tf_output_array(
      env, y_array, output_op_handle_field_id, output_op_index_field_id);
  if (y == nullptr) return nullptr;
  int nx = env->GetArrayLength(x_array);
  std::unique_ptr<TF_Output[]> x = to_tf_output_array(
      env, x_array, output_op_handle_field_id, output_op_index_field_id);
  if (x == nullptr) return nullptr;
  std::unique_ptr<TF_Output[]> dx = to_tf_output_array(
      env, dx_array, output_op_handle_field_id, output_op_index_field_id);
  std::unique_ptr<TF_Output[]> dy(new TF_Output[nx]);

  // Call the C API "TF_AddGradients" function and throw an exception if an error occurs
  TF_Status *status = TF_NewStatus();
  TF_AddGradients(g, y.get(), ny, x.get(), nx, dx.get(), status, dy.get());
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  // Construct the return gradients array
  jmethodID output_class_constructor = env->GetStaticMethodID(
      output_class, "apply", "(JI)Lorg/platanios/tensorflow/jni/OpOutput;");
  jobjectArray gradients_array = env->NewObjectArray(nx, output_class, NULL);
  for (int i = 0; i < nx; ++i) {
    env->SetObjectArrayElement(
        gradients_array, i, env->CallStaticObjectMethod(
            output_class, output_class_constructor, reinterpret_cast<jlong>(dy[i].oper), dy[i].index));
  }
  return gradients_array;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_importGraphDef(
    JNIEnv* env, jobject object, jlong graph_handle, jbyteArray graph_def, jstring name_prefix) {
  TF_Graph *g = require_graph_handle(env, graph_handle);
  if (g == nullptr) return;

  TF_ImportGraphDefOptions *options = TF_NewImportGraphDefOptions();
  jboolean is_copy;
  const char *name_prefix_c_string = env->GetStringUTFChars(name_prefix, &is_copy);
  TF_ImportGraphDefOptionsSetPrefix(options, name_prefix_c_string);
  env->ReleaseStringUTFChars(name_prefix, name_prefix_c_string);
  static_assert(sizeof(jbyte) == 1, "unexpected size of the jbyte type");
  jbyte *bytes = env->GetByteArrayElements(graph_def, &is_copy);
  TF_Buffer *buffer = TF_NewBufferFromString(bytes, static_cast<size_t>(env->GetArrayLength(graph_def)));

  // Call the C API "TF_GraphImportGraphDef" function and throw an exception if an error occurs
  TF_Status *status = TF_NewStatus();
  TF_GraphImportGraphDef(g, buffer, options, status);
  throwExceptionIfNotOK(env, status);

  // Continue cleaning up resources even if an exception was thrown
  TF_DeleteStatus(status);
  TF_DeleteBuffer(buffer);
  env->ReleaseByteArrayElements(graph_def, bytes, JNI_ABORT);
  TF_DeleteImportGraphDefOptions(options);
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Graph_00024_toGraphDef(
    JNIEnv* env, jobject object, jlong graph_handle) {
  TF_Graph *g = require_graph_handle(env, graph_handle);
  if (g == nullptr) return nullptr;

  // Call the C API "TF_GraphToGraphDef" function and throw an exception if an error occurs
  jbyteArray return_array = nullptr;
  TF_Buffer *buf = TF_NewBuffer();
  TF_Status *status = TF_NewStatus();
  TF_GraphToGraphDef(g, buf, status);
  if (throwExceptionIfNotOK(env, status)) {
    // sizeof(jsize) is less than sizeof(size_t) on some platforms.
    if (buf->length > std::numeric_limits<jint>::max()) {
      throwException(env, kIndexOutOfBoundsException, "GraphDef is too large to serialize into a Java byte array.");
    } else {
      static_assert(sizeof(jbyte) == 1, "Unexpected size of the Java byte type.");
      jint return_array_length = static_cast<jint>(buf->length);
      return_array = env->NewByteArray(return_array_length);
      env->SetByteArrayRegion(return_array, 0, return_array_length, static_cast<const jbyte *>(buf->data));
    }
  }

  // Clean up and return the byte array
  TF_DeleteStatus(status);
  TF_DeleteBuffer(buf);
  return return_array;
}
