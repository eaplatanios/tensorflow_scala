#include "include/operation_jni.h"

#include <memory>

#include "include/c_api.h"
#include "include/exception_jni.h"

namespace {
template <class T>
T* requireHandleImpl(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(T*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(
        env, kNullPointerException,
        "close() has been called on the Graph this Operation was a part of");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

TF_Operation* requireOperationHandle(JNIEnv* env, jlong handle) {
  return requireHandleImpl<TF_Operation>(env, handle);
}

TF_Graph* requireGraphHandle(JNIEnv* env, jlong handle) {
  return requireHandleImpl<TF_Graph>(env, handle);
}

TF_OperationDescription* requireOperationDescriptionHandle(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "Operation has already been built");
    return 0;
  }
  return reinterpret_cast<TF_OperationDescription*>(handle);
}

bool resolveOutput(JNIEnv* env, jlong op_handle, jint index, TF_Output* out) {
  if (op_handle == 0) {
    throwException(env, kIllegalStateException,
                   "close() was called on the Graph");
    return false;
  }
  out->oper = reinterpret_cast<TF_Operation*>(op_handle);
  out->index = static_cast<int>(index);
  return true;
}

TF_Tensor* requireTensor(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "close() has been called on the Tensor");
    return nullptr;
  }
  return reinterpret_cast<TF_Tensor*>(handle);
}
}  // namespace

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_name(JNIEnv* env,
                                                                             jobject object,
                                                                             jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationName(op));
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_opType(JNIEnv* env,
                                                                                 jobject object,
                                                                                 jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationOpType(op));
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_device(JNIEnv* env,
                                                                                 jobject object,
                                                                                 jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationDevice(op));
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_numInputs(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumInputs(op);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_numControlInputs(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumControlInputs(op);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_numOutputs(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumOutputs(op);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_numControlOutputs(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumControlOutputs(op);
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_numConsumers(JNIEnv* env,
                                                                                jobject object,
                                                                                jlong handle,
                                                                                jint output_index) {
  TF_Operation* op = requireOperationHandle(env, handle);
  if (op == nullptr) return 0;
  TF_Output output{op, output_index};
  return TF_OperationOutputNumConsumers(output);
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_controlInputs(JNIEnv* env,
                                                                                             jobject object,
                                                                                             jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
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

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_controlOutputs(JNIEnv* env,
                                                                                             jobject object,
                                                                                             jlong handle) {
  TF_Operation* op = requireOperationHandle(env, handle);
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

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_inputDataType(
    JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jint input_index) {
  TF_Graph* graph = requireGraphHandle(env, graph_handle);
  if (graph == nullptr) return 0;
  TF_Operation* op = requireOperationHandle(env, op_handle);
  if (op == nullptr) return 0;

  int num_inputs = TF_OperationNumInputs(op);
  if (input_index < 0 || input_index >= num_inputs) {
    throwException(
        env, kIndexOutOfBoundsException,
        "invalid input index (%d) for an operation that has %d inputs",
        input_index, num_inputs);
    return 0;
  }

  return static_cast<jint>(TF_OperationInputType(TF_Input{op, input_index}));
}

JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_outputDataType(
    JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jint output_index) {
  TF_Graph* graph = requireGraphHandle(env, graph_handle);
  if (graph == nullptr) return 0;
  TF_Operation* op = requireOperationHandle(env, op_handle);
  if (op == nullptr) return 0;

  int num_outputs = TF_OperationNumOutputs(op);
  if (output_index < 0 || output_index >= num_outputs) {
    throwException(
        env, kIndexOutOfBoundsException,
        "invalid output index (%d) for an operation that has %d outputs",
        output_index, num_outputs);
    return 0;
  }

  return static_cast<jint>(TF_OperationOutputType(TF_Output{op, output_index}));
}

JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_shape(
    JNIEnv* env, jobject object, jlong graph_handle, jlong op_handle, jint output_index) {
  TF_Graph* graph = requireGraphHandle(env, graph_handle);
  if (graph == nullptr) return nullptr;
  TF_Operation* op = requireOperationHandle(env, op_handle);
  if (op == nullptr) return nullptr;

  int num_outputs = TF_OperationNumOutputs(op);
  if (output_index < 0 || output_index >= num_outputs) {
    throwException(
        env, kIndexOutOfBoundsException,
        "invalid output index (%d) for an operation that has %d outputs",
        output_index, num_outputs);
    return nullptr;
  }

  TF_Output output{op, output_index};
  TF_Status* status = TF_NewStatus();
  jsize num_dims = TF_GraphGetTensorNumDims(graph, output, status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  if (num_dims < 0) return nullptr;
  static_assert(sizeof(jlong) == sizeof(int64_t),
                "Java long is not compatible with the TensorFlow C API");
  // One might have trivially wanted to do:
  // TF_GraphGetTensorShape(graph, output, static_cast<int64_t*>(dims), ...)
  // but on some platforms this fails with:
  // static_cast from 'jlong *' (aka 'long *') to 'int64_t *' (aka 'long long
  // *') is not allowed
  // For now, do the expensive but safe thing of copying.
  std::unique_ptr<int64_t[]> cdims(new int64_t[num_dims]);
  TF_GraphGetTensorShape(graph, output, cdims.get(), static_cast<int>(num_dims),
                         status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  jlongArray ret = env->NewLongArray(num_dims);
  jlong* dims = env->GetLongArrayElements(ret, nullptr);
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = static_cast<jlong>(cdims[i]);
  }
  env->ReleaseLongArrayElements(ret, dims, 0);
  return ret;
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_allOps(JNIEnv* env,
                                                                                      jobject object) {
  TF_Buffer* opListBuffer = TF_GetAllOpList();
  jbyteArray ret = env->NewByteArray(opListBuffer->length);
  jbyte* cpy = env->GetByteArrayElements(ret, nullptr);
  memcpy(cpy, opListBuffer->data, opListBuffer->length);
  env->ReleaseByteArrayElements(ret, cpy, 0);
  return ret;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_allocate(
    JNIEnv* env, jobject object, jlong graph_handle, jstring type, jstring name) {
  if (graph_handle == 0) {
    throwException(env, kIllegalStateException,
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

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_finish(
    JNIEnv* env, jobject object, jlong handle) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  TF_Operation* op = TF_FinishOperation(d, status);
  if (throwExceptionIfNotOK(env, status)) {
    return reinterpret_cast<jlong>(op);
  }
  return 0;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_addInput(
    JNIEnv* env, jobject object, jlong handle, jlong op_handle, jint index) {
  TF_Output out;
  if (!resolveOutput(env, op_handle, index, &out)) return;
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  TF_AddInput(d, out);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_addInputList(
    JNIEnv* env, jobject object, jlong handle, jlongArray op_handles,
    jintArray indices) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const size_t n = static_cast<size_t>(env->GetArrayLength(op_handles));
  if (env->GetArrayLength(indices) != n) {
    throwException(env, kIllegalArgumentException,
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

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_addControlInput
  (JNIEnv* env, jobject object, jlong handle, jlong inputOpHandle) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  TF_Operation* op = requireOperationHandle(env, inputOpHandle);
  if (op == nullptr) return;
  TF_AddControlInput(d, op);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_setDevice(
    JNIEnv* env, jobject object, jlong handle, jstring device) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const char* cdevice = env->GetStringUTFChars(device, nullptr);
  TF_SetDevice(d, cdevice);
  env->ReleaseStringUTFChars(device, cdevice);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_setAttrString(
    JNIEnv* env, jobject object, jlong handle, jstring name, jbyteArray value) {
  static_assert(sizeof(jbyte) == 1,
                "Require Java byte to be represented as a single byte");
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  const char* cname = env->GetStringUTFChars(name, nullptr);
  jbyte* cvalue = env->GetByteArrayElements(value, nullptr);
  TF_SetAttrString(d, cname, cvalue, env->GetArrayLength(value));
  env->ReleaseByteArrayElements(value, cvalue, JNI_ABORT);
  env->ReleaseStringUTFChars(name, cname);
}

#define DEFINE_SET_ATTR_SCALAR(name, jtype, ctype)                                           \
  JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_setAttr##name(        \
      JNIEnv* env, jobject object, jlong handle, jstring name, jtype value) {                \
    static_assert(                                                                           \
        sizeof(ctype) >= sizeof(jtype),                                                      \
        "Information loss when converting between Java and C types");                        \
    TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);             \
    if (d == nullptr) return;                                                                \
    const char* cname = env->GetStringUTFChars(name, nullptr);                               \
    TF_SetAttr##name(d, cname, static_cast<ctype>(value));                                   \
    env->ReleaseStringUTFChars(name, cname);                                                 \
  }

#define DEFINE_SET_ATTR_LIST(name, jname, jtype, ctype)                            \
  JNIEXPORT void JNICALL                                                           \
      Java_org_platanios_tensorflow_jni_Operation_00024_setAttr##name##List(           \
          JNIEnv* env, jobject object, jlong handle, jstring name,                 \
          jtype##Array value) {                                                    \
    TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);   \
    if (d == nullptr) return;                                                      \
    const char* cname = env->GetStringUTFChars(name, nullptr);                     \
    /* Make a copy of the array to paper over any differences */                   \
    /* in byte representations of the jtype and ctype         */                   \
    /* For example, jint vs TF_DataType.                      */                   \
    /* If this copy turns out to be a problem in practice     */                   \
    /* can avoid it for many types.                           */                   \
    const int n = env->GetArrayLength(value);                                      \
    std::unique_ptr<ctype[]> cvalue(new ctype[n]);                                 \
    jtype* elems = env->Get##jname##ArrayElements(value, nullptr);                 \
    for (int i = 0; i < n; ++i) {                                                  \
      cvalue[i] = static_cast<ctype>(elems[i]);                                    \
    }                                                                              \
    TF_SetAttr##name##List(d, cname, cvalue.get(), n);                             \
    env->Release##jname##ArrayElements(value, elems, JNI_ABORT);                   \
    env->ReleaseStringUTFChars(name, cname);                                       \
  }

#define DEFINE_SET_ATTR(name, jname, jtype, ctype) \
  DEFINE_SET_ATTR_SCALAR(name, jtype, ctype)       \
  DEFINE_SET_ATTR_LIST(name, jname, jtype, ctype)

DEFINE_SET_ATTR(Int, Long, jlong, int64_t);
DEFINE_SET_ATTR(Float, Float, jfloat, float);
DEFINE_SET_ATTR(Bool, Boolean, jboolean, unsigned char);
DEFINE_SET_ATTR(Type, Int, jint, TF_DataType);
#undef DEFINE_SET_ATTR
#undef DEFINE_SET_ATTR_LIST
#undef DEFINE_SET_ATTR_SCALAR

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_setAttrTensor(
    JNIEnv* env, jobject object, jlong handle, jstring name,
    jlong tensor_handle) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  TF_Tensor* t = requireTensor(env, tensor_handle);
  if (t == nullptr) return;
  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_Status* status = TF_NewStatus();
  TF_SetAttrTensor(d, cname, t, status);
  throwExceptionIfNotOK(env, status);
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_setAttrTensorList(
    JNIEnv* env, jobject object, jlong handle, jstring name,
    jlongArray tensor_handles) {
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
  throwExceptionIfNotOK(env, status);
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Operation_00024_setAttrShape(
    JNIEnv* env, jobject object, jlong handle, jstring name, jlongArray shape,
    jint num_dims) {
  TF_OperationDescription* d = requireOperationDescriptionHandle(env, handle);
  if (d == nullptr) return;
  std::unique_ptr<int64_t[]> cvalue;
  // num_dims and env->GetArrayLength(shape) are assumed to be consistent.
  // i.e., either num_dims < 0 or num_dims == env->GetArrayLength(shape).
  if (num_dims > 0) {
    cvalue.reset(new int64_t[num_dims]);
    jlong* elems = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i) {
      cvalue[i] = static_cast<int64_t>(elems[i]);
    }
    env->ReleaseLongArrayElements(shape, elems, JNI_ABORT);
  }
  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_SetAttrShape(d, cname, cvalue.get(), static_cast<int>(num_dims));
  env->ReleaseStringUTFChars(name, cname);
}
