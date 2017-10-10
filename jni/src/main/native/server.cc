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

#include "exception.h"
#include "server.h"
#include "utilities.h"

#include "tensorflow/c/status_helper.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Server_00024_newServer(
    JNIEnv* env, jobject object, jbyteArray server_def_proto) {
  tensorflow::ServerDef server_def;
  jbyte* c_server_def_proto = env->GetByteArrayElements(server_def_proto, nullptr);
  tensorflow::Status status;
  if (!server_def.ParseFromArray(c_server_def_proto, static_cast<size_t>(env->GetArrayLength(server_def_proto))))
    status = tensorflow::errors::InvalidArgument("Unparsable ServerDef proto.");
  std::unique_ptr<tensorflow::ServerInterface>* server;
  if (status.ok())
    status = tensorflow::NewServer(server_def, server);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> c_status(TF_NewStatus(), TF_DeleteStatus);
  if (server_def_proto != nullptr)
    env->ReleaseByteArrayElements(server_def_proto, c_server_def_proto, JNI_ABORT);
  tensorflow::Set_TF_Status_from_Status(c_status.get(), status);
  CHECK_STATUS(env, c_status.get(), 0);
  return reinterpret_cast<jlong>(server->release());
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Server_00024_target(
    JNIEnv* env, jobject object, jlong server_handle) {
  typedef tensorflow::ServerInterface ServerInterface;
  REQUIRE_HANDLE(server, ServerInterface, server_handle, void());
  return env->NewStringUTF(server->target().c_str());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Server_00024_startServer(
    JNIEnv* env, jobject object, jlong server_handle) {
  typedef tensorflow::ServerInterface ServerInterface;
  REQUIRE_HANDLE(server, ServerInterface, server_handle, void());
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  tensorflow::Set_TF_Status_from_Status(status.get(), server->Start());
  throw_exception_if_not_ok(env, status.get());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Server_00024_stopServer(
    JNIEnv* env, jobject object, jlong server_handle) {
  typedef tensorflow::ServerInterface ServerInterface;
  REQUIRE_HANDLE(server, ServerInterface, server_handle, void());
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  tensorflow::Set_TF_Status_from_Status(status.get(), server->Stop());
  throw_exception_if_not_ok(env, status.get());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Server_00024_joinServer(
    JNIEnv* env, jobject object, jlong server_handle) {
  typedef tensorflow::ServerInterface ServerInterface;
  REQUIRE_HANDLE(server, ServerInterface, server_handle, void());
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  tensorflow::Set_TF_Status_from_Status(status.get(), server->Join());
  throw_exception_if_not_ok(env, status.get());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Server_00024_deleteServer(
    JNIEnv* env, jobject object, jlong server_handle) {
  typedef tensorflow::ServerInterface ServerInterface;
  REQUIRE_HANDLE(server, ServerInterface, server_handle, void());
  delete server;
}
