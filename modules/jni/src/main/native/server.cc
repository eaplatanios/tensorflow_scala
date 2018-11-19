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
#include "server.h"
#include "utilities.h"

#include <memory>

#include "tensorflow/c/c_api.h"

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Server_00024_newServer(
    JNIEnv* env, jobject object, jbyteArray server_def_proto) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  jbyte* c_server_def_proto = env->GetByteArrayElements(server_def_proto, nullptr);
  TF_Server* server = TF_NewServer(
    c_server_def_proto, static_cast<size_t>(env->GetArrayLength(server_def_proto)), status.get());
  if (server_def_proto != nullptr)
    env->ReleaseByteArrayElements(server_def_proto, c_server_def_proto, JNI_ABORT);
  CHECK_STATUS(env, status.get(), 0);
  return reinterpret_cast<jlong>(server);
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Server_00024_target(
    JNIEnv* env, jobject object, jlong server_handle) {
  REQUIRE_HANDLE(server, TF_Server, server_handle, nullptr);
  return env->NewStringUTF(TF_ServerTarget(server));
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Server_00024_startServer(
    JNIEnv* env, jobject object, jlong server_handle) {
  REQUIRE_HANDLE(server, TF_Server, server_handle, void());
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_ServerStart(server, status.get());
  CHECK_STATUS(env, status.get(), void());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Server_00024_stopServer(
    JNIEnv* env, jobject object, jlong server_handle) {
  REQUIRE_HANDLE(server, TF_Server, server_handle, void());
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_ServerStop(server, status.get());
  CHECK_STATUS(env, status.get(), void());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Server_00024_joinServer(
    JNIEnv* env, jobject object, jlong server_handle) {
  REQUIRE_HANDLE(server, TF_Server, server_handle, void());
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  TF_ServerJoin(server, status.get());
  CHECK_STATUS(env, status.get(), void());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Server_00024_deleteServer(
    JNIEnv* env, jobject object, jlong server_handle) {
  REQUIRE_HANDLE(server, TF_Server, server_handle, void());
  TF_DeleteServer(server);
}
