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

#include "file_io.h"
#include "utilities.h"

#include <string.h>

#include "tensorflow/c/status_helper.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/file_system.h"

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_fileExists(
    JNIEnv* env, jobject object, jstring filename) {
  const char* c_filename = env->GetStringUTFChars(filename, nullptr);
  tensorflow::Status s = tensorflow::Env::Default()->FileExists(std::string(c_filename));
  env->ReleaseStringUTFChars(filename, c_filename);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), void());
  }
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_deleteFile(
    JNIEnv* env, jobject object, jstring filename) {
  const char* c_filename = env->GetStringUTFChars(filename, nullptr);
  tensorflow::Status s = tensorflow::Env::Default()->DeleteFile(std::string(c_filename));
  env->ReleaseStringUTFChars(filename, c_filename);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), void());
  }
}

JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_readFileToString(
    JNIEnv* env, jobject object, jstring filename) {
  const char* c_filename = env->GetStringUTFChars(filename, nullptr);
  std::string file_content;
  tensorflow::Status s = tensorflow::ReadFileToString(
    tensorflow::Env::Default(), std::string(c_filename), &file_content);
  env->ReleaseStringUTFChars(filename, c_filename);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), nullptr);
  }
  return env->NewStringUTF(file_content.c_str());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_writeStringToFile(
    JNIEnv* env, jobject object, jstring filename, jstring content) {
  const char* c_filename = env->GetStringUTFChars(filename, nullptr);
  const char* c_content = env->GetStringUTFChars(content, nullptr);
  tensorflow::Status s = tensorflow::WriteStringToFile(
    tensorflow::Env::Default(), std::string(c_filename), std::string(c_content));
  env->ReleaseStringUTFChars(content, c_content);
  env->ReleaseStringUTFChars(filename, c_filename);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), void());
  }
}

JNIEXPORT jobjectArray JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_getChildren(
    JNIEnv* env, jobject object, jstring filename) {
  const char* c_filename = env->GetStringUTFChars(filename, nullptr);
  std::vector<std::string> children;
  tensorflow::Status s = tensorflow::Env::Default()->GetChildren(std::string(c_filename), &children);
  env->ReleaseStringUTFChars(filename, c_filename);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), nullptr);
  }
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray children_array = env->NewObjectArray(children.size(), string_class, NULL);
  for (int i = 0; i < children.size(); ++i) {
    env->SetObjectArrayElement(children_array, i, env->NewStringUTF(children[i].c_str()));
  }
  return children_array;
}

JNIEXPORT jobjectArray JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_getMatchingFiles(
    JNIEnv* env, jobject object, jstring filename) {
  const char* c_filename = env->GetStringUTFChars(filename, nullptr);
  std::vector<std::string> children;
  tensorflow::Status s = tensorflow::Env::Default()->GetMatchingPaths(std::string(c_filename), &children);
  env->ReleaseStringUTFChars(filename, c_filename);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), nullptr);
  }
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray children_array = env->NewObjectArray(children.size(), string_class, NULL);
  for (int i = 0; i < children.size(); ++i) {
    env->SetObjectArrayElement(children_array, i, env->NewStringUTF(children[i].c_str()));
  }
  return children_array;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_mkDir(
    JNIEnv* env, jobject object, jstring dirname) {
  const char* c_dirname = env->GetStringUTFChars(dirname, nullptr);
  tensorflow::Status s = tensorflow::Env::Default()->CreateDir(std::string(c_dirname));
  env->ReleaseStringUTFChars(dirname, c_dirname);
  if (!s.ok() && s.code() != tensorflow::error::ALREADY_EXISTS) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), void());
  }
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_mkDirs(
    JNIEnv* env, jobject object, jstring dirname) {
  const char* c_dirname = env->GetStringUTFChars(dirname, nullptr);
  tensorflow::Status s = tensorflow::Env::Default()->RecursivelyCreateDir(std::string(c_dirname));
  env->ReleaseStringUTFChars(dirname, c_dirname);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), void());
  }
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_copyFile(
    JNIEnv* env, jobject object, jstring old_path, jstring new_path, jboolean overwrite) {
  const char* c_old_path = env->GetStringUTFChars(old_path, nullptr);
  const char* c_new_path = env->GetStringUTFChars(new_path, nullptr);
  std::string cpp_old_path = std::string(c_old_path);
  std::string cpp_new_path = std::string(c_new_path);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  // If overwrite is false and the `new_path` file exists then we have a problem.
  if (!overwrite && tensorflow::Env::Default()->FileExists(cpp_new_path).ok()) {
    TF_SetStatus(status.get(), TF_ALREADY_EXISTS, "File already exists.");
    env->ReleaseStringUTFChars(new_path, c_new_path);
    env->ReleaseStringUTFChars(old_path, c_old_path);
    CHECK_STATUS(env, status.get(), void());
  }
  std::string file_content;
  tensorflow::Status s = tensorflow::ReadFileToString(tensorflow::Env::Default(), cpp_old_path, &file_content);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status.get(), s);
    env->ReleaseStringUTFChars(new_path, c_new_path);
    env->ReleaseStringUTFChars(old_path, c_old_path);
    CHECK_STATUS(env, status.get(), void());
  }
  s = tensorflow::WriteStringToFile(tensorflow::Env::Default(), cpp_new_path, file_content);
  env->ReleaseStringUTFChars(new_path, c_new_path);
  env->ReleaseStringUTFChars(old_path, c_old_path);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), void());
  }
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_renameFile(
    JNIEnv* env, jobject object, jstring old_path, jstring new_path, jboolean overwrite) {
  const char* c_old_path = env->GetStringUTFChars(old_path, nullptr);
  const char* c_new_path = env->GetStringUTFChars(new_path, nullptr);
  std::string cpp_old_path = std::string(c_old_path);
  std::string cpp_new_path = std::string(c_new_path);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  // If overwrite is false and the `new_path` file exists then we have a problem.
  if (!overwrite && tensorflow::Env::Default()->FileExists(cpp_new_path).ok()) {
    TF_SetStatus(status.get(), TF_ALREADY_EXISTS, "File already exists.");
    env->ReleaseStringUTFChars(new_path, c_new_path);
    env->ReleaseStringUTFChars(old_path, c_old_path);
    CHECK_STATUS(env, status.get(), void());
  }
  std::string file_content;
  tensorflow::Status s = tensorflow::Env::Default()->RenameFile(cpp_old_path, cpp_new_path);
  env->ReleaseStringUTFChars(new_path, c_new_path);
  env->ReleaseStringUTFChars(old_path, c_old_path);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), void());
  }
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_deleteRecursively(
    JNIEnv* env, jobject object, jstring dirname) {
  const char* c_dirname = env->GetStringUTFChars(dirname, nullptr);
  tensorflow::int64 undeleted_files, undeleted_dirs;
  tensorflow::Status s = tensorflow::Env::Default()->DeleteRecursively(
    std::string(c_dirname), &undeleted_files, &undeleted_dirs);
  env->ReleaseStringUTFChars(dirname, c_dirname);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), void());
  }
  if (undeleted_files > 0 || undeleted_dirs > 0) {
    TF_SetStatus(status.get(), TF_PERMISSION_DENIED, "Could not fully delete the specified directory.");
    CHECK_STATUS(env, status.get(), void());
  }
}

JNIEXPORT jboolean JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_isDirectory(
    JNIEnv* env, jobject object, jstring dirname) {
  const char* c_dirname = env->GetStringUTFChars(dirname, nullptr);
  tensorflow::Status s = tensorflow::Env::Default()->IsDirectory(std::string(c_dirname));
  env->ReleaseStringUTFChars(dirname, c_dirname);
  if (s.ok()) {
    return JNI_TRUE;
  }
  // A FAILED_PRECONDITION status response means that path exists but is not a directory.
  if (s.code() != tensorflow::error::FAILED_PRECONDITION) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), JNI_FALSE);
  }
  return JNI_FALSE;
}

JNIEXPORT jobject JNICALL Java_org_platanios_tensorflow_jni_FileIO_00024_statistics(
    JNIEnv* env, jobject object, jstring path) {
  const char* c_path = env->GetStringUTFChars(path, nullptr);
  tensorflow::FileStatistics* statistics = new tensorflow::FileStatistics();
  tensorflow::Status s = tensorflow::Env::Default()->Stat(std::string(c_path), statistics);
  env->ReleaseStringUTFChars(path, c_path);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), nullptr);
  }
  jclass file_statistics_class = env->FindClass("org/platanios/tensorflow/jni/FileStatistics");
  jmethodID file_statistics_constructor = env->GetStaticMethodID(
      file_statistics_class, "apply", "(JJB)Lorg/platanios/tensorflow/jni/FileStatistics;");
  return env->CallStaticObjectMethod(
    file_statistics_class, file_statistics_constructor,
    static_cast<jlong>(statistics->length), static_cast<jlong>(statistics->mtime_nsec),
    static_cast<jboolean>(statistics->is_directory));
}
