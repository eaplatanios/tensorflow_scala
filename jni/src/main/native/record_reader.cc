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

#include "record_reader.h"
#include "utilities.h"

#include <string.h>

#include "tensorflow/c/record_reader.h"
#include "tensorflow/c/status_helper.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_newRandomAccessFile(
    JNIEnv* env, jobject object, jstring filename) {
  const char* c_filename = env->GetStringUTFChars(filename, nullptr);
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  tensorflow::Status s = tensorflow::Env::Default()->NewRandomAccessFile(std::string(c_filename), &file);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), 0);
  }
  env->ReleaseStringUTFChars(filename, c_filename);
  return reinterpret_cast<jlong>(file.release());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_deleteRandomAccessFile(
    JNIEnv* env, jobject object, jlong file_handle) {
  REQUIRE_HANDLE(file, tensorflow::RandomAccessFile, file_handle, void());
  delete file;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_newRecordReader(
    JNIEnv* env, jobject object, jlong file_handle, jstring compression_type) {
  REQUIRE_HANDLE(file, tensorflow::RandomAccessFile, file_handle, 0);
  const char* c_compression_type = env->GetStringUTFChars(compression_type, nullptr);
  tensorflow::io::RecordReaderOptions options =
    tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions(std::string(c_compression_type));
  auto* reader = new tensorflow::io::RecordReader(file, options);
  env->ReleaseStringUTFChars(compression_type, c_compression_type);
  return reinterpret_cast<jlong>(reader);
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_recordReaderRead(
    JNIEnv* env, jobject object, jlong reader_handle, jlong offset) {
  REQUIRE_HANDLE(reader, tensorflow::io::RecordReader, reader_handle, nullptr);
  tensorflow::uint64 c_offset = static_cast<tensorflow::uint64>(offset);
  std::string record;
  tensorflow::Status s = reader->ReadRecord(&c_offset, &record);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), 0);
  }
  jbyteArray record_array = env->NewByteArray(static_cast<jsize>(record.size()));
  jbyte* record_array_elements = env->GetByteArrayElements(record_array, nullptr);
  memcpy(record_array_elements, record.data(), record.size());
  env->ReleaseByteArrayElements(record_array, record_array_elements, JNI_COMMIT);
  return record_array;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_deleteRecordReader(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::io::RecordReader, reader_handle, void());
  delete reader;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_newSequentialRecordReader(
    JNIEnv* env, jobject object, jlong file_handle, jstring compression_type) {
  REQUIRE_HANDLE(file, tensorflow::RandomAccessFile, file_handle, 0);
  const char* c_compression_type = env->GetStringUTFChars(compression_type, nullptr);
  tensorflow::io::RecordReaderOptions options =
    tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions(std::string(c_compression_type));
  auto* reader = new tensorflow::io::SequentialRecordReader(file, options);
  env->ReleaseStringUTFChars(compression_type, c_compression_type);
  return reinterpret_cast<jlong>(reader);
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_sequentialRecordReaderReadNext(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::io::SequentialRecordReader, reader_handle, nullptr);
  std::string record;
  tensorflow::Status s = reader->ReadRecord(&record);
  if (!s.ok()) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
    Set_TF_Status_from_Status(status.get(), s);
    CHECK_STATUS(env, status.get(), 0);
  }
  jbyteArray record_array = env->NewByteArray(static_cast<jsize>(record.size()));
  jbyte* record_array_elements = env->GetByteArrayElements(record_array, nullptr);
  memcpy(record_array_elements, record.data(), record.size());
  env->ReleaseByteArrayElements(record_array, record_array_elements, JNI_COMMIT);
  return record_array;
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_deleteSequentialRecordReader(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::io::SequentialRecordReader, reader_handle, void());
  delete reader;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_newRecordReaderWrapper(
    JNIEnv* env, jobject object, jstring filename, jstring compression_type, jlong start_offset) {
  const char* c_filename = env->GetStringUTFChars(filename, nullptr);
  const char* c_compression_type = env->GetStringUTFChars(compression_type, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  auto* reader = tensorflow::io::RecordReaderWrapper::New(
    std::string(c_filename), static_cast<tensorflow::uint64>(start_offset), std::string(c_compression_type),
    status.get());;
  CHECK_STATUS(env, status.get(), 0);
  env->ReleaseStringUTFChars(compression_type, c_compression_type);
  env->ReleaseStringUTFChars(filename, c_filename);
  return reinterpret_cast<jlong>(reader);
}

JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_recordReaderWrapperReadNext(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::io::RecordReaderWrapper, reader_handle, nullptr);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(TF_NewStatus(), TF_DeleteStatus);
  reader->GetNext(status.get());
  CHECK_STATUS(env, status.get(), nullptr);
  std::string record = reader->record();
  jbyteArray record_array = env->NewByteArray(static_cast<jsize>(record.size()));
  jbyte* record_array_elements = env->GetByteArrayElements(record_array, nullptr);
  memcpy(record_array_elements, record.data(), record.size());
  env->ReleaseByteArrayElements(record_array, record_array_elements, JNI_COMMIT);
  return record_array;
}

JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_recordReaderWrapperOffset(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::io::RecordReaderWrapper, reader_handle, -1);
  return static_cast<jlong>(reader->offset());
}

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_RecordReader_00024_deleteRecordReaderWrapper(
    JNIEnv* env, jobject object, jlong reader_handle) {
  REQUIRE_HANDLE(reader, tensorflow::io::RecordReaderWrapper, reader_handle, void());
  delete reader;
}
