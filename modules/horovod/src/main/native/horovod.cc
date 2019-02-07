/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

#include "horovod.h"

#include "common/operations.h"

JNIEXPORT void JNICALL Java_org_platanios_tensorflow_horovod_Horovod_00024_init(JNIEnv* env, jobject object) {
  horovod::common::horovod_init();
}

JNIEXPORT int JNICALL Java_org_platanios_tensorflow_horovod_Horovod_00024_rank(JNIEnv* env, jobject object) {
  return horovod::common::horovod_rank();
}

JNIEXPORT int JNICALL Java_org_platanios_tensorflow_horovod_Horovod_00024_localRank(JNIEnv* env, jobject object) {
  return horovod::common::horovod_local_rank();
}

JNIEXPORT int JNICALL Java_org_platanios_tensorflow_horovod_Horovod_00024_size(JNIEnv* env, jobject object) {
  return horovod::common::horovod_size();
}

JNIEXPORT int JNICALL Java_org_platanios_tensorflow_horovod_Horovod_00024_localSize(JNIEnv* env, jobject object) {
  return horovod::common::horovod_local_size();
}

JNIEXPORT int JNICALL Java_org_platanios_tensorflow_horovod_Horovod_00024_mpiThreadsSupported(JNIEnv* env, jobject object) {
  return horovod::common::horovod_mpi_threads_supported();
}
