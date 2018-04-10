/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERIALIZATION_TRAITS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERIALIZATION_TRAITS_H_

#include "grpc++/impl/codegen/proto_utils.h"
#include "grpc++/support/slice.h"
#include "grpc/grpc.h"

namespace grpc {

namespace tensorflow_helper {

const int kGrpcBufferWriterMaxBufferLength = 8192;

class GrpcBufferWriter final
    : public ::grpc::protobuf::io::ZeroCopyOutputStream {
 public:
  explicit GrpcBufferWriter(grpc_byte_buffer** bp, int block_size)
      : block_size_(block_size), byte_count_(0), have_backup_(false) {
    *bp = grpc_raw_byte_buffer_create(NULL, 0);
    slice_buffer_ = &(*bp)->data.raw.slice_buffer;
  }

  ~GrpcBufferWriter() override {
    if (have_backup_) {
      grpc_slice_unref(backup_slice_);
    }
  }

  bool Next(void** data, int* size) override {
    if (have_backup_) {
      slice_ = backup_slice_;
      have_backup_ = false;
    } else {
      slice_ = grpc_slice_malloc(block_size_);
    }
    *data = GRPC_SLICE_START_PTR(slice_);
    // On win x64, int is only 32bit
    GPR_CODEGEN_ASSERT(GRPC_SLICE_LENGTH(slice_) <= INT_MAX);
    byte_count_ += * size = (int)GRPC_SLICE_LENGTH(slice_);
    grpc_slice_buffer_add(slice_buffer_, slice_);
    return true;
  }

  void BackUp(int count) override {
    grpc_slice_buffer_pop(slice_buffer_);
    if (count == block_size_) {
      backup_slice_ = slice_;
    } else {
      backup_slice_ =
          grpc_slice_split_tail(&slice_, GRPC_SLICE_LENGTH(slice_) - count);
      grpc_slice_buffer_add(slice_buffer_, slice_);
    }
    // It's dangerous to keep an inlined grpc_slice as the backup slice, since
    // on a following Next() call, a reference will be returned to this slice
    // via GRPC_SLICE_START_PTR, which will not be an address held by
    // slice_buffer_.
    have_backup_ = backup_slice_.refcount != NULL;
    byte_count_ -= count;
  }

  grpc::protobuf::int64 ByteCount() const override { return byte_count_; }

 private:
  const int block_size_;
  int64_t byte_count_;
  grpc_slice_buffer* slice_buffer_;
  bool have_backup_;
  grpc_slice backup_slice_;
  grpc_slice slice_;
};

class GrpcBufferReader final
    : public ::grpc::protobuf::io::ZeroCopyInputStream {
 public:
  explicit GrpcBufferReader(grpc_byte_buffer* buffer)
      : byte_count_(0), backup_count_(0) {
    (void)grpc_byte_buffer_reader_init(&reader_, buffer);
  }
  ~GrpcBufferReader() override { grpc_byte_buffer_reader_destroy(&reader_); }

  bool Next(const void** data, int* size) override {
    if (backup_count_ > 0) {
      *data = GRPC_SLICE_START_PTR(slice_) + GRPC_SLICE_LENGTH(slice_) -
              backup_count_;
      GPR_CODEGEN_ASSERT(backup_count_ <= INT_MAX);
      *size = (int)backup_count_;
      backup_count_ = 0;
      return true;
    }
    if (!grpc_byte_buffer_reader_next(&reader_, &slice_)) {
      return false;
    }
    grpc_slice_unref(slice_);
    *data = GRPC_SLICE_START_PTR(slice_);
    // On win x64, int is only 32bit
    GPR_CODEGEN_ASSERT(GRPC_SLICE_LENGTH(slice_) <= INT_MAX);
    byte_count_ += * size = (int)GRPC_SLICE_LENGTH(slice_);
    return true;
  }

  void BackUp(int count) override { backup_count_ = count; }

  bool Skip(int count) override {
    const void* data;
    int size;
    while (Next(&data, &size)) {
      if (size >= count) {
        BackUp(size - count);
        return true;
      }
      // size < count;
      count -= size;
    }
    // error or we have too large count;
    return false;
  }

  grpc::protobuf::int64 ByteCount() const override {
    return byte_count_ - backup_count_;
  }

 private:
  int64_t byte_count_;
  int64_t backup_count_;
  grpc_byte_buffer_reader reader_;
  grpc_slice slice_;
};

}  // namespace tensorflow_helper

// Defines specialized serialization/deserialization routines that
// default to allowing a 2GB max message size.
//
// To instantiate this template for a particular type `T`, use
// `TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(T)`, as defined below.
template <typename T>
class UnlimitedSizeProtoSerializationTraits {
 public:
  static Status Serialize(const T& msg, grpc_byte_buffer** bp,
                          bool* own_buffer) {
    *own_buffer = true;
    int byte_size = msg.ByteSize();
    if (byte_size < 0) {
      return Status(StatusCode::INTERNAL, "Message length was negative");
    } else if (byte_size <=
               tensorflow_helper::kGrpcBufferWriterMaxBufferLength) {
      grpc_slice slice = grpc_slice_malloc(byte_size);
      GPR_CODEGEN_ASSERT(
          GRPC_SLICE_END_PTR(slice) ==
          msg.SerializeWithCachedSizesToArray(GRPC_SLICE_START_PTR(slice)));
      *bp = grpc_raw_byte_buffer_create(&slice, 1);
      grpc_slice_unref(slice);
      return Status::OK;
    } else {
      tensorflow_helper::GrpcBufferWriter writer(
          bp, tensorflow_helper::kGrpcBufferWriterMaxBufferLength);
      return msg.SerializeToZeroCopyStream(&writer)
                 ? Status::OK
                 : Status(StatusCode::INTERNAL, "Failed to serialize message");
    }
  }

  static Status Deserialize(grpc_byte_buffer* buffer, T* msg,
                            int max_message_size = INT_MAX) {
    if (buffer == nullptr) {
      return Status(StatusCode::INTERNAL, "No payload");
    }
    Status result = Status::OK;
    {
      tensorflow_helper::GrpcBufferReader reader(buffer);
      ::grpc::protobuf::io::CodedInputStream decoder(&reader);
      if (max_message_size == 0) {
        // NOTE(mrry): Override maximum message size to 2GB.
        decoder.SetTotalBytesLimit(INT_MAX, INT_MAX);
      } else {
        decoder.SetTotalBytesLimit(max_message_size, max_message_size);
      }
      if (!msg->ParseFromCodedStream(&decoder)) {
        result = Status(StatusCode::INTERNAL, msg->InitializationErrorString());
      }
      if (!decoder.ConsumedEntireMessage()) {
        result = Status(StatusCode::INTERNAL, "Did not read entire message");
      }
    }
    grpc_byte_buffer_destroy(buffer);
    return result;
  }
};

}  // namespace grpc

// For the given protobuf message type `MessageType`, specializes the
// gRPC serialization and deserialization such that the default
// maximum message size is 2GB.
#define TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(MessageType)             \
  namespace grpc {                                                    \
  template <>                                                         \
  class SerializationTraits<MessageType>                              \
      : public UnlimitedSizeProtoSerializationTraits<MessageType> {}; \
  }  // namespace grpc

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERIALIZATION_TRAITS_H_
