// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/versions.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fframework_2fversions_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fframework_2fversions_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3009000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3009002 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2fframework_2fversions_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2fframework_2fversions_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto;
namespace tensorflow {
class VersionDef;
class VersionDefDefaultTypeInternal;
extern VersionDefDefaultTypeInternal _VersionDef_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::VersionDef* Arena::CreateMaybeMessage<::tensorflow::VersionDef>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class VersionDef :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.VersionDef) */ {
 public:
  VersionDef();
  virtual ~VersionDef();

  VersionDef(const VersionDef& from);
  VersionDef(VersionDef&& from) noexcept
    : VersionDef() {
    *this = ::std::move(from);
  }

  inline VersionDef& operator=(const VersionDef& from) {
    CopyFrom(from);
    return *this;
  }
  inline VersionDef& operator=(VersionDef&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArena() const final {
    return GetArenaNoVirtual();
  }
  inline void* GetMaybeArenaPointer() const final {
    return MaybeArenaPtr();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const VersionDef& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const VersionDef* internal_default_instance() {
    return reinterpret_cast<const VersionDef*>(
               &_VersionDef_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(VersionDef& a, VersionDef& b) {
    a.Swap(&b);
  }
  inline void Swap(VersionDef* other) {
    if (other == this) return;
    if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(VersionDef* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline VersionDef* New() const final {
    return CreateMaybeMessage<VersionDef>(nullptr);
  }

  VersionDef* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<VersionDef>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const VersionDef& from);
  void MergeFrom(const VersionDef& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  #else
  bool MergePartialFromCodedStream(
      ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(VersionDef* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.VersionDef";
  }
  protected:
  explicit VersionDef(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto);
    return ::descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kBadConsumersFieldNumber = 3,
    kProducerFieldNumber = 1,
    kMinConsumerFieldNumber = 2,
  };
  // repeated int32 bad_consumers = 3;
  int bad_consumers_size() const;
  void clear_bad_consumers();
  ::PROTOBUF_NAMESPACE_ID::int32 bad_consumers(int index) const;
  void set_bad_consumers(int index, ::PROTOBUF_NAMESPACE_ID::int32 value);
  void add_bad_consumers(::PROTOBUF_NAMESPACE_ID::int32 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
      bad_consumers() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
      mutable_bad_consumers();

  // int32 producer = 1;
  void clear_producer();
  ::PROTOBUF_NAMESPACE_ID::int32 producer() const;
  void set_producer(::PROTOBUF_NAMESPACE_ID::int32 value);

  // int32 min_consumer = 2;
  void clear_min_consumer();
  ::PROTOBUF_NAMESPACE_ID::int32 min_consumer() const;
  void set_min_consumer(::PROTOBUF_NAMESPACE_ID::int32 value);

  // @@protoc_insertion_point(class_scope:tensorflow.VersionDef)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 > bad_consumers_;
  mutable std::atomic<int> _bad_consumers_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 producer_;
  ::PROTOBUF_NAMESPACE_ID::int32 min_consumer_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fframework_2fversions_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// VersionDef

// int32 producer = 1;
inline void VersionDef::clear_producer() {
  producer_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 VersionDef::producer() const {
  // @@protoc_insertion_point(field_get:tensorflow.VersionDef.producer)
  return producer_;
}
inline void VersionDef::set_producer(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  producer_ = value;
  // @@protoc_insertion_point(field_set:tensorflow.VersionDef.producer)
}

// int32 min_consumer = 2;
inline void VersionDef::clear_min_consumer() {
  min_consumer_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 VersionDef::min_consumer() const {
  // @@protoc_insertion_point(field_get:tensorflow.VersionDef.min_consumer)
  return min_consumer_;
}
inline void VersionDef::set_min_consumer(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  min_consumer_ = value;
  // @@protoc_insertion_point(field_set:tensorflow.VersionDef.min_consumer)
}

// repeated int32 bad_consumers = 3;
inline int VersionDef::bad_consumers_size() const {
  return bad_consumers_.size();
}
inline void VersionDef::clear_bad_consumers() {
  bad_consumers_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int32 VersionDef::bad_consumers(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.VersionDef.bad_consumers)
  return bad_consumers_.Get(index);
}
inline void VersionDef::set_bad_consumers(int index, ::PROTOBUF_NAMESPACE_ID::int32 value) {
  bad_consumers_.Set(index, value);
  // @@protoc_insertion_point(field_set:tensorflow.VersionDef.bad_consumers)
}
inline void VersionDef::add_bad_consumers(::PROTOBUF_NAMESPACE_ID::int32 value) {
  bad_consumers_.Add(value);
  // @@protoc_insertion_point(field_add:tensorflow.VersionDef.bad_consumers)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
VersionDef::bad_consumers() const {
  // @@protoc_insertion_point(field_list:tensorflow.VersionDef.bad_consumers)
  return bad_consumers_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
VersionDef::mutable_bad_consumers() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.VersionDef.bad_consumers)
  return &bad_consumers_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fframework_2fversions_2eproto