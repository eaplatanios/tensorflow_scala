// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: google/bigtable/admin/v2/common.proto

#ifndef PROTOBUF_INCLUDED_google_2fbigtable_2fadmin_2fv2_2fcommon_2eproto
#define PROTOBUF_INCLUDED_google_2fbigtable_2fadmin_2fv2_2fcommon_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/generated_enum_reflection.h>
#include "google/api/annotations.pb.h"
#include <google/protobuf/timestamp.pb.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_google_2fbigtable_2fadmin_2fv2_2fcommon_2eproto 

namespace protobuf_google_2fbigtable_2fadmin_2fv2_2fcommon_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[1];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_google_2fbigtable_2fadmin_2fv2_2fcommon_2eproto
namespace google {
namespace bigtable {
namespace admin {
namespace v2 {
}  // namespace v2
}  // namespace admin
}  // namespace bigtable
}  // namespace google
namespace google {
namespace bigtable {
namespace admin {
namespace v2 {

enum StorageType {
  STORAGE_TYPE_UNSPECIFIED = 0,
  SSD = 1,
  HDD = 2,
  StorageType_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  StorageType_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool StorageType_IsValid(int value);
const StorageType StorageType_MIN = STORAGE_TYPE_UNSPECIFIED;
const StorageType StorageType_MAX = HDD;
const int StorageType_ARRAYSIZE = StorageType_MAX + 1;

const ::google::protobuf::EnumDescriptor* StorageType_descriptor();
inline const ::std::string& StorageType_Name(StorageType value) {
  return ::google::protobuf::internal::NameOfEnum(
    StorageType_descriptor(), value);
}
inline bool StorageType_Parse(
    const ::std::string& name, StorageType* value) {
  return ::google::protobuf::internal::ParseNamedEnum<StorageType>(
    StorageType_descriptor(), name, value);
}
// ===================================================================


// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace v2
}  // namespace admin
}  // namespace bigtable
}  // namespace google

namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::google::bigtable::admin::v2::StorageType> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::google::bigtable::admin::v2::StorageType>() {
  return ::google::bigtable::admin::v2::StorageType_descriptor();
}

}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_google_2fbigtable_2fadmin_2fv2_2fcommon_2eproto
