// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_model_proto_IMPL_H_
#define tensorflow_core_framework_model_proto_IMPL_H_

#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/framework/model.pb_text.h"
#include "tensorflow/core/lib/strings/proto_text_util.h"
#include "tensorflow/core/lib/strings/scanner.h"

namespace tensorflow {
namespace data {
namespace model {
namespace proto {

namespace internal {

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::data::model::proto::Model& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::data::model::proto::Model* msg);

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::data::model::proto::Node& msg);
bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::data::model::proto::Node* msg);

}  // namespace internal

}  // namespace proto
}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // tensorflow_core_framework_model_proto_IMPL_H_
