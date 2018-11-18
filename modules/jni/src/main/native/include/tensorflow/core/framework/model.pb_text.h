// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_model_proto_H_
#define tensorflow_core_framework_model_proto_H_

#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace model {
namespace proto {

// Message-text conversion for tensorflow.data.model.proto.Model
string ProtoDebugString(
    const ::tensorflow::data::model::proto::Model& msg);
string ProtoShortDebugString(
    const ::tensorflow::data::model::proto::Model& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::data::model::proto::Model* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.data.model.proto.Node
string ProtoDebugString(
    const ::tensorflow::data::model::proto::Node& msg);
string ProtoShortDebugString(
    const ::tensorflow::data::model::proto::Node& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::data::model::proto::Node* msg)
        TF_MUST_USE_RESULT;

}  // namespace proto
}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // tensorflow_core_framework_model_proto_H_
