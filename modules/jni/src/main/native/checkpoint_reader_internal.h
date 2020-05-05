/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef TENSORFLOW_C_CHECKPOINT_READER_H_
#define TENSORFLOW_C_CHECKPOINT_READER_H_

#include <memory>
#include <string>

#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/io/table.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

class Status;

string MetaFilename(StringPiece prefix);

// The empty string, hence always the first key in the metadata table.  Its
// corresponding value is a BundleHeaderProto.
extern const char* const kHeaderEntryKey;

class BundleReader {
 public:
  BundleReader(Env* const env, StringPiece prefix);
  ~BundleReader();

  // Is ok() iff the reader construction is successful (completed the read of
  // the metadata).
  Status status() const  { return status_; }

  // Queries whether the bundle contains an entry keyed by "key".  Calls Seek()
  // internally, so this call invalidates the reader's current position.
  // REQUIRES: status().ok()
  bool Contains(StringPiece key);

  // Looks up the dtype and the shape of the tensor keyed by "key".
  // REQUIRES: status().ok()
  Status LookupDtypeAndShape(StringPiece key, DataType* dtype, TensorShape* shape);

  // Looks up the tensor keyed by "key".  If "key" refers to a partitioned
  // tensor, attempts to look up the full contents using all stored slices.
  //
  // Caller must make sure "val" has the same shape and dtype as the
  // corresponding contents, so that its buffer can be filled without needing
  // extra allocation.  These can be queried via "LookupDtypeAndShape()".
  //
  // On error, "val" may contain nonsense data.  Returns a NotFound error if
  // tensor keyed by "key" does not exist in this bundle.
  //
  // Validates the stored crc32c checksum against the restored bytes.
  // REQUIRES: status().ok()
  Status Lookup(StringPiece key, Tensor* val);

  // Seeks to the first position in the bundle whose key is no less than "key".
  // REQUIRES: status().ok()
  void Seek(StringPiece key) { return iter_->Seek(key); }

  // Moves to the next position in the bundle.
  // REQUIRES: status().ok()
  void Next() const { iter_->Next(); }

  // Returns true iff the reader is positioned to a key/val pair.
  // REQUIRES: status().ok()
  bool Valid() const { return iter_->Valid(); }

  // Returns the key at the current position.
  // REQUIRES: status().ok() && Valid()
  StringPiece key() const { return iter_->key(); }

  // Returns the raw value at the current position.
  // REQUIRES: status().ok() && Valid()
  StringPiece value() const { return iter_->value(); }

  string DebugString();

 private:
  Status status_;
  table::Iterator* iter_;
};

namespace checkpoint {

class TensorSliceReader;

// A wrapper around BundleReader (for V2 checkpoints) and
// checkpoint::TensorSliceReader (for V1), that is more easily SWIG wrapped for
// other languages.
//
// The class currently only interacts with single-slice (i.e., non-partitioned)
// variables.
class CheckpointReader {
 public:
  CheckpointReader(const string& filename, TF_Status* status);

  bool HasTensor(const string& name) const;
  const string DebugString() const;

  // Returns a map from variable names to their shapes.  Slices of a partitioned
  // tensor are combined into a single entry.
  const TensorSliceReader::VarToShapeMap& GetVariableToShapeMap() const;

  // Returns a map from variable names to their data types.  Slices of a
  // partitioned tensor are combined into a single entry.
  const TensorSliceReader::VarToDataTypeMap& GetVariableToDataTypeMap() const;

  // Attempts to look up the tensor named "name" and stores the found result in
  // "out_tensor".
  void GetTensor(const string& name,
                 std::unique_ptr<tensorflow::Tensor>* out_tensor,
                 TF_Status* out_status) const;

 private:
  // Uses "v2_reader_" to build "var name -> shape" and "var name -> data type"
  // maps; both owned by caller.
  // REQUIRES: "v2_reader_ != nullptr && v2_reader_.status().ok()".
  std::pair<std::unique_ptr<TensorSliceReader::VarToShapeMap>,
            std::unique_ptr<TensorSliceReader::VarToDataTypeMap> >
  BuildV2VarMaps();

  // Invariant: exactly one of "reader_" and "v2_reader_" is non-null.
  std::unique_ptr<TensorSliceReader> reader_;
  std::unique_ptr<BundleReader> v2_reader_;

  std::unique_ptr<TensorSliceReader::VarToShapeMap> var_to_shape_map_;
  std::unique_ptr<TensorSliceReader::VarToDataTypeMap> var_to_data_type_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(CheckpointReader);
};

}  // namespace checkpoint
}  // namespace tensorflow

#endif  // TENSORFLOW_C_CHECKPOINT_READER_H_
