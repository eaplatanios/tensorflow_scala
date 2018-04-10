/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_FRAMEWORK_DATASET_H_
#define TENSORFLOW_CORE_FRAMEWORK_DATASET_H_

#include <memory>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/dataset_stateful_op_whitelist.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tracing.h"

// Polymorphic datasets should support all primitive TensorFlow
// types. Use this macro to expand `m(T)` once for each primitive type
// `T`, e.g. to build a `switch` statement.
#define TF_CALL_DATASET_TYPES(m) TF_CALL_ALL_TYPES(m) TF_CALL_QUANTIZED_TYPES(m)

namespace tensorflow {

// Interface for reading values from a key-value store.
// Used for restoring iterator state.
class IteratorStateReader {
 public:
  virtual Status ReadScalar(StringPiece key, int64* val) = 0;
  virtual Status ReadScalar(StringPiece key, string* val) = 0;
  virtual Status ReadTensor(StringPiece key, Tensor* val) = 0;
  virtual bool Contains(StringPiece key) = 0;

  virtual ~IteratorStateReader() {}
};

// Interface for writing values to a key-value store.
// Used for saving iterator state.
class IteratorStateWriter {
 public:
  virtual Status WriteScalar(StringPiece key, const int64 val) = 0;
  virtual Status WriteScalar(StringPiece key, const string& val) = 0;
  virtual Status WriteTensor(StringPiece key, const Tensor& val) = 0;

  virtual ~IteratorStateWriter() {}
};

// Forward declarations to avoid introducing a dependency on headers in
// "tensorflow/core/graph/...".
class GraphDefBuilder;
class GraphDatasetBase;
class Node;

// Wrapper around GraphDefBuilder. Used to serialize Dataset graph.
class GraphDefBuilderWrapper {
 public:
  explicit GraphDefBuilderWrapper(GraphDefBuilder* b) : b_(b) {}

  // Adds a Const node with scalar value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  template <typename T>
  Status AddScalar(const T& val, Node** output) {
    Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
    val_t.scalar<T>()() = val;
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddScalar: Failed to build Const op.");
    }
    return Status::OK();
  }

  // Adds a Const node with vector value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  // TODO(shivaniagrawal): Consider changing to gtl::ArraySlice?
  template <typename T>
  Status AddVector(const std::vector<T>& val, Node** output) {
    Tensor val_t = Tensor(DataTypeToEnum<T>::v(),
                          TensorShape({static_cast<int64>(val.size())}));
    for (int i = 0; i < val.size(); i++) {
      val_t.flat<T>()(i) = val[i];
    }
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddVector: Failed to build Const op.");
    }
    return Status::OK();
  }

  // Adds a Const node with Tensor value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  Status AddTensor(const Tensor& val, Node** output) {
    AddTensorInternal(val, output);
    if (*output == nullptr) {
      return errors::Internal("AddTensor: Failed to build Const op.");
    }
    return Status::OK();
  }

  Status AddDataset(const GraphDatasetBase* dataset,
                    const std::vector<Node*>& inputs, Node** output) {
    return AddDataset(dataset, inputs, {}, output);
  }

  // Adds a node corresponding to the `DatasetType` to the Graph.
  // Return value of `DatasetType::op_name()` is used as the op type for the
  // node.
  // Values for the output_types and output_shapes node attributes are also
  // written if those attributes are defined in the OpDef.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  Status AddDataset(const GraphDatasetBase* dataset,
                    const std::vector<Node*>& inputs,
                    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
                    Node** output) {
    std::vector<std::pair<size_t, Node*>> enumerated_inputs(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
      enumerated_inputs[i] = std::make_pair(i, inputs[i]);
    }
    return AddDataset(dataset, enumerated_inputs, {}, attrs, output);
  }

  Status AddDataset(
      const GraphDatasetBase* dataset,
      const std::vector<std::pair<size_t, Node*>>& inputs,
      const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
      const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
      Node** output);

  // Adds a user-defined function with name `function_name` to the graph and
  // recursively adds all functions it references. If a function with a matching
  // name has already been added, returns with OK status. If a user-defined with
  // name `function_name` is not found in the FunctionLibraryDefinition, returns
  // an InvalidArgumentError. If the function with name `function_name` or any
  // of its dependent functions are stateful, returns an InvalidArgument error.
  Status AddFunction(OpKernelContext* ctx, const string& function_name);

  template <typename T>
  void BuildAttrValue(const T& value, AttrValue* attr) {
    SetAttrValue(value, attr);
  }

 private:
  void AddTensorInternal(const Tensor& val, Node** output);

  Status EnsureFunctionIsStateless(OpKernelContext* ctx,
                                   const string& function_name) const {
    const FunctionLibraryDefinition* lib_def =
        ctx->function_library()->GetFunctionLibraryDefinition();
    const FunctionDef* function_def = lib_def->Find(function_name);
    if (!function_def) {
      return errors::InvalidArgument("Unable to find FunctionDef for ",
                                     function_name, " in registry.");
    }
    for (const NodeDef& node_def : function_def->node_def()) {
      const OpDef* op_def;
      TF_RETURN_IF_ERROR(lib_def->LookUpOpDef(node_def.op(), &op_def));
      // TODO(b/65524810): Hack to allow functions to capture Dataset op
      // nodes needed for FlatMap. Currently, source datasets nodes have been
      // marked stateful to avoid constant folding since we do not have a
      // good way of serializing them.
      if (IsOpWhitelisted(op_def)) {
        continue;
      }
      if (op_def->is_stateful()) {
        return errors::InvalidArgument(
            "Op[name: ", node_def.name(), ", type: ", node_def.op(), "] ",
            "in function ", function_name, " is stateful. ",
            "Saving stateful functions is not supported yet.");
      }
    }
    return Status::OK();
  }

  // Returns whether an op has been whitelisted for use inside map_fns.
  // Uses a heuristic to whitelist source dataset ops which have been
  // marked stateful due to b/65524810.
  // Also looks up the `op_def->name` in the global
  // `WhitelistedStatefulOpRegistry`.
  bool IsOpWhitelisted(const OpDef* op_def) const {
    return (str_util::EndsWith(op_def->name(), "Dataset") &&
            op_def->output_arg_size() == 1 &&
            op_def->output_arg(0).type() == DT_VARIANT) ||
           dataset::WhitelistedStatefulOpRegistry::Global()->Contains(
               op_def->name());
  }

  bool HasAttr(const string& op_type_name, const string& attr_name) const;

  bool HasAttr(const OpDef* op_def, const string& attr_name) const {
    for (auto attr : op_def->attr()) {
      if (attr.name() == attr_name) {
        return true;
      }
    }
    return false;
  }

  Status AddAttrFunctions(const AttrValue& attr_value, OpKernelContext* ctx) {
    if (attr_value.has_func()) {
      TF_RETURN_IF_ERROR(AddFunction(ctx, attr_value.func().name()));
    } else if (attr_value.has_list()) {
      for (const NameAttrList& name_attr_list : attr_value.list().func()) {
        TF_RETURN_IF_ERROR(AddFunction(ctx, name_attr_list.name()));
      }
    }
    return Status::OK();
  }

  GraphDefBuilder* b_;
};

class StatsAggregator;

// A cut-down version of OpKernelContext for running computations in
// iterators. Note that we cannot simply use OpKernelContext here
// because we might run computation in an iterator whose lifetime is
// not nested within the lifetime of a single OpKernelContext
// (e.g. asynchronous prefetching).
//
// TODO(mrry): We will probably need to support more of
// OpKernelContext here. For example, should allocation be handled by
// the IteratorContext?
// TODO(mrry): We're making some daring assumptions about the lifetime
// of the runner passed in here. A runner will be deleted when the original
// step ends, but all existing runners only close over session-lifetime (or
// longer-lived) state, so we can make a copy of the function. There's nothing
// in the definition of the API from which we took the runner to guarantee that
// what we are doing is safe. We should formalize the properties here.
class IteratorContext {
 public:
  struct Params {
    // Interface to operating system functionality.
    Env* env;

    // Function call support.
    std::function<void(std::function<void()>)> runner = nullptr;

    // A function that returns the current `StatsAggregator` instance to be
    // used when recording statistics about the iterator.
    //
    // NOTE(mrry): This is somewhat awkward, because (i) the `StatsAggregator`
    // is a property of the `IteratorResource` (which this class does not know
    // about), and (ii) it can change after the `IteratorContext` has been
    // created. Better suggestions are welcome!
    std::function<std::shared_ptr<StatsAggregator>()> stats_aggregator_getter =
        nullptr;

    // The FunctionLibraryRuntime object to be used to make function calls.
    FunctionLibraryRuntime* lib = nullptr;
    std::shared_ptr<const FunctionLibraryDefinition> function_library = nullptr;

    // The Allocator to be used to allocate the output of an iterator.
    std::function<Allocator*(AllocatorAttributes)> allocator_getter = nullptr;
  };

  explicit IteratorContext(Params params) : params_(std::move(params)) {}

  Env* env() const { return params_.env; }

  std::function<void(std::function<void()>)>* runner() {
    return &params_.runner;
  }

  std::shared_ptr<StatsAggregator> stats_aggregator() {
    if (params_.stats_aggregator_getter) {
      return params_.stats_aggregator_getter();
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<const FunctionLibraryDefinition> function_library() {
    return params_.function_library;
  }

  FunctionLibraryRuntime* lib() { return params_.lib; }

  void set_lib(FunctionLibraryRuntime* lib) { params_.lib = lib; }

  Allocator* allocator(AllocatorAttributes attrs) {
    return params_.allocator_getter(attrs);
  }

  std::function<Allocator*(AllocatorAttributes)> allocator_getter() {
    return params_.allocator_getter;
  }

  std::function<std::shared_ptr<StatsAggregator>()> stats_aggregator_getter() {
    return params_.stats_aggregator_getter;
  }

 private:
  Params params_;
};

// Represents the current position in a range of outputs, where the
// range of outputs is typically represented by an `DatasetBase`,
// defined below.
class IteratorBase {
 public:
  virtual ~IteratorBase() {}

  // Gets the next output from the range that this iterator is traversing.
  //
  // If at least one output remains in this iterator's range, that
  // output will be stored in `*out_tensors` and `false` will be
  // stored in `*end_of_sequence`.
  //
  // If no more outputs remain in this iterator's range, `true` will
  // be stored in `*end_of_sequence`, and the content of
  // `*out_tensors` will be undefined.
  //
  // This method is thread-safe.
  //
  // TODO(mrry): Define `GetNextAsync()` or `GetNextManyAsync()`, and
  // potentially remove this method.
  virtual Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) = 0;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // iterator.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this iterator.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;

  // Saves the state of this iterator.
  virtual Status Save(OpKernelContext* ctx, IteratorStateWriter* writer) {
    return SaveInternal(writer);
  }

  // Restores the state of this iterator.
  virtual Status Restore(IteratorContext* ctx, IteratorStateReader* reader) {
    return RestoreInternal(ctx, reader);
  }

 protected:
  // This is needed so that sub-classes of IteratorBase can call
  // `SaveInternal` on their parent iterators, e.g., in
  // `RepeatDataasetOp::Dataset`.
  Status SaveParent(IteratorStateWriter* writer,
                    const std::unique_ptr<IteratorBase>& parent) {
    return parent->SaveInternal(writer);
  }

  // This is needed so that sub-classes of IteratorBase can call
  // `RestoreInternal` on their parent iterators, e.g., in
  // `RepeatDataasetOp::Dataset`.
  Status RestoreParent(IteratorContext* ctx, IteratorStateReader* reader,
                       const std::unique_ptr<IteratorBase>& parent) {
    return parent->RestoreInternal(ctx, reader);
  }

  // Saves the state of this iterator recursively.
  virtual Status SaveInternal(IteratorStateWriter* writer) {
    return errors::Unimplemented("SaveInternal");
  }

  // Restores the state of this iterator recursively.
  virtual Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) {
    return errors::Unimplemented("RestoreInternal");
  }
};

// Represents a (potentially infinite) range of outputs, where each
// output is a tuple of tensors.
class DatasetBase : public core::RefCounted {
 public:
  // Returns a new iterator for iterating over the range of elements in
  // this dataset.
  //
  // This method may be called multiple times on the same instance,
  // and the resulting iterators will have distinct state. Each
  // iterator will traverse all elements in this dataset from the
  // start.
  //
  // Ownership of the created iterator will be transferred to the caller.
  //
  // The prefix identifies the sequence of iterators leading up to the newly
  // created iterator.
  virtual std::unique_ptr<IteratorBase> MakeIterator(
      const string& prefix) const = 0;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // dataset.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this dataset.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;

  // A human-readable debug string for this dataset.
  virtual string DebugString() = 0;

  // Serializes the dataset and writes it to the `writer`.
  virtual Status Save(OpKernelContext* ctx, IteratorStateWriter* writer) const {
    return errors::Unimplemented("DatasetBase::Save");
  }

 protected:
  // TODO(srbs): Ideally all graph related logic should reside in
  // GraphDatasetBase. However, that would require Datasets defined in all ops
  // to derive from GraphDatasetBase. Once that is done we can move
  // DatasetGraphDefBuilder and AsGraphDefInternal to GraphDatasetBase.
  class DatasetGraphDefBuilder : public GraphDefBuilderWrapper {
   public:
    DatasetGraphDefBuilder(GraphDefBuilder* b) : GraphDefBuilderWrapper(b) {}
    Status AddParentDataset(OpKernelContext* ctx, const DatasetBase* dataset,
                            Node** output) {
      return dataset->AsGraphDefInternal(ctx, this, output);
    }
  };

  virtual Status AsGraphDefInternal(OpKernelContext* ctx,
                                    DatasetGraphDefBuilder* b,
                                    Node** node) const {
    return AsGraphDefInternal(b, node);
  }

  virtual Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                                    Node** node) const {
    return errors::Unimplemented("AsGraphDefInternal");
  }
};

// Base-class for datasets that are built by ops.
class GraphDatasetBase : public DatasetBase {
 public:
  GraphDatasetBase(OpKernelContext* ctx)
      : op_name_(ctx->op_kernel().type_string()) {}

  const string op_name() const { return op_name_; }

  Status Save(OpKernelContext* ctx,
              IteratorStateWriter* writer) const override {
    string serialized_graph_def;
    string output_node;
    TF_RETURN_IF_ERROR(Serialize(ctx, &serialized_graph_def, &output_node));
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(kDatasetGraphKey, serialized_graph_def));
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(kDatasetGraphOutputNodeKey, output_node));
    return Status::OK();
  }

  // Key for storing the Dataset graph in the serialized format.
  TF_EXPORT static const char kDatasetGraphKey[];

  // Key for storing the output node of the Dataset graph in the serialized
  // format.
  TF_EXPORT static const char kDatasetGraphOutputNodeKey[];

 private:
  Status Serialize(OpKernelContext* ctx, string* serialized_graph_def,
                   string* output_node) const;

  const string op_name_;
};

// Represents an iterator that is associated with a particular parent dataset.
template <class DatasetType>
class DatasetIterator : public IteratorBase {
 public:
  struct Params {
    // Owns one reference on the shared dataset resource.
    const DatasetType* dataset;

    // Identifies the sequence of iterators leading up to this iterator.
    const string prefix;
  };

  explicit DatasetIterator(const Params& params) : params_(params) {
    params_.dataset->Ref();
  }

  ~DatasetIterator() override { params_.dataset->Unref(); }

  // The dataset from which this iterator was created.
  const DatasetType* dataset() const { return params_.dataset; }

  // The sequence of iterators leading up to this iterator.
  const string prefix() const { return params_.prefix; }

  const DataTypeVector& output_dtypes() const override {
    return params_.dataset->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return params_.dataset->output_shapes();
  }

  Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) final {
    port::Tracing::TraceMe activity(params_.prefix);
    Status s = GetNextInternal(ctx, out_tensors, end_of_sequence);
    if (TF_PREDICT_FALSE(errors::IsOutOfRange(s) && !*end_of_sequence)) {
      s = errors::Internal(
          "Iterator \"", params_.prefix,
          "\" returned OutOfRange without setting `*end_of_sequence`. This "
          "indicates that an error may have occurred. Original message: ",
          s.error_message());
      LOG(ERROR) << s;
    }
    return s;
  }

  Status Save(OpKernelContext* ctx, IteratorStateWriter* writer) final {
    TF_RETURN_IF_ERROR(dataset()->Save(ctx, writer));
    return IteratorBase::Save(ctx, writer);
  }

 protected:
  // Internal implementation of GetNext that is wrapped in tracing logic.
  virtual Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) = 0;

  string full_name(const string& name) const {
    return strings::StrCat(prefix(), ":", name);
  }

 private:
  Params params_;
};

// Encapsulates the work required to plug a DatasetBase into the core TensorFlow
// graph execution engine.
class DatasetOpKernel : public OpKernel {
 public:
  DatasetOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) final;

 protected:
  // Subclasses should implement this method. It will be called during Compute
  // execution.
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase** output) = 0;

  template <typename T>
  Status ParseScalarArgument(OpKernelContext* ctx,
                             const StringPiece& argument_name, T* output) {
    const Tensor* argument_t;
    TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
    if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
      return errors::InvalidArgument(argument_name, " must be a scalar");
    }
    *output = argument_t->scalar<T>()();
    return Status::OK();
  }
};

// Encapsulates the work required to plug unary Datasets into the core
// TensorFlow graph execution engine.
class UnaryDatasetOpKernel : public DatasetOpKernel {
 public:
  UnaryDatasetOpKernel(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase** output) = 0;
};

// Encapsulates the work required to plug binary Datasets into the core
// TensorFlow graph execution engine.
class BinaryDatasetOpKernel : public DatasetOpKernel {
 public:
  BinaryDatasetOpKernel(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase* another_input,
                           DatasetBase** output) = 0;
};

// Validates and extracts a `DatasetBase` object from `tensor`.
//
// `tensor` must have been written by a call to SetVariantTensorToDataset().
//
// The retrieved pointer is a borrowed reference to the dataset, which is owned
// by the tensor. The consumer must either acquire its own reference to the
// dataset by calling `(*out_dataset)->Ref()`, or ensure that `tensor` is not
// destroyed or mutated while the retrieved pointer is in use.
Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset);

// Stores a `DatasetBase` object in `tensor`.
//
// The ownership of `dataset` is transferred to `tensor`.
Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DATASET_H_
