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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Sets the use_bfloat16 on a container of test cases according to the values in
// use_bfloat16_params. Generates one set of test cases for each values in
// use_bfloat16_params with that value. Returns the result.
template <typename TestCase>
std::vector<TestCase> ExpandUseBfloat16(
    tensorflow::gtl::ArraySlice<bool> use_bfloat16_params,
    tensorflow::gtl::ArraySlice<TestCase> specs) {
  std::vector<TestCase> expanded;
  for (bool use_bfloat16 : use_bfloat16_params) {
    for (const auto& spec : specs) {
      expanded.push_back(spec);
      expanded.back().use_bfloat16 = use_bfloat16;
    }
  }
  return expanded;
}

// A client library test establishes an in-process XLA client connection.
class ClientLibraryTestBase : public ::testing::Test {
 protected:
  explicit ClientLibraryTestBase(
      perftools::gputools::Platform* platform = nullptr);

  // Creates a new ClientLibraryTestBase with custom client options.
  ClientLibraryTestBase(perftools::gputools::Platform* platform,
                        const LocalClientOptions& client_options);

  // Returns the name of the test currently being run.
  string TestName() const;

  void SetFastMathDisabled(bool disabled) {
    execution_options_.mutable_debug_options()->set_xla_enable_fast_math(
        !disabled);
  }

  void SetSeed(uint64 seed) { execution_options_.set_seed(seed); }

  // Provides mutable access to the execution DebugOptions field; this lets
  // tests tweak the options that will be used to compile/run the graph.
  DebugOptions* mutable_debug_options() {
    return execution_options_.mutable_debug_options();
  }

  // TODO(b/25566808): Add helper that populates a literal from a testdata file.

  // Convenience methods for building and running a computation with the member
  // execution options. Modify execution_options_ in your test if you want to
  // customize the options.
  template <typename BuilderT>
  StatusOr<std::unique_ptr<GlobalData>> Execute(
      BuilderT* builder, tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  // TODO(b/74197823): Remove the template type 'BuilderT' in all methods once
  // the migration to XlaBuilder is complete.

  template <typename BuilderT>
  StatusOr<std::unique_ptr<Literal>> ExecuteAndTransfer(
      BuilderT* builder, tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_output_layout = nullptr);

  StatusOr<std::unique_ptr<Literal>> ExecuteAndTransfer(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_output_layout = nullptr);

  StatusOr<std::unique_ptr<Literal>> ExecuteAndTransfer(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_output_layout = nullptr);

  // Convenience OrDie variants of above methods.
  std::unique_ptr<GlobalData> ExecuteOrDie(
      ComputationBuilder* builder,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  std::unique_ptr<Literal> ExecuteAndTransferOrDie(
      ComputationBuilder* builder,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  // Run a computation and return its value as a string. If an error
  // occurs, then instead return the error as a string.
  string ExecuteToString(XlaBuilder* builder,
                         tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  string ExecuteToString(ComputationBuilder* builder,
                         tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  // Convenience methods for building and running a computation, transferring
  // the result, and comparing it to the expected value(s). Methods are
  // templated on the native host type which maps to specific XLA types (See
  // ComputationBuilder/XlaBuilder for details). For each rank, two forms are
  // provided: one for floating point types with an ErrorSpec parameter, and one
  // for integral types without the ErrorSpec parameter.
  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR0(BuilderT* builder, NativeT expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR0(BuilderT* builder, NativeT expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR1(BuilderT* builder,
                           tensorflow::gtl::ArraySlice<NativeT> expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR1(BuilderT* builder,
                           tensorflow::gtl::ArraySlice<NativeT> expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  // As above, but uses a bitmap to hold the predicate vector to avoid
  // deficiencies of vector<bool>.
  void ComputeAndCompareR1(ComputationBuilder* builder,
                           const tensorflow::core::Bitmap& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR2(BuilderT* builder, const Array2D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR2(BuilderT* builder, const Array2D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR3(BuilderT* builder, const Array3D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR3(BuilderT* builder, const Array3D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR4(BuilderT* builder, const Array4D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT, typename BuilderT>
  void ComputeAndCompareR4(BuilderT* builder, const Array4D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  // Build and run the computation and compare the result with the given
  // literal. shape_with_layout indicates the result layout to request when
  // calling Execute.
  template <typename BuilderT>
  void ComputeAndCompareLiteral(
      BuilderT* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_layout = nullptr);
  template <typename BuilderT>
  void ComputeAndCompareLiteral(
      BuilderT* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error,
      const Shape* shape_with_layout = nullptr);

  // ComputeAndCompare variant which returns an error status.
  template <typename BuilderT>
  tensorflow::Status ComputeAndCompareLiteralWithStatus(
      BuilderT* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_layout = nullptr);
  template <typename BuilderT>
  tensorflow::Status ComputeAndCompareLiteralWithStatus(
      BuilderT* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error,
      const Shape* shape_with_layout = nullptr);

  // Compare the result of the computation to a strings. In XLA strings are
  // represented using rank-1 U8 shapes.
  void ComputeAndCompareR1U8(
      ComputationBuilder* builder, tensorflow::StringPiece expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  // Convenience method for running a built computation, transferring the
  // result, and comparing it to the expected tuple literal.
  template <typename BuilderT>
  void ComputeAndCompareTuple(
      BuilderT* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename BuilderT>
  void ComputeAndCompareTuple(
      BuilderT* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error);

  // Convenience method for running a built computation and comparing the result
  // with the HloEvaluator.
  void ComputeAndCompare(ComputationBuilder* builder,
                         const ComputationDataHandle& operand,
                         tensorflow::gtl::ArraySlice<Literal> arguments);
  void ComputeAndCompare(ComputationBuilder* builder,
                         const ComputationDataHandle& operand,
                         tensorflow::gtl::ArraySlice<Literal> arguments,
                         ErrorSpec error);

  // Create scalar operations for use in reductions.
  Computation CreateScalarRelu();
  Computation CreateScalarMax();
  Computation CreateScalarReluSensitivity();

  // Special case convenience functions for creating filled arrays.

  // Creates an array of pseudorandom values lying between the given minimum and
  // maximum values.
  template <typename NativeT>
  std::vector<NativeT> CreatePseudorandomR1(const int width, NativeT min_value,
                                            NativeT max_value, uint32 seed);
  template <typename NativeT>
  std::unique_ptr<Array2D<NativeT>> CreatePseudorandomR2(const int rows,
                                                         const int cols,
                                                         NativeT min_value,
                                                         NativeT max_value,
                                                         uint32 seed);

  // Creates a (rows x cols) array filled in the following form:
  //
  //  [      0              1 ...                   cols-1]
  //  [  1,000          1,001 ...          1000.0 + cols-1]
  //  [    ...            ... ...                      ...]
  //  [(rows-1)*1000.0    ... ... (rows-1)*1000.0 + cols-1]
  //
  // If provided, offset is added uniformly to every element (e.g. an offset of
  // 64 would cause 0 in the above to be 64, 1 to be 65, 1000 to be 1064, etc.)
  std::unique_ptr<Array2D<float>> CreatePatternedMatrix(const int rows,
                                                        const int cols,
                                                        float offset = 0.0);

  // Creates a (rows x cols) array as above, padded out to
  // (rows_padded x cols_padded) with zeroes.  Requires rows_padded >= rows
  // and cols_padded > cols.
  std::unique_ptr<Array2D<float>> CreatePatternedMatrixWithZeroPadding(
      const int rows, const int cols, const int rows_padded,
      const int cols_padded);

  // Creates a parameter instruction, transfers the literal for the parameter to
  // server, then stores into "data_handle" the global handle for that
  // parameter. When the use_bfloat16 flag is set but the literal has F32
  // elements, the literal will be converted to BF16 before being transferred.
  template <typename BuilderT, typename HandleT>
  std::unique_ptr<GlobalData> CreateParameterAndTransferLiteral(
      int64 parameter_number, const Literal& literal, const string& name,
      BuilderT* builder, HandleT* data_handle);

  // As above, but the caller can specify the device that the literal is
  // transferred to. If device_handle is nullptr, the literal will be
  // transferred to the default device.
  template <typename BuilderT, typename HandleT>
  std::unique_ptr<GlobalData> CreateParameterAndTransferLiteral(
      int64 parameter_number, const Literal& literal, const string& name,
      const DeviceHandle* device_handle, BuilderT* builder,
      HandleT* data_handle);

  // Creates a parameter instruction and sets the value that will be passed to
  // the computation as specified. This function must be used for all parameters
  // or none and no parameters must be passed when invoking the computation if
  // using this mechanism. If using this mechanism, then each parameter must be
  // set exactly once. The first added parameter gets index 0, then 1 and so on.
  ComputationDataHandle AddParam(const Literal& argument,
                                 ComputationBuilder* builder);
  XlaOp AddParam(const Literal& argument, XlaBuilder* builder);

  template <class T>
  ComputationDataHandle AddParam(const Array<T>& argument,
                                 ComputationBuilder* builder) {
    return AddParam(*Literal::CreateFromArray(argument), builder);
  }
  template <class T>
  XlaOp AddParam(const Array<T>& argument, XlaBuilder* builder) {
    return AddParam(*Literal::CreateFromArray(argument), builder);
  }

  // Creates a constant instruction with the given literal. When the
  // use_bfloat16 flag is set but the literal has F32 elements, the elements
  // will be converted to BF16s.
  ComputationDataHandle CreateConstantFromLiteral(const Literal& literal,
                                                  ComputationBuilder* builder);
  XlaOp CreateConstantFromLiteral(const Literal& literal, XlaBuilder* builder);

  // Creates a constant instruction with the given array. When the use_bfloat16
  // flag is set but the array has float elements, the elements will be
  // converted to bfloat16s.
  template <typename NativeT>
  ComputationDataHandle CreateConstantFromArray(const Array<NativeT>& array,
                                                ComputationBuilder* builder) {
    return CreateConstantFromLiteral(*Literal::CreateFromArray(array), builder);
  }

  template <typename NativeT>
  XlaOp CreateConstantFromArray(const Array<NativeT>& array,
                                XlaBuilder* builder) {
    return CreateConstantFromLiteral(*Literal::CreateFromArray(array), builder);
  }

  // Same as CreateConstantFromArray, but for scalars.
  template <typename NativeT>
  ComputationDataHandle CreateConstantFromScalar(NativeT value,
                                                 ComputationBuilder* builder) {
    return CreateConstantFromLiteral(*Literal::CreateR0<NativeT>(value),
                                     builder);
  }

  template <typename NativeT>
  XlaOp CreateConstantFromScalar(NativeT value, XlaBuilder* builder) {
    return CreateConstantFromLiteral(*Literal::CreateR0<NativeT>(value),
                                     builder);
  }

  // Creates a parameter instruction that wraps a given value and then stores
  // into "data_handle" the global handle for that parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT, typename BuilderT, typename HandleT>
  std::unique_ptr<GlobalData> CreateR0Parameter(NativeT value,
                                                int64 parameter_number,
                                                const string& name,
                                                BuilderT* builder,
                                                HandleT* data_handle);

  // Creates a parameter instruction that wraps the given values and then stores
  // into "data_handle" the global handle for that parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT, typename BuilderT, typename HandleT>
  std::unique_ptr<GlobalData> CreateR1Parameter(
      tensorflow::gtl::ArraySlice<NativeT> values, int64 parameter_number,
      const string& name, BuilderT* builder, HandleT* data_handle);

  // Creates a parameter instruction that wraps the given constant array
  // "array_2d" and then stores to "data_handle" the global handle for that
  // parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT, typename BuilderT, typename HandleT>
  std::unique_ptr<GlobalData> CreateR2Parameter(
      const Array2D<NativeT>& array_2d, int64 parameter_number,
      const string& name, BuilderT* builder, HandleT* data_handle);

  // Creates a parameter instruction that wraps the given constant array
  // "array_3d" and then stores to "data_handle" the global handle for that
  // parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT, typename BuilderT, typename HandleT>
  std::unique_ptr<GlobalData> CreateR3Parameter(
      const Array3D<NativeT>& array_3d, int64 parameter_number,
      const string& name, BuilderT* builder, HandleT* data_handle);

  // Getter and setter for the use_bfloat16 flag, which indicates whether to run
  // tests with all float-type input/output converted to bfloat16.
  bool use_bfloat16() const { return use_bfloat16_; }
  void set_use_bfloat16(bool value) { use_bfloat16_ = value; }

  // The float type used in this test, BF16 or F32 according to use_bfloat16.
  PrimitiveType FloatType() const { return use_bfloat16_ ? BF16 : F32; }

  Client* client_;
  ExecutionOptions execution_options_;

 private:
  // Build and run the computation with all permutations of output layouts.
  tensorflow::Status ComputeAndCompareLiteralWithAllOutputLayouts(
      const xla::Computation& computation, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const std::function<void(const Literal& actual,
                               const string& error_message)>& verify_output);
  // Build and run the computation with all permutations of layouts of all input
  // arguments.
  tensorflow::Status ComputeAndCompareLiteralWithAllInputLayouts(
      const xla::Computation& computation, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const std::function<void(const Literal& actual,
                               const string& error_message)>& verify_output,
      const Shape* output_with_layout = nullptr);

  tensorflow::Status ComputeAndCompareLiteralWithAllOutputLayouts(
      const xla::XlaComputation& computation, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const std::function<void(const Literal& actual,
                               const string& error_message)>& verify_output);
  tensorflow::Status ComputeAndCompareLiteralWithAllInputLayouts(
      const xla::XlaComputation& computation, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const std::function<void(const Literal& actual,
                               const string& error_message)>& verify_output,
      const Shape* output_with_layout = nullptr);

  // Executes the computation and calculates the expected reference value using
  // the HloEvaluator. Returns two literal in the order of (expected, actual).
  StatusOr<std::pair<std::unique_ptr<Literal>, std::unique_ptr<Literal>>>
  ComputeValueAndReference(ComputationBuilder* builder,
                           const ComputationDataHandle& operand,
                           tensorflow::gtl::ArraySlice<Literal> arguments);

  // Whether to run tests with all float-type input/output converted to
  // bfloat16.
  bool use_bfloat16_ = false;

  // Arguments to be passed to the computation when it runs.
  std::vector<std::unique_ptr<GlobalData>> arguments_;
};

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR0(
    BuilderT* builder, NativeT expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR0<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR0(
    BuilderT* builder, NativeT expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value,
                "Float or complex type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR0<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR1(
    BuilderT* builder, tensorflow::gtl::ArraySlice<NativeT> expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR1<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR1(
    BuilderT* builder, tensorflow::gtl::ArraySlice<NativeT> expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value,
                "Float or complex type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR1<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR2(
    BuilderT* builder, const Array2D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR2FromArray2D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR2(
    BuilderT* builder, const Array2D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value,
                "Float or complex type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR2FromArray2D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR3(
    BuilderT* builder, const Array3D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR3FromArray3D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR3(
    BuilderT* builder, const Array3D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value,
                "Float or complex type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR3FromArray3D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR4(
    BuilderT* builder, const Array4D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR4FromArray4D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT, typename BuilderT>
void ClientLibraryTestBase::ComputeAndCompareR4(
    BuilderT* builder, const Array4D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value,
                "Float or complex type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR4FromArray4D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT, typename BuilderT, typename HandleT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR0Parameter(
    NativeT value, int64 parameter_number, const string& name,
    BuilderT* builder, HandleT* data_handle) {
  std::unique_ptr<Literal> literal = Literal::CreateR0(value);
  if (use_bfloat16_ && literal->shape().element_type() == F32) {
    literal = LiteralTestUtil::ConvertF32ToBF16(*literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(*literal).ConsumeValueOrDie();
  *data_handle = builder->Parameter(parameter_number, literal->shape(), name);
  return data;
}

template <typename NativeT, typename BuilderT, typename HandleT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR1Parameter(
    tensorflow::gtl::ArraySlice<NativeT> values, int64 parameter_number,
    const string& name, BuilderT* builder, HandleT* data_handle) {
  std::unique_ptr<Literal> literal = Literal::CreateR1(values);
  if (use_bfloat16_ && literal->shape().element_type() == F32) {
    literal = LiteralTestUtil::ConvertF32ToBF16(*literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(*literal).ConsumeValueOrDie();
  *data_handle = builder->Parameter(parameter_number, literal->shape(), name);
  return data;
}

template <typename NativeT, typename BuilderT, typename HandleT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR2Parameter(
    const Array2D<NativeT>& array_2d, int64 parameter_number,
    const string& name, BuilderT* builder, HandleT* data_handle) {
  std::unique_ptr<Literal> literal = Literal::CreateR2FromArray2D(array_2d);
  if (use_bfloat16_ && literal->shape().element_type() == F32) {
    literal = LiteralTestUtil::ConvertF32ToBF16(*literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(*literal).ConsumeValueOrDie();
  *data_handle = builder->Parameter(parameter_number, literal->shape(), name);
  return data;
}

template <typename NativeT, typename BuilderT, typename HandleT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR3Parameter(
    const Array3D<NativeT>& array_3d, int64 parameter_number,
    const string& name, BuilderT* builder, HandleT* data_handle) {
  std::unique_ptr<Literal> literal = Literal::CreateR3FromArray3D(array_3d);
  if (use_bfloat16_ && literal->shape().element_type() == F32) {
    literal = LiteralTestUtil::ConvertF32ToBF16(*literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(*literal).ConsumeValueOrDie();
  *data_handle = builder->Parameter(parameter_number, literal->shape(), name);
  return data;
}

template <typename NativeT>
std::vector<NativeT> ClientLibraryTestBase::CreatePseudorandomR1(
    const int width, NativeT min_value, NativeT max_value, uint32 seed) {
  std::vector<NativeT> result(width);
  PseudorandomGenerator<NativeT> generator(min_value, max_value, seed);
  for (int i = 0; i < width; ++i) {
    result[i] = generator.get();
  }
  return result;
}

template <typename NativeT>
std::unique_ptr<Array2D<NativeT>> ClientLibraryTestBase::CreatePseudorandomR2(
    const int rows, const int cols, NativeT min_value, NativeT max_value,
    uint32 seed) {
  auto result = MakeUnique<Array2D<NativeT>>(rows, cols);
  PseudorandomGenerator<NativeT> generator(min_value, max_value, seed);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      (*result)(y, x) = generator.get();
    }
  }
  return result;
}

template <typename BuilderT, typename HandleT>
std::unique_ptr<GlobalData>
ClientLibraryTestBase::CreateParameterAndTransferLiteral(int64 parameter_number,
                                                         const Literal& literal,
                                                         const string& name,
                                                         BuilderT* builder,
                                                         HandleT* data_handle) {
  return CreateParameterAndTransferLiteral(parameter_number, literal, name,
                                           nullptr, builder, data_handle);
}

template <typename BuilderT, typename HandleT>
std::unique_ptr<GlobalData>
ClientLibraryTestBase::CreateParameterAndTransferLiteral(
    int64 parameter_number, const Literal& literal, const string& name,
    const DeviceHandle* device_handle, BuilderT* builder,
    HandleT* data_handle) {
  const Literal* param_literal = &literal;
  std::unique_ptr<Literal> converted_literal;
  if (use_bfloat16_) {
    converted_literal = LiteralTestUtil::ConvertF32ToBF16(literal);
    param_literal = converted_literal.get();
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(*param_literal, device_handle)
          .ConsumeValueOrDie();
  *data_handle =
      builder->Parameter(parameter_number, param_literal->shape(), name);
  return data;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_
