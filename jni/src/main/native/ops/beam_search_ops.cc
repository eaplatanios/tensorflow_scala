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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "beam_search_ops.h"

#include <memory>
#include <vector>

// TODO: [JNI] Remove when op_kernel.h is fixed.
#include "tensorflow/core/framework/kernel_def.pb.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("GatherTree")
    .Input("step_ids: T")
    .Input("parent_ids: T")
    .Input("max_sequence_lengths: int32")
    .Input("end_token: T")
    .Output("beams: T")
    .Attr("T: {int32}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle step_ids, parent_ids, max_sequence_lengths, end_token;

      // step_ids, parent_ids, and output are all shaped:
      //   [max_time, batch_size, beam_width].
      // max_sequence_length is shaped [batch_size] and end_token is a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &step_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &parent_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &max_sequence_lengths));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &end_token));
      TF_RETURN_IF_ERROR(c->Merge(step_ids, parent_ids, &step_ids));
      DimensionHandle batch_size = c->Dim(step_ids, 1);
      TF_RETURN_IF_ERROR(
          c->Merge(batch_size, c->Dim(max_sequence_lengths, 0), &batch_size));
      ShapeHandle step_ids_prefix = c->Matrix(c->Dim(step_ids, 0), batch_size);
      TF_RETURN_IF_ERROR(c->MergePrefix(step_ids, step_ids_prefix, &step_ids,
                                        &step_ids_prefix));

      c->set_output(0, step_ids);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Calculates the full beams from the per-step ids and parent beam ids.

This op implements the following mathematical equations:

```python
TODO(ebrevdo): fill in
```

step_ids: `[max_time, batch_size, beam_width]`.
parent_ids: `[max_time, batch_size, beam_width]`.
max_sequence_lengths: `[batch_size]`.
end_token: `[]`.
beams: `[max_time, batch_size, beam_width]`.
)doc");

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class GatherTreeOp : public OpKernel {
 public:
  explicit GatherTreeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Device& device = ctx->eigen_device<Device>();
    const Tensor& step_ids = ctx->input(0);
    const Tensor& parent_ids = ctx->input(1);
    const Tensor& max_sequence_lengths = ctx->input(2);
    const Tensor& end_token = ctx->input(3);
    const TensorShape& step_ids_shape = step_ids.shape();
    OP_REQUIRES(
        ctx, step_ids_shape.dims() == 3,
        errors::InvalidArgument("step_ids must be a 3-tensor, saw shape: ",
                                step_ids_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(max_sequence_lengths.shape()),
                errors::InvalidArgument(
                    "max_sequence_lengths must be a vector, saw shape: ",
                    max_sequence_lengths.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(end_token.shape()),
        errors::InvalidArgument("end_token must be a scalar, saw shape: ",
                                end_token.shape().DebugString()));
    OP_REQUIRES(
        ctx, step_ids_shape == parent_ids.shape(),
        errors::InvalidArgument(
            "step_ids.shape must match parent_ids.shape.  but shapes are: ",
            step_ids_shape.DebugString(), " and ",
            parent_ids.shape().DebugString()));
    OP_REQUIRES(
        ctx,
        step_ids_shape.dim_size(1) == max_sequence_lengths.shape().dim_size(0),
        errors::InvalidArgument("batch size dimensions step_ids.shape[1] and "
                                "max_sequence_lengths.shape[0] must match.  "
                                "but shapes are: ",
                                step_ids_shape.DebugString(), " and ",
                                max_sequence_lengths.shape().DebugString()));
    Tensor* beams;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, step_ids_shape, &beams));
    typename TTypes<T, 3>::ConstTensor step_ids_t(step_ids.tensor<T, 3>());
    typename TTypes<T, 3>::ConstTensor parent_ids_t(parent_ids.tensor<T, 3>());
    typename TTypes<int32>::ConstVec max_seq_lens_t =
        max_sequence_lengths.vec<int32>();
    typename TTypes<T>::ConstScalar end_token_t(end_token.scalar<T>());
    typename TTypes<T, 3>::Tensor beams_t(beams->tensor<T, 3>());
    const T end_token_value = end_token_t();
    functor::GatherTree<Device, T>()(ctx, device, step_ids_t, parent_ids_t,
                                     max_seq_lens_t, end_token_value, beams_t);
  }
};

#define REGISTER_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("GatherTree").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GatherTreeOp<CPUDevice, T>);
REGISTER_KERNEL(int32);
#undef REGISTER_KERNEL

namespace functor {

// CPU specialization
template <>
struct GatherTree<CPUDevice, int32> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  TTypes<int32, 3>::ConstTensor step_ids,
                  TTypes<int32, 3>::ConstTensor parent_ids,
                  TTypes<int32>::ConstVec max_sequence_lengths,
                  const int32 end_token, TTypes<int32, 3>::Tensor beams) {
    const int32 max_time = parent_ids.dimension(0);
    const int32 batch_size = parent_ids.dimension(1);
    const int32 beam_width = parent_ids.dimension(2);
    beams.setConstant(end_token);

    auto DoWork = [&, ctx, end_token](int start_batch_beam,
                                      int limit_batch_beam) {
      for (int32 i = start_batch_beam; i < limit_batch_beam; ++i) {
        const int32 batch = i / beam_width;
        const int32 beam = i % beam_width;
        const int32 max_seq_len_b =
            Eigen::numext::mini(max_time, max_sequence_lengths(batch));
        if (max_seq_len_b <= 0) {
          continue;
        }
        beams(max_seq_len_b - 1, batch, beam) =
            step_ids(max_seq_len_b - 1, batch, beam);
        int32 parent = parent_ids(max_seq_len_b - 1, batch, beam);
        for (int32 level = max_seq_len_b - 2; level >= 0; --level) {
          if (parent < 0 || parent > beam_width) {
            ctx->SetStatus(
                errors::InvalidArgument("Saw invalid parent id ", parent,
                                        " at (batch, time, beam) == (", batch,
                                        ", ", level, ", ", beam, ")"));
            return;
          }
          beams(level, batch, beam) = step_ids(level, batch, parent);
          parent = parent_ids(level, batch, parent);
        }
        // Not necessary when using a BeamSearchDecoder, but necessary
        // when a user feeds in possibly broken trajectory (i.e., non-eos
        // entries in a beam following eos entries).
        bool finished = false;
        for (int32 time = 0; time < max_seq_len_b; ++time) {
          if (finished) {
            beams(time, batch, beam) = end_token;
          } else if (beams(time, batch, beam) == end_token) {
            finished = true;
          }
        }
      }
    };
    // Guesstimate of cost; ~5 lookup/store/compare per inner beam
    // traversal time step.
    const int64 batch_beam_cost =
        Eigen::TensorOpCost::DivCost<int32>() +
        6 * Eigen::TensorOpCost::AddCost<int32>() +
        2 * max_time * (5 * Eigen::TensorOpCost::AddCost<int32>());
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          batch_size * beam_width, batch_beam_cost, DoWork);
  }
};

}  // namespace functor

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                            \
  template <>                                                          \
  void GatherTree<GPUDevice, T>::operator()(                           \
      OpKernelContext* ctx, const GPUDevice& d,                        \
      typename TTypes<T, 3>::ConstTensor step_ids,                     \
      typename TTypes<T, 3>::ConstTensor parent_ids,                   \
      TTypes<int32>::ConstVec max_sequence_lengths, const T end_token, \
      typename TTypes<T, 3>::Tensor beams);                            \
  extern template struct GatherTree<GPUDevice, T>;

DECLARE_GPU_SPEC(int32);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                          \
  REGISTER_KERNEL_BUILDER(Name("GatherTree")            \
                              .Device(DEVICE_GPU)       \
                              .TypeConstraint<T>("T")   \
                              .HostMemory("end_token"), \
                          GatherTreeOp<GPUDevice, T>);

REGISTER_GPU_KERNEL(int32);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
