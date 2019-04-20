/* Geometric Convolution
 * Original author: Shiyi Lan
 * All Rights Reserved. 2019. 
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

#include <cstdio>

using namespace tensorflow;

REGISTER_OP("Aggregate")
  .Input("feat: float32")
  .Input("xyz: float32")
  .Attr("radius: float")
  .Attr("decayradius: float")
  .Attr("delta: int")
  .Output("out: float32")
  .Output("norm_buffer: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    ::tensorflow::shape_inference::ShapeHandle output, feat, norm;
    c->WithRank(c->input(0), 3, &feat);

    ::tensorflow::shape_inference::DimensionHandle num_channel = \
        c->MakeDim(c->Value(c->Dim(c->input(0), 2))/6);
    c->ReplaceDim(feat, 2, num_channel, &feat);

    c->set_output(0, feat);

    norm = c->MakeShape({c->Dim(c->input(0), 0), c->Dim(c->input(0), 1)});

    c->set_output(1, norm);

    return Status::OK();
  });


REGISTER_OP("AggregateGrad")
  .Input("feat: float32")
  .Input("xyz: float32")
  .Input("out_g: float32")
  .Attr("radius: float")
  .Attr("decayradius: float")
  .Attr("delta: int")
  .Output("grad: float32")
  .Output("norm_buffer: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle grad, norm;
    c->WithRank(c->input(0), 3, &grad);

    c->set_output(0, grad);

    norm = c->MakeShape({c->Dim(c->input(0), 0), c->Dim(c->input(0), 1)});

    c->set_output(1, norm);

    return Status::OK();
  });

void aggregateLauncher(int b, int n, int c, const float *feat, const float* xyz, float* out, float* norm, const float radius, const float decay_float, const int delta=0);

class AggregateGradGpuOp;

class AggregateGpuOp: public OpKernel {
  public:
    explicit AggregateGpuOp(OpKernelConstruction* context):OpKernel(context){
      OP_REQUIRES_OK(context, context->GetAttr("radius", &radius));
      OP_REQUIRES_OK(context, context->GetAttr("decayradius", &decay_radius));
      OP_REQUIRES_OK(context, context->GetAttr("delta", &delta));
    }
    void Compute(OpKernelContext* context) override {
      const Tensor& feat_tensor = context->input(0);
      const Tensor& xyz_tensor  = context->input(1);
      const Tensor& radius_tensor       = context->input(2);
      const Tensor& decay_radius_tensor = context->input(3);

      auto feat_flat    = feat_tensor.flat<float>();
      auto xyz_flat     = xyz_tensor.flat<float>();

      const float* feat  = &(feat_flat(0));
      const float* xyz   = &(xyz_flat(0));

      OP_REQUIRES(context, feat_tensor.dims()==3, errors::InvalidArgument("Aggregate expects (batch_size, num_point, num_channel) inp shape"));
      OP_REQUIRES(context, xyz_tensor.dims()==3, errors::InvalidArgument("Aggregate expects (batch_size, num_point, num_channel) inp shape"));

      int b = feat_tensor.shape().dim_size(0);
      int n = feat_tensor.shape().dim_size(1);
      int c = feat_tensor.shape().dim_size(2) / 6;

      Tensor* out_tensor = NULL;
      Tensor* norm_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, c}, &out_tensor));
      OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b, n}, &norm_tensor));
      auto out_flat = out_tensor->flat<float>();
      auto norm_flat = norm_tensor->flat<float>();
      float* out = &(out_flat(0));
      float* norm = &(norm_flat(0));
      aggregateLauncher(b, n, c, feat, xyz, out, norm, radius, decay_radius, delta);
    }

  friend AggregateGradGpuOp;
  private:
    float radius, decay_radius;
    int delta;
};
REGISTER_KERNEL_BUILDER(Name("Aggregate").Device(DEVICE_GPU), AggregateGpuOp)

void aggregategradLauncher(const int b, const int n, const int c, const float* feat, const float* xyz, const float* out, 
    float* norm, float* grad, const float radius, const float decay_radius, const int delta);


class AggregateGradGpuOp: public OpKernel {
  public:
    explicit AggregateGradGpuOp(OpKernelConstruction* context):OpKernel(context){
      OP_REQUIRES_OK(context, context->GetAttr("radius", &radius));
      OP_REQUIRES_OK(context, context->GetAttr("decayradius", &decay_radius));
      OP_REQUIRES_OK(context, context->GetAttr("delta", &delta));
    }
    void Compute(OpKernelContext* context) override {
      const Tensor& feat_tensor   = context->input(0);
      const Tensor& xyz_tensor    = context->input(1);
      const Tensor& out_tensor    = context->input(2);

      auto feat_flat      = feat_tensor.flat<float>();
      auto xyz_flat       = xyz_tensor.flat<float>();
      auto out_flat       = out_tensor.flat<float>();

      const float* feat   = &(feat_flat(0));
      const float* xyz    = &(xyz_flat(0));
      const float* out    = &(out_flat(0));

      OP_REQUIRES(context, out_tensor.dims()==3, errors::InvalidArgument("Aggregate Grad expects (batch_size, num_point, num_channel/6) inp shape"));

      int b = feat_tensor.shape().dim_size(0);
      int n = feat_tensor.shape().dim_size(1);
      int c = feat_tensor.shape().dim_size(2) / 6;

      Tensor *grad_tensor = NULL;
      Tensor *norm_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, c * 6}, &grad_tensor));
      OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b, n}, &norm_tensor));

      auto grad_flat = grad_tensor->flat<float>();
      auto norm_flat = norm_tensor->flat<float>();
      float* grad = &(grad_flat(0));
      float* norm = &(norm_flat(0));
      aggregategradLauncher(b, n, c, feat, xyz, out, norm, grad, radius, decay_radius, delta);
    }
  private:
    float radius, decay_radius;
    int delta;
};
REGISTER_KERNEL_BUILDER(Name("AggregateGrad").Device(DEVICE_GPU), AggregateGradGpuOp);
