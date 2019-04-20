/* Geometric Convolution
 * Original author: Shiyi Lan
 * All Rights Reserved. 2019.
 */

#define get_square_euclidean_dist(x,y,z) \
        ((x)*(x)+(y)*(y)+(z)*(z))
#define _CUDA_NUM_THREADS 512
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

static int _GET_BLOCKS(const int N) {
  return (N + _CUDA_NUM_THREADS - 1) / _CUDA_NUM_THREADS;
}

__global__ void Normalization(const int top_count, float* aggre_feat, 
    const float* norm_buffer, const int num_batchs, const int num_points, 
    const int num_channels) {

  CUDA_KERNEL_LOOP(index, top_count) {
    const int base = index * num_channels;
    for (int i = 0; i < num_channels; ++i)
      aggre_feat[base + i] /= norm_buffer[index] + 1;
  }
}



__global__ void AggregateKernel(const int num_pairs, const int num_batchs, const int num_points, const int num_channels,
    const float* feat, const float* xyz, float* aggre_feat, float* norm_buffer, const float std_square_dist,
    const float square_decay_dist, const int delta) {

  CUDA_KERNEL_LOOP(index, num_pairs) {
    const int p0 = index % num_points;
    const int p1 = index / num_points % num_points;
    if (p0 == p1) continue;
    const int b  = index / (num_points * num_points);

    const int pos0 = (b * num_points + p0) * 3;
    const int pos1 = (b * num_points + p1) * 3;
    const float x0 = xyz[pos0], y0 = xyz[pos0+1], z0 = xyz[pos0+2];
    const float x1 = xyz[pos1], y1 = xyz[pos1+1], z1 = xyz[pos1+2];
    const float dx = x0 - x1, dy = y0 - y1, dz = z0 - z1;

    const float square_dist = get_square_euclidean_dist(dx, dy, dz);
    const float dist = sqrt(square_dist);
    if (dist < 1e-4) continue;
    float dist_weight = 0;
    if (square_dist < square_decay_dist) {
      if (square_dist <= std_square_dist)
        dist_weight = 1;
      else
        dist_weight = max(1 - (square_dist - std_square_dist) / (square_decay_dist - std_square_dist), 0.0);

      const float weights[3] = {abs(dx)/dist, abs(dy)/dist, abs(dz)/dist};

      int act[3];
      act[0] = (dx > 0) ? 1 : 0;
      act[1] = (dy > 0) ? 1 : 0;
      act[2] = (dz > 0) ? 1 : 0;

      atomicAdd(norm_buffer + b * num_points + p1, dist_weight);

      for (int i = 0; i < 3; ++i) {
        int dir = (i<<1)  + act[i];
        int p1_idx = (b * num_points + p1) * num_channels;
        int p0_idx = ((b * num_points + p0) * 6 + dir) * num_channels;
        float weight = weights[i] * dist_weight;
        for (int c = 0; c < num_channels; ++c)
          if (!delta)
            atomicAdd(aggre_feat + p1_idx + c, feat[p0_idx + c] * weight);
          else
            atomicAdd(aggre_feat + p1_idx + c, (feat[p0_idx + c] - feat[((b * num_points + p1) + dir) * num_channels]) * weight);
      }
    }
  }
}


__global__ void AggregateGradKernel(const int num_pairs, const int num_batchs, const int num_points, const int num_channels,
    const float* top_feat_grad, const float* xyz, float* bottom_feat_grad, float* norm_buffer, const float std_square_dist,
    const float square_decay_dist, const int delta) {

  CUDA_KERNEL_LOOP(index, num_pairs) {
    const int p0 = index % num_points;
    const int p1 = index / num_points % num_points;
    if (p0 == p1) continue;
    const int b  = index / (num_points * num_points);

    const int pos0 = (b * num_points + p0) * 3;
    const int pos1 = (b * num_points + p1) * 3;
    const float x0 = xyz[pos0], y0 = xyz[pos0+1], z0 = xyz[pos0+2];
    const float x1 = xyz[pos1], y1 = xyz[pos1+1], z1 = xyz[pos1+2];
    const float dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;

    const float square_dist = get_square_euclidean_dist(dx, dy, dz);
    const float dist = sqrt(square_dist);
    if (dist < 1e-4) continue;
    float dist_weight = 0;
    if (square_dist < square_decay_dist) {
      if (square_dist <= std_square_dist)
        dist_weight = 1;
      else
        dist_weight = max(1 - (square_dist - std_square_dist) / (square_decay_dist - std_square_dist), .0);

      const float weights[3] = {abs(dx)/dist, abs(dy)/dist, abs(dz)/dist};

      int act[3];
      act[0] = (dx > 0) ? 1 : 0;
      act[1] = (dy > 0) ? 1 : 0;
      act[2] = (dz > 0) ? 1 : 0;

      atomicAdd(norm_buffer + b * num_points + p1, dist_weight);

      for (int i = 0; i < 3; ++i) {
        int dir = (i<<1)  + act[i];
        int p0_idx = (b * num_points + p0) * num_channels;
        int p1_idx = ((b * num_points + p1) * 6 + dir) * num_channels;
        float weight = weights[i] * dist_weight;
        for (int c = 0; c < num_channels; ++c)
          atomicAdd(bottom_feat_grad + p1_idx + c, top_feat_grad[p0_idx + c] * weight);
      }
    }
  }
}



void aggregateLauncher(int b, int n, int c, const float* feat, const float* xyz, float* out, float* norm_buffer, const float radius, const float decay_radius, const int delta=0) {
  const int num_pair = b * n * n;
  const int top_count = b * n;
  cudaMemset(norm_buffer, 0, sizeof(float) * b * n);
  cudaMemset(out, 0, sizeof(float) * b * n * c);
  AggregateKernel<<<_GET_BLOCKS(num_pair), _CUDA_NUM_THREADS>>>(num_pair, b, n, c, feat, xyz, out, norm_buffer, radius * radius, decay_radius * decay_radius, delta);
  Normalization<<<_GET_BLOCKS(top_count), _CUDA_NUM_THREADS>>>(top_count, out, norm_buffer, b, n, c);
}



void aggregategradLauncher(const int b, const int n, const int c, const float* feat, const float* xyz, const float* out, float* norm_buffer, float* grad, const float radius, const float decay_radius, const int delta=0) {
  const int num_pair = b * n * n;
  const int top_count = b * n;
  cudaMemset(norm_buffer, 0, sizeof(float) * b * n);
  cudaMemset(grad, 0, sizeof(float) * b * n * c * 6);
  AggregateGradKernel<<<_GET_BLOCKS(num_pair), _CUDA_NUM_THREADS>>>(num_pair, b, n, c, out, xyz, grad, norm_buffer, radius * radius, decay_radius * decay_radius, delta);
  Normalization<<<_GET_BLOCKS(top_count), _CUDA_NUM_THREADS>>>(top_count, grad, norm_buffer, b, n, c * 6);
}


