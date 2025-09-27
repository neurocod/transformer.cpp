#include "layers/MultiHeadAttention.h"

MultiHeadAttention::MultiHeadAttention(const std::string& name, int embedDim, int numHeads):
  _embedDim(embedDim), _numHeads(numHeads),
  head_dim_(embedDim / numHeads),
  query_proj_(embedDim, embedDim, name + "att.q"),
  key_proj_(embedDim, embedDim, name + "att.k"),
  value_proj_(embedDim, embedDim, name + "att.v"),
  output_proj_(embedDim, embedDim, name + "att.out") {
  if (_embedDim % _numHeads != 0) {
    throw std::runtime_error(
        "Embedding dimension must be divisible by the number of heads.");
  }
}

Tensor::Ptr MultiHeadAttention::scaled_dot_product_attention(
    Tensor::Ptr &q, Tensor::Ptr &k,
    Tensor::Ptr &v, Tensor::Ptr mask) {
  // q, k, v shapes are typically (batchSize, numHeads, sequenceLength, head_dim)

  // Calculate attention scores: scores = Q . K^T
  // K^T needs transposing the last two dimensions: (batchSize, numHeads, head_dim, sequenceLength)
  std::vector<int> k_shape = k->shape();
  std::vector<int> k_transpose_perm(k_shape.size());
  std::iota(k_transpose_perm.begin(), k_transpose_perm.end(), 0);
  // Swap the last two dimensions
  if (k_transpose_perm.size() >= 2) {
    std::swap(k_transpose_perm[k_transpose_perm.size() - 1],
              k_transpose_perm[k_transpose_perm.size() - 2]);
  }
  Tensor::Ptr k_transposed = k->transpose(k_transpose_perm);

  // Perform batched matrix multiplication: (batchSize, numHeads,
  // sequenceLength, head_dim) . (batchSize, numHeads, head_dim,
  // sequenceLength) Result shape: (batchSize, numHeads, sequenceLength,
  // sequenceLength)
  Tensor::Ptr scores = q->dot(k_transposed);

  // Scale the scores by the square root of the head dimension
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  Tensor::Ptr scale_tensor = Tensor::create(
      {1}, std::make_shared<Vec>(Vec{scale}));
  Tensor::Ptr scaled_scores = *scores * scale_tensor;

  if (mask != nullptr) {
    // Mask shape is (batchSize, 1, sequenceLength, sequenceLength) or
    // (batchSize, sequenceLength, sequenceLength)
    scaled_scores = *scaled_scores + mask;
  }

  // Apply softmax to get attention weights
  Tensor::Ptr attention_weights = scaled_scores->softmax(scaled_scores->shape().size() - 1);

  // Multiply attention weights by values: attention_output = weights . V
  // weights shape: (batchSize, numHeads, sequenceLength, sequenceLength)
  // v shape: (batchSize, numHeads, sequenceLength, head_dim)
  // Result shape: (batchSize, numHeads, sequenceLength, head_dim)
  Tensor::Ptr attention_output = attention_weights->dot(v);

  return attention_output;
}

Tensor::Ptr MultiHeadAttention::forward(
    Tensor::Ptr &query, Tensor::Ptr &key,
    Tensor::Ptr &value, Tensor::Ptr mask) {
  // query, key, value input shapes are (batchSize, sequenceLength, embedDim)

  const std::vector<int> &query_shape = query->shape();
  const std::vector<int> &key_shape = key->shape();
  const std::vector<int> &value_shape = value->shape();

  if (query_shape.size() != 3 || key_shape.size() != 3 ||
      value_shape.size() != 3 || query_shape[0] != key_shape[0] ||
      query_shape[0] != value_shape[0] || query_shape[2] != _embedDim ||
      key_shape[2] != _embedDim || value_shape[2] != _embedDim) {
    throw std::runtime_error(
        "Invalid input tensor shapes for MultiHeadAttention forward.");
  }

  size_t batchSize = query_shape[0];
  size_t query_sequence_length = query_shape[1];
  size_t key_sequence_length = key_shape[1];

  // Result shape: (batchSize, sequenceLength, embedDim) for q_proj, k_proj,
  // v_proj
  Tensor::Ptr q_proj = query_proj_.forward(query);
  Tensor::Ptr k_proj = key_proj_.forward(key);
  Tensor::Ptr v_proj = value_proj_.forward(value);

  // Reshape from (batchSize, sequenceLength, embedDim) to (batchSize,
  // sequenceLength, numHeads, head_dim)
  Tensor::Ptr qReshaped = q_proj->reshape({(int)batchSize, (int)query_sequence_length, _numHeads, head_dim_});
  Tensor::Ptr kReshaped = k_proj->reshape({(int)batchSize, (int)key_sequence_length, _numHeads, head_dim_});
  Tensor::Ptr vReshaped = v_proj->reshape({(int)batchSize, (int)key_sequence_length, _numHeads, head_dim_});

  // Transpose to get (batchSize, numHeads, sequenceLength, head_dim)
  // Permutation: [0, 2, 1, 3] for a 4D tensor
  std::vector<int> perm = {0, 2, 1, 3};
  Tensor::Ptr qSplitHeads = qReshaped->transpose(perm);
  Tensor::Ptr kSplitHeads = kReshaped->transpose(perm);
  Tensor::Ptr vSplitHeads = vReshaped->transpose(perm);

  Tensor::Ptr attentionOutputPerHead = scaled_dot_product_attention(
      qSplitHeads, kSplitHeads, vSplitHeads, mask);

  // Transpose back to (batchSize, query_sequence_length, numHeads, head_dim)
  // Inverse permutation of [0, 2, 1, 3] is [0, 2, 1, 3]
  std::vector<int> inverse_perm = {0, 2, 1, 3};
  Tensor::Ptr attention_output_transposed = attentionOutputPerHead->transpose(inverse_perm);

  // Reshape to (batchSize, query_sequence_length, embedDim)
  Tensor::Ptr attentionOutputConcat = attention_output_transposed->reshape(
          {(int)batchSize, (int)query_sequence_length, _embedDim});

  Tensor::Ptr final_output = output_proj_.forward(attentionOutputConcat);
  return final_output;
}
