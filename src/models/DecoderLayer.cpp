#include "models/DecoderLayer.h"

DecoderLayer::DecoderLayer(int embedDim, int numHeads, int ffHiddenDim,
	float dropoutRate) :
	masked_self_attention_("decoder-masked-", embedDim, numHeads),
  layernorm1_("Decoder.1", embedDim),
	dropout1_(dropoutRate),
  cross_attention_("Decoder-cross-", embedDim, numHeads),
	layernorm2_("Decoder.2", embedDim),
  dropout2_(dropoutRate),
	feed_forward_(embedDim, ffHiddenDim),
  layernorm3_("Decoder.3", embedDim),
	dropout3_(dropoutRate),
  _dropoutRate(dropoutRate) {
}

Tensor::Ptr DecoderLayer::forward(Tensor::Ptr &target_input,
                      Tensor::Ptr &encoder_output,
                      Tensor::Ptr &look_ahead_mask,
                      Tensor::Ptr &padding_mask, bool isTraining) {
  // target_input shape: (batchSize, target_sequence_length, embedDim)
  // encoder_output shape: (batchSize, input_sequence_length, embedDim)

  Tensor::Ptr masked_attention_output = masked_self_attention_.forward(
	  target_input, target_input, target_input, look_ahead_mask);

  Tensor::Ptr masked_attention_output_dropped = dropout1_.forward(masked_attention_output, isTraining);

  Tensor::Ptr masked_attention_residual = *target_input + masked_attention_output_dropped;
  Tensor::Ptr layernorm1_output = layernorm1_.forward(masked_attention_residual);

  Tensor::Ptr cross_attention_output = cross_attention_.forward(
	  layernorm1_output, encoder_output, encoder_output, padding_mask);

  Tensor::Ptr cross_attention_output_dropped = dropout2_.forward(cross_attention_output, isTraining);

  Tensor::Ptr cross_attention_residual = *layernorm1_output + cross_attention_output_dropped;
  Tensor::Ptr layernorm2_output = layernorm2_.forward(cross_attention_residual);

  Tensor::Ptr feed_forward_output = feed_forward_.forward(layernorm2_output);

  Tensor::Ptr feed_forward_output_dropped = dropout3_.forward(feed_forward_output, isTraining);

  Tensor::Ptr feed_forward_residual = *layernorm2_output + feed_forward_output_dropped;
  Tensor::Ptr layernorm3_output = layernorm3_.forward(feed_forward_residual);

  return layernorm3_output;
}
