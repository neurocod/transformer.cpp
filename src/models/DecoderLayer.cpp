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
  dropout_rate_(dropoutRate) {
}

std::shared_ptr<Tensor>
DecoderLayer::forward(std::shared_ptr<Tensor> &target_input,
                      std::shared_ptr<Tensor> &encoder_output,
                      std::shared_ptr<Tensor> &look_ahead_mask,
                      std::shared_ptr<Tensor> &padding_mask, bool is_training) {
  // target_input shape: (batchSize, target_sequence_length, embedDim)
  // encoder_output shape: (batchSize, input_sequence_length, embedDim)

  std::shared_ptr<Tensor> masked_attention_output =
      masked_self_attention_.forward(target_input, target_input, target_input,
                                     look_ahead_mask);

  std::shared_ptr<Tensor> masked_attention_output_dropped =
      dropout1_.forward(masked_attention_output, is_training);

  std::shared_ptr<Tensor> masked_attention_residual =
      *target_input + masked_attention_output_dropped;
  std::shared_ptr<Tensor> layernorm1_output =
      layernorm1_.forward(masked_attention_residual);

  std::shared_ptr<Tensor> cross_attention_output = cross_attention_.forward(
      layernorm1_output, encoder_output, encoder_output, padding_mask);

  std::shared_ptr<Tensor> cross_attention_output_dropped =
      dropout2_.forward(cross_attention_output, is_training);

  std::shared_ptr<Tensor> cross_attention_residual =
      *layernorm1_output + cross_attention_output_dropped;
  std::shared_ptr<Tensor> layernorm2_output =
      layernorm2_.forward(cross_attention_residual);

  std::shared_ptr<Tensor> feed_forward_output =
      feed_forward_.forward(layernorm2_output);

  std::shared_ptr<Tensor> feed_forward_output_dropped =
      dropout3_.forward(feed_forward_output, is_training);

  std::shared_ptr<Tensor> feed_forward_residual =
      *layernorm2_output + feed_forward_output_dropped;
  std::shared_ptr<Tensor> layernorm3_output =
      layernorm3_.forward(feed_forward_residual);

  return layernorm3_output;
}
