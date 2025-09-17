#include "models/EncoderLayer.h"

EncoderLayer::EncoderLayer(int embedDim, int numHeads, int ffHiddenDim, float dropoutRate):
	self_attention_("encoder-self-", embedDim, numHeads),
	layernorm1_("encoder.1", embedDim),
	dropout1_(dropoutRate),
	feed_forward_(embedDim, ffHiddenDim),
	layernorm2_("encoder.2", embedDim),
	dropout2_(dropoutRate),
	dropout_rate_(dropoutRate) {
}

std::shared_ptr<Tensor>
EncoderLayer::forward(std::shared_ptr<Tensor> &input,
											std::shared_ptr<Tensor> &padding_mask, bool is_training) {
	// Input shape: (batchSize, sequence_length, embedDim)

	std::shared_ptr<Tensor> attention_output =
			self_attention_.forward(input, input, input, padding_mask);

	std::shared_ptr<Tensor> attention_output_dropped =
			dropout1_.forward(attention_output, is_training);

	// output = input + attention_output_dropped
	std::shared_ptr<Tensor> attention_residual =
			*input + attention_output_dropped;
	std::shared_ptr<Tensor> layernorm1_output =
			layernorm1_.forward(attention_residual);

	std::shared_ptr<Tensor> feed_forward_output =
			feed_forward_.forward(layernorm1_output);

	std::shared_ptr<Tensor> feed_forward_output_dropped =
			dropout2_.forward(feed_forward_output, is_training);

	// output = layernorm1_output + feed_forward_output_dropped
	std::shared_ptr<Tensor> feed_forward_residual =
			*layernorm1_output + feed_forward_output_dropped;
	std::shared_ptr<Tensor> layernorm2_output =
			layernorm2_.forward(feed_forward_residual);

	return layernorm2_output;
}