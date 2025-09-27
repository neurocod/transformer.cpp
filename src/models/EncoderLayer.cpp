#include "models/EncoderLayer.h"

EncoderLayer::EncoderLayer(int embedDim, int numHeads, int ffHiddenDim, float dropoutRate):
	self_attention_("encoder-self-", embedDim, numHeads),
	layernorm1_("encoder.1", embedDim),
	dropout1_(dropoutRate),
	feed_forward_(embedDim, ffHiddenDim),
	layernorm2_("encoder.2", embedDim),
	dropout2_(dropoutRate),
	_dropoutRate(dropoutRate) {
}

Tensor::Ptr EncoderLayer::forward(Tensor::Ptr &input,
											Tensor::Ptr &padding_mask, bool isTraining) {
	// Input shape: (batchSize, sequenceLength, embedDim)

	Tensor::Ptr attention_output = self_attention_.forward(input, input, input, padding_mask);

	Tensor::Ptr attention_output_dropped = dropout1_.forward(attention_output, isTraining);

	// output = input + attention_output_dropped
	Tensor::Ptr attention_residual = *input + attention_output_dropped;
	Tensor::Ptr layernorm1_output = layernorm1_.forward(attention_residual);

	Tensor::Ptr feed_forward_output = feed_forward_.forward(layernorm1_output);

	Tensor::Ptr feed_forward_output_dropped = dropout2_.forward(feed_forward_output, isTraining);

	// output = layernorm1_output + feed_forward_output_dropped
	Tensor::Ptr feed_forward_residual = *layernorm1_output + feed_forward_output_dropped;
	Tensor::Ptr layernorm2_output = layernorm2_.forward(feed_forward_residual);

	return layernorm2_output;
}