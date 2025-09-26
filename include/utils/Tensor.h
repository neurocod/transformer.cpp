#pragma once
class BinaryWriter;
class BinaryReader;

enum class OperationType {
  None,
  Add,
  Sub,
  Mul,
  Dot,
  Transpose,
  Reshape,
  Sum,
  Div,
  ReLU,
  GELU,
  Sigmoid,
  Tanh,
  LogSoftmax,
  NegativeLogLikelihood,
  LayerNorm,
  Softmax,
  Dropout,
  EmbeddingLookup
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  using Ptr = std::shared_ptr<Tensor>;
  using Vec = std::vector<float>;
  Tensor();
  // if name is provided, it means _isOptimizable == true
  Tensor(const std::vector<int>& shape, const std::string& name = {});
  Tensor(const std::vector<int> &shape, const std::shared_ptr<Vec> &data, const std::string& name = {});
  ~Tensor() {}
  static Tensor::Ptr create();
  static Tensor::Ptr create(const std::vector<int> &shape, const std::string &name = {});
  static Tensor::Ptr create(const std::vector<int> &shape, const std::shared_ptr<Vec> &data,
                            const std::string &name = {});

  std::shared_ptr<Tensor> sharedPtr() { return shared_from_this(); }
  void write(BinaryWriter& writer) const;
  bool read(BinaryReader& reader);
  std::string debugString() const;
  static std::string shapeString(const std::vector<int>& shape);
  const char *dbg() const; // in Visual Studio Immediate Window, write: t.dbg(), sb

  // Intermediate values for different functions
  // Backward permutation
  std::vector<int> forward_permutation_;
  // Backward reshape
  std::vector<int> original_shape_before_reshape_;
  // Layernorm
  Tensor::Ptr layernorm_gamma_;
  Tensor::Ptr layernorm_beta_;
  Tensor::Ptr layernorm_mean_;
  Tensor::Ptr layernorm_inv_stddev_;
  Tensor::Ptr layernorm_centered_input_;
  float layernorm_epsilon_;
  // Dropout
  Tensor::Ptr dropout_mask_;
  float dropout_scale_;
  // Embedding
  Tensor::Ptr embedding_indices_;

  // Getters for shape, data, and gradient
  const std::vector<int> &shape() const { return _shape; };
  const Vec &data() const { return *_data; };
  const Vec &grad() const { return *_grad; };
  Vec &dataRef() { return *_data; }
  Vec &gradRef() { return *_grad; }

  void set_data(const std::shared_ptr<Vec> &data);
  void set_parents(const std::vector<Tensor::Ptr> &parents) {
    parents_ = parents;
  }
  void set_creator_op(OperationType op) { creator_op_ = op; }

  static std::vector<Tensor::Ptr> &get_optimizable_tensors() {
    return optimizable_tensors_;
  }
  // Get element by multi-dimensional index
  float get(const std::vector<int> &indices) const;
  // Set element by multi-dimensional index
  void set(const std::vector<int> &indices, float value);
  bool is_optimizable() const { return _isOptimizable; }

  // Basic tensor operations
  Tensor::Ptr operator+(const Tensor::Ptr &other) const;
  Tensor::Ptr operator-(const Tensor::Ptr &other) const;
  Tensor::Ptr operator*(const Tensor::Ptr &other) const;
  Tensor::Ptr operator/(const Tensor::Ptr &other) const;

  Tensor::Ptr dot(const Tensor::Ptr &other) const;
  Tensor::Ptr sum() const;

  Tensor::Ptr transpose(const std::vector<int> &permutation) const;
  Tensor::Ptr reshape(const std::vector<int> &new_shape) const;

  Tensor::Ptr softmax(int dim = -1) const;

  // Gradient handling
  void zeroGrad();
  void backward(const Tensor::Ptr &gradOutput);

  // Helper to calculate total number of elements from shape
  size_t num_elements() const;

  const std::vector<Tensor::Ptr> &get_parents() const {
    return parents_;
  }

private:
  static std::vector<Tensor::Ptr> optimizable_tensors_;

  std::vector<int> _shape;
  std::shared_ptr<Vec> _data;
  std::shared_ptr<Vec> _grad;
  // allows to track the computation graph, enabling correct gradient calculation and backpropagation
  std::vector<Tensor::Ptr> parents_;
  const std::string _name;
  const bool _isOptimizable = false;

  OperationType creator_op_ = OperationType::None;

  size_t get_linear_index(const std::vector<int> &indices) const;
  std::vector<size_t> calculate_strides(const std::vector<int> &shape) const;

  static std::vector<int>
  calculate_broadcast_shape(const std::vector<int> &shape1,
                            const std::vector<int> &shape2);
  Tensor::Ptr broadcast_to(const std::vector<int> &new_shape) const;
  static bool is_broadcastable(const std::vector<int> &shape1,
                               const std::vector<int> &shape2);
  bool are_shapes_compatible(const Tensor &other) const {
    return _shape == other._shape;
  }

  void reduce_gradient(const Tensor::Ptr &gradOutput,
                       Tensor::Ptr &parent_grad,
                       const std::vector<int> &parent_shape);
  void backward_add(const Tensor::Ptr &gradOutput);
  void backward_sub(const Tensor::Ptr &gradOutput);
  void backward_mul(const Tensor::Ptr &gradOutput);
  void backward_div(const Tensor::Ptr &gradOutput);

  void backward_transpose(const Tensor::Ptr &gradOutput);
  void backward_reshape(const Tensor::Ptr &gradOutput);

  void backward_dot(const Tensor::Ptr &gradOutput);
  void backward_sum(const Tensor::Ptr &gradOutput);

  void backward_relu(const Tensor::Ptr &gradOutput);
  void backward_gelu(const Tensor::Ptr &gradOutput);
  void backward_sigmoid(const Tensor::Ptr &gradOutput);
  void backward_tanh(const Tensor::Ptr &gradOutput);
  void backward_logsoftmax(const Tensor::Ptr &gradOutput);
  void backward_nllloss(const Tensor::Ptr &gradOutput);
  void backward_layernorm(const Tensor::Ptr &gradOutput);
  void backward_softmax(const Tensor::Ptr &gradOutput);
  void backward_dropout(const Tensor::Ptr &gradOutput);
  void backward_embedding_lookup(const Tensor::Ptr &gradOutput);
};
#ifdef _DEBUG
// in Visual Studio Immediate Window, write: dbg(anyTensorVariable), sb
const char *dbg(const Tensor &t);
const char *dbg(const Tensor *t);
const char *dbg(const Tensor::Ptr &t);
#endif // _DEBUG