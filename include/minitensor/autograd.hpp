#pragma once

#include "minitensor/tensor.hpp"

#include <vector>

namespace mt {

// --- Graph nodes ------------------------------------------------------------------------

// One recorded operation in the autograd graph, stored in the grad_fn of its result.
// It keeps the operation's inputs alive and knows how to turn the gradient of the result
// into gradients of the inputs. Shared ownership points only backwards:
//     result -> node -> inputs
// A node never owns its own result: result -> node -> result would be a cycle that
// shared_ptr can never free (a memory leak on every training step).
class Node {
public:
    explicit Node(std::vector<Tensor> inputs);
    virtual ~Node() = default;

    // Chain rule for this operation: given grad_output = d loss / d result, returns
    // d loss / d input for every input, in the order of inputs(), each with that input's shape.
    virtual std::vector<Tensor> backward(const Tensor& grad_output) const = 0;

    // Name for error messages and debugging, e.g. "MulBackward".
    virtual const char* name() const = 0;

    // The tensors the operation was called with (also the way from the result to the leaves).
    const std::vector<Tensor>& inputs() const;

private:
    std::vector<Tensor> inputs_;
};

// --- Recording on/off -------------------------------------------------------------------

// True unless a NoGradGuard is alive in this thread.
bool is_grad_enabled();

// While it exists, operations do not record nodes (like `with torch.no_grad():` in PyTorch).
// backward() runs under one, so the gradient formulas do not build a graph of their own.
// The destructor restores the previous state, so guards can be nested.
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;  // a copy would restore the state a second time
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

}  // namespace mt
