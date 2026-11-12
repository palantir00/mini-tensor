#include "minitensor/autograd.hpp"

#include "minitensor/ops.hpp"

#include "internal.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mt {

// --- Node -------------------------------------------------------------------------------

Node::Node(std::vector<Tensor> inputs) : inputs_(std::move(inputs)) {}

const std::vector<Tensor>& Node::inputs() const { return inputs_; }

// --- Recording on/off -------------------------------------------------------------------

namespace {

// One flag per thread, as in PyTorch: a guard in one thread does not switch off the others.
thread_local bool grad_enabled = true;

// Depth-first search from `node` towards the leaves. A node is appended only after the nodes
// that produced its inputs, so `order` lists every operation after everything it depends on.
void topo_sort(const Node* node, std::unordered_set<const Node*>& visited, std::vector<const Node*>& order) {
    if (!visited.insert(node).second) return;  // already reached through another path
    for (const Tensor& input : node->inputs()) {
        if (!input.is_leaf()) topo_sort(input.grad_fn().get(), visited, order);
    }
    order.push_back(node);
}

}  // namespace

bool is_grad_enabled() { return grad_enabled; }

NoGradGuard::NoGradGuard() : previous_(grad_enabled) { grad_enabled = false; }

NoGradGuard::~NoGradGuard() { grad_enabled = previous_; }

bool detail::should_record(const std::vector<Tensor>& inputs) {
    if (!grad_enabled) return false;
    for (const Tensor& t : inputs) {
        if (t.requires_grad()) return true;
    }
    return false;
}

// --- Engine -----------------------------------------------------------------------------
// Members of Tensor, defined here next to the rest of autograd.

void Tensor::backward() const {
    if (numel() != 1) {
        throw std::invalid_argument("backward: a tensor of shape " + detail::shape_to_string(shape()) +
                                    " needs an explicit gradient; without one it must have exactly one element");
    }
    backward(ones(shape()));  // d this / d this = 1
}

void Tensor::backward(const Tensor& grad) const {
    if (!requires_grad()) {
        throw std::logic_error("backward: this tensor does not require grad "
                               "(no tensor with requires_grad was used to compute it)");
    }
    if (grad.shape() != shape()) {
        throw std::invalid_argument("backward: gradient of shape " + detail::shape_to_string(grad.shape()) +
                                    " does not match tensor of shape " + detail::shape_to_string(shape()));
    }
    NoGradGuard no_grad;  // the gradient formulas must not record a graph of their own

    if (is_leaf()) {  // nothing to propagate: the gradient goes straight into grad()
        accumulate_grad(grad);
        return;
    }

    // 1. Order. Walking `order` from the back, every operation comes before the operations that
    //    produced its inputs, i.e. a node runs only after every node that used its result.
    std::vector<const Node*> order;
    std::unordered_set<const Node*> visited;
    topo_sort(impl_->grad_fn.get(), visited, order);

    // 2. Gradient of each node's result, summed over all operations that used that result.
    //    Keyed by node: a non-leaf tensor has exactly one grad_fn, and a node exactly one result.
    std::unordered_map<const Node*, Tensor> pending;
    pending.emplace(impl_->grad_fn.get(), grad);

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const Node* node = *it;
        const Tensor grad_output = pending.at(node);
        pending.erase(node);  // not needed any more, free it early

        const std::vector<Tensor> grads = node->backward(grad_output);  // chain rule, one step
        const std::vector<Tensor>& inputs = node->inputs();
        if (grads.size() != inputs.size()) {
            throw std::logic_error(std::string(node->name()) + ": returned " + std::to_string(grads.size()) +
                                   " gradients for " + std::to_string(inputs.size()) + " inputs");
        }

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const Tensor& input = inputs[i];
            if (!input.requires_grad()) continue;  // e.g. the data x of a network: nobody wants dx
            if (grads[i].shape() != input.shape()) {
                throw std::logic_error(std::string(node->name()) + ": gradient of shape " +
                                       detail::shape_to_string(grads[i].shape()) + " for input " +
                                       std::to_string(i) + " of shape " + detail::shape_to_string(input.shape()));
            }
            if (input.is_leaf()) {
                input.accumulate_grad(grads[i]);  // final stop: the leaf's grad()
            } else {
                // A result used by several operations gets the sum of their contributions (+=).
                auto [entry, inserted] = pending.try_emplace(input.grad_fn().get(), grads[i]);
                if (!inserted) entry->second = add(entry->second, grads[i]);
            }
        }
    }
}

void Tensor::accumulate_grad(const Tensor& g) const {
    if (!impl_->grad) {
        // First gradient: always an own contiguous copy. g itself may be shared with other
        // gradients, or be an expand view with stride 0, where writing in place would be wrong.
        impl_->grad = from_data(g.to_vector(), g.shape()).impl_;
        return;
    }
    // Later gradients are added in place (the buffer is ours), so grad() handles taken
    // before still see the current value, as in PyTorch.
    std::vector<float>& dst = impl_->grad->storage->data;
    const std::vector<float> src = g.to_vector();
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
}

}  // namespace mt
