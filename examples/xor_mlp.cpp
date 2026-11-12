// XOR learned by a tiny neural network built on mini-tensor:
//     2 inputs -> 8 hidden units (ReLU) -> 1 output, mean squared error,
//     plain gradient descent on all four samples at once (full batch).
// No single straight line separates the XOR classes, so a linear model cannot learn it;
// the hidden layer with a nonlinearity is what makes it possible.
//
//     ./build/examples/xor_mlp [loss.csv]
//
// Writes the loss of every epoch to the CSV file (default: loss.csv in the current directory)
// and exits with 1 if the network did not learn XOR, so ctest runs it as an end-to-end test.

#include "minitensor/autograd.hpp"
#include "minitensor/ops.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::int64_t kHidden = 8;
constexpr float kLearningRate = 0.1f;
constexpr int kEpochs = 500;  // with these settings every seed 0..19 is below 0.01 by epoch 160
constexpr std::uint64_t kSeed = 0;

// He (Kaiming) initialization for a layer followed by ReLU: w ~ N(0, 2 / fan_in).
// It keeps the variance of the activations roughly the same from layer to layer.
mt::Tensor he_init(std::int64_t fan_in, std::int64_t fan_out, std::uint64_t seed) {
    const mt::Tensor scale = mt::from_data({std::sqrt(2.0f / static_cast<float>(fan_in))}, {});
    return (mt::randn({fan_in, fan_out}, seed) * scale).set_requires_grad(true);
}

struct Mlp {
    // Different seeds: with the same seed both layers would start from the same random numbers.
    mt::Tensor w1 = he_init(2, kHidden, kSeed);
    mt::Tensor b1 = mt::zeros({kHidden}).set_requires_grad(true);
    mt::Tensor w2 = he_init(kHidden, 1, kSeed + 1);
    mt::Tensor b2 = mt::zeros({1}).set_requires_grad(true);

    // x [n, 2] -> prediction [n, 1]. The biases are broadcast over the n samples.
    mt::Tensor forward(const mt::Tensor& x) const {
        const mt::Tensor h = mt::relu(mt::matmul(x, w1) + b1);  // [n, 8]
        return mt::matmul(h, w2) + b2;                          // [n, 1]
    }

    // Handles to the same tensors: changing them in place changes the model.
    std::vector<mt::Tensor> parameters() const { return {w1, b1, w2, b2}; }
};

mt::Tensor mse(const mt::Tensor& prediction, const mt::Tensor& target) {
    const mt::Tensor diff = prediction - target;
    return mt::mean(diff * diff);
}

// One step of gradient descent: p -= lr * d loss / dp for every parameter. Under NoGradGuard,
// because the update is not part of the model (and sub_ refuses to run while recording).
void sgd_step(std::vector<mt::Tensor>& params, const mt::Tensor& lr) {
    mt::NoGradGuard no_grad;
    for (mt::Tensor& p : params) p.sub_(lr * p.grad().value());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string csv_path = argc > 1 ? argv[1] : "loss.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "cannot write " << csv_path << '\n';
        return EXIT_FAILURE;
    }
    csv << "epoch,loss\n";

    // The four XOR samples: inputs [4, 2], targets [4, 1].
    const mt::Tensor x = mt::from_data({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2});
    const mt::Tensor y = mt::from_data({0, 1, 1, 0}, {4, 1});
    const mt::Tensor lr = mt::from_data({kLearningRate}, {});

    Mlp model;
    std::vector<mt::Tensor> params = model.parameters();

    float last_loss = 0.0f;
    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        for (mt::Tensor& p : params) p.zero_grad();      // gradients would add up across steps
        const mt::Tensor loss = mse(model.forward(x), y);  // forward: builds the graph
        loss.backward();                                    // backward: fills grad() of the parameters
        sgd_step(params, lr);                               // update, not recorded

        last_loss = loss.at({});
        csv << epoch << ',' << last_loss << '\n';
        if (epoch % 50 == 0) std::cout << "epoch " << std::setw(3) << epoch << "  loss " << last_loss << '\n';
    }
    std::cout << "epoch " << std::setw(3) << kEpochs - 1 << "  loss " << last_loss << "\n\n";

    // Predictions with the trained weights; no graph needed (like torch.no_grad() for evaluation).
    mt::NoGradGuard no_grad;
    const mt::Tensor prediction = model.forward(x);
    bool learned = last_loss < 0.01f;
    std::cout << std::fixed << std::setprecision(3);
    for (std::int64_t i = 0; i < 4; ++i) {
        const float p = prediction.at({i, 0});
        const float target = y.at({i, 0});
        learned = learned && std::fabs(p - target) < 0.5f;  // on the right side of 0.5
        std::cout << "x = (" << static_cast<int>(x.at({i, 0})) << ", " << static_cast<int>(x.at({i, 1}))
                  << ")  target " << static_cast<int>(target) << "  prediction " << p << '\n';
    }
    std::cout << (learned ? "\nlearned XOR\n" : "\ndid NOT learn XOR\n");
    return learned ? EXIT_SUCCESS : EXIT_FAILURE;
}
