#include "reference.hpp"

#include "gradcheck.hpp"  // unravel

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef MT_TEST_DATA_DIR
#error "MT_TEST_DATA_DIR must be set by CMake (absolute path to tests/data)"
#endif

namespace mt_test {

namespace {

// "[2, 3]", "{1, 0}": shapes and indices in failure messages.
std::string join(const std::vector<std::int64_t>& numbers, char open, char close) {
    std::ostringstream out;
    out << open;
    for (std::size_t i = 0; i < numbers.size(); ++i) out << (i > 0 ? ", " : "") << numbers[i];
    out << close;
    return out.str();
}

}  // namespace

Reference load_reference(const std::string& name) {
    const std::string path = std::string(MT_TEST_DATA_DIR) + "/" + name + ".txt";
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path + " (run scripts/gen_reference.py)");

    Reference ref;
    std::string header;
    while (std::getline(file, header)) {
        if (header.empty() || header[0] == '#') continue;  // comments

        // Header line: name and shape, e.g. "grad_a 3 1" (just the name for a scalar).
        std::istringstream head(header);
        std::string tensor_name;
        head >> tensor_name;
        mt::Shape shape;
        std::int64_t size = 0;
        while (head >> size) shape.push_back(size);
        if (!head.eof()) throw std::runtime_error(path + ": bad header line \"" + header + "\"");

        // Next line: all values in row-major order.
        std::string line;
        if (!std::getline(file, line)) throw std::runtime_error(path + ": no values for " + tensor_name);
        std::istringstream in(line);
        std::vector<float> values;
        float v = 0.0f;
        while (in >> v) values.push_back(v);  // decimal -> nearest float, so 9 digits come back exact
        if (!in.eof() || static_cast<std::int64_t>(values.size()) != mt::numel(shape)) {
            throw std::runtime_error(path + ": " + tensor_name + " has shape " + join(shape, '[', ']') +
                                     " but the values line does not hold that many numbers");
        }
        if (!ref.emplace(tensor_name, mt::from_data(std::move(values), shape)).second) {
            throw std::runtime_error(path + ": " + tensor_name + " appears twice");
        }
    }
    return ref;
}

::testing::AssertionResult allclose(const mt::Tensor& actual, const mt::Tensor& expected, float atol, float rtol) {
    if (actual.shape() != expected.shape()) {
        return ::testing::AssertionFailure() << "shape " << join(actual.shape(), '[', ']') << ", expected "
                                             << join(expected.shape(), '[', ']');
    }
    const std::vector<float> a = actual.to_vector();  // logical order, so any layout works
    const std::vector<float> e = expected.to_vector();

    std::int64_t failures = 0;
    std::size_t worst = 0;
    float worst_excess = 0.0f;  // how far the worst element is beyond its tolerance
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - e[i]);
        const float tolerance = atol + rtol * std::fabs(e[i]);
        if (!(diff <= tolerance)) {  // written this way so that NaN fails too
            if (failures == 0 || diff - tolerance > worst_excess) {
                worst = i;
                worst_excess = diff - tolerance;
            }
            ++failures;
        }
    }
    if (failures == 0) return ::testing::AssertionSuccess();

    const mt::Index index = unravel(static_cast<std::int64_t>(worst), actual.shape());
    return ::testing::AssertionFailure() << failures << " of " << a.size() << " elements differ; worst at "
                                         << join(index, '{', '}') << ": got " << a[worst] << ", expected "
                                         << e[worst] << " (|diff| " << std::fabs(a[worst] - e[worst])
                                         << " > allowed " << atol + rtol * std::fabs(e[worst]) << ")";
}

}  // namespace mt_test
