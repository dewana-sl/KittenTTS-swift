#pragma once
#include <string>
#include <vector>
#include <memory>
#include "layer.hpp"

// Reads a W1X1 binary weights file and calls load_weights() on each layer
// in the list, matched by the name tag stored in the binary index.
// Throws std::runtime_error on file or format errors.
void load_weights(const std::string& weights_path,
                  std::vector<std::unique_ptr<ILayer>>& layers);
