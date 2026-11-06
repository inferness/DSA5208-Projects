#pragma once

#include <cstdlib>
#include <string>

inline const char *results_directory() {
  const char *dir = std::getenv("RESULTS_DIR");
  return (dir != nullptr && *dir != '\0') ? dir : "results";
}

inline std::string result_path(const std::string &name) {
  return std::string(results_directory()) + "/" + name;
}
