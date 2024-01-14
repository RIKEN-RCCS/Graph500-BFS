//
// Authored by ARAI Junya <araijn@gmail.com> in 2023-11-02.
//
#pragma once

#include "common.hpp"

namespace indexed_bfs {
namespace argument {
namespace detail {

struct arguments {
  // SCALE value of a Kronecker graph
  int scale;
  // Name of an implementation to run
  std::string bfs;
  // Number of BFS search keys
  size_t root_count;
};

static const arguments default_args = {
    0,                             // scale: Unused
    std::string("shared_topdown"), // bfs
    64,                            // root_count
};

template <typename T> T str_to(const char *const /* str */) {
  assert(!"Unimplemented");
  return T();
}

template <> std::string str_to<std::string>(const char *const str) {
  return std::string(str);
}

template <> int str_to<int>(const char *const str) { return std::stoi(str); }

template <> unsigned long str_to<unsigned long>(const char *const str) {
  return std::stoul(str);
}

template <typename T>
T read_arg(const char *const str, const char *const arg_name) {
  try {
    return str_to<T>(str);
  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument for " << arg_name << ": " << optarg
              << std::endl;
    common::die();
  } catch (const std::out_of_range &e) {
    std::cerr << "Out-of-range value for " << arg_name << ": " << optarg
              << std::endl;
    common::die();
  }
  return T(); // Unreachable
}

template <typename T> T read_arg(const char *const str, const char arg_key) {
  const char arg_name[3] = {'-', arg_key, '\0'};
  return read_arg<T>(str, arg_name);
}

static void die_with_usage() {
  std::cerr << "Usage: indexed_bfs [OPTION]... SCALE\n"
            << "\n"
            << "  -h      Show usage\n"
            << "  -b BFS  BFS implementation (default: " << default_args.bfs
            << ")\n"
            << "  -n NUM  Number of BFS search keys\n"
            << std::endl;

  common::die();
}

static arguments parse_args(const int argc, char *const argv[]) {
  arguments args = default_args;

  int c;
  while ((c = getopt(argc, argv, "hb:n:")) != -1) {
    switch (c) {
    case 'h':
      die_with_usage();
      break;
    case 'b':
      args.bfs = read_arg<std::string>(optarg, c);
      break;
    case 'n':
      args.root_count = read_arg<size_t>(optarg, c);
      break;
    default:
      die_with_usage();
      break;
    }
  }

  args.scale = read_arg<int>(argv[optind], "SCALE");

  return args;
}

static void print_args(const arguments &args) {
  std::cout << "arguments:" << std::endl;
  std::cout << "  scale: " << args.scale << std::endl;
  std::cout << "  bfs: " << args.bfs << std::endl;
}

} // namespace detail

using detail::arguments;
using detail::parse_args;
using detail::print_args;

} // namespace argument
} // namespace indexed_bfs
