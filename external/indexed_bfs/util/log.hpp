//
// This logger offers the following benefits compared to `std::cerr`:
// - A printed line is (probably) not interrupted by outputs from other threads
//   or processes
// - Additional information such as a timestamp and code location are
//   automatically inserted
// - Formatting by the fmt library is supported if fmt is included before
//   the include of this file.
//
// Authored by ARAI Junya <araijn@gmail.com> on 2023-12-22
//
#pragma once

#include "macro.hpp"
#include <chrono>
#include <functional>
#include <sstream>

#define INDEXED_BFS_LOG(severity)                                              \
  if (severity < ::indexed_bfs::util::log::config.min_severity) {              \
  } else                                                                       \
    ::indexed_bfs::util::log::detail::entry(severity, __FILE__,                \
                                            __PRETTY_FUNCTION__, __LINE__)

#define LOG_T INDEXED_BFS_LOG(::indexed_bfs::util::log::severity::trace)
#define LOG_D INDEXED_BFS_LOG(::indexed_bfs::util::log::severity::debug)
#define LOG_I INDEXED_BFS_LOG(::indexed_bfs::util::log::severity::info)
#define LOG_W INDEXED_BFS_LOG(::indexed_bfs::util::log::severity::warning)
#define LOG_E INDEXED_BFS_LOG(::indexed_bfs::util::log::severity::error)

namespace indexed_bfs {
namespace util {
namespace log {
namespace detail {

// A greater level (in int) is more severe.
enum class severity {
  trace,
  debug,
  info,
  warning,
  error,
};

struct record {
  detail::severity severity;
  const char *file;
  const char *function;
  int line;
  const char *message;
};

constexpr const char *severity_string(const severity s) {
  constexpr const char *strs[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"};
  return strs[static_cast<int>(s)];
}

static int milliseconds_of(const std::chrono::system_clock::time_point &p) {
  using namespace std::chrono;
  const auto ms = duration_cast<milliseconds>(p.time_since_epoch());
  return static_cast<int>((ms % 1000).count());
}

static void default_printer(const record &r) {
  const auto now = std::chrono::system_clock::now();
  const time_t time = std::chrono::system_clock::to_time_t(now);
  const std::tm *tm = std::localtime(&time);
  const char *const sev = severity_string(r.severity);
  fprintf(stderr, "%04d-%02d-%02d %02d:%02d:%02d.%03d %-5s [%s@%d] %s\n",
          tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday, tm->tm_hour,
          tm->tm_min, tm->tm_sec, milliseconds_of(now), sev, r.function, r.line,
          r.message);
}

//
// Configuration
//
static struct {
  // Logs with a severity less than this are not printed.
  detail::severity min_severity;
  // Function to print a log record
  std::function<void(const record &)> printer;
} config = {
    severity::warning,
    default_printer,
};

struct entry {
  using ostream_type = std::basic_ostream<std::ostringstream::char_type,
                                          std::ostringstream::traits_type>;

  detail::severity severity;
  const char *file;
  const char *func;
  const int line;
  std::ostringstream message;

  entry(const detail::severity _severity, const char *const _file,
        const char *const _func, const int _line)
      : severity(_severity), file(_file), func(_func), line(_line), message() {}

  ~entry() {
    const std::string f = macro::pretty_function_name_with_namespace(func);
    config.printer({severity, file, f.c_str(), line, message.str().c_str()});
  }

  entry(entry &&x) = default;
  entry(const entry &x) = delete;
  entry &operator=(entry &&x) = default;
  entry &operator=(const entry &x) = delete;

  template <typename T> entry &operator<<(const T &x) {
    message << x;
    return *this;
  }

  // Overload for `std::endl`
  entry &operator<<(ostream_type &(*x)(ostream_type &os)) {
    message << x;
    return *this;
  }

  // Provide fmt-style formatting if fmt is included.
#ifdef FMT_VERSION
  template <typename... Args>
  void operator()(fmt::string_view format_str, const Args &...args) {
    fmt::vformat_to(std::ostream_iterator<char>(message), format_str,
                    fmt::make_format_args(args...));
  }
#endif
};

} // namespace detail

using detail::config;
using detail::record;
using detail::severity;
using detail::severity_string;

} // namespace log
} // namespace util
} // namespace indexed_bfs
