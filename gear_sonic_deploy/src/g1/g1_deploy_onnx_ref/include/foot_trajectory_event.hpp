#ifndef FOOT_TRAJECTORY_EVENT_HPP
#define FOOT_TRAJECTORY_EVENT_HPP

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace FootTrajectoryEvent {

inline std::string EventFilePath() {
  const char* env_path = std::getenv("G1_DANCE_TRAJECTORY_EVENT_FILE");
  if (env_path != nullptr && env_path[0] != '\0') {
    return std::string(env_path);
  }
  return "/tmp/g1_dance_trajectory_event.json";
}

inline std::string JsonEscape(const std::string& input) {
  std::ostringstream escaped;
  for (char ch : input) {
    switch (ch) {
      case '"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\b': escaped << "\\b"; break;
      case '\f': escaped << "\\f"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default: escaped << ch; break;
    }
  }
  return escaped.str();
}

inline long long NowMilliseconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

inline bool WriteEvent(
    const std::string& event,
    const std::string& motion_name,
    int motion_index,
    int frame,
    int timesteps) {
  static unsigned long long sequence = 0;
  sequence += 1;

  const std::filesystem::path event_path(EventFilePath());
  std::error_code ec;
  if (!event_path.parent_path().empty()) {
    std::filesystem::create_directories(event_path.parent_path(), ec);
  }

  const auto timestamp_ms = NowMilliseconds();
  const auto tmp_path = event_path.string() + ".tmp";
  std::ofstream file(tmp_path, std::ios::trunc);
  if (!file.is_open()) {
    std::cerr << "[FootTrajectoryEvent] Failed to open " << tmp_path << std::endl;
    return false;
  }

  file << "{\n"
       << "  \"event\": \"" << JsonEscape(event) << "\",\n"
       << "  \"motion_name\": \"" << JsonEscape(motion_name) << "\",\n"
       << "  \"motion_index\": " << motion_index << ",\n"
       << "  \"frame\": " << frame << ",\n"
       << "  \"timesteps\": " << timesteps << ",\n"
       << "  \"timestamp_ms\": " << timestamp_ms << ",\n"
       << "  \"sequence\": " << sequence << "\n"
       << "}\n";
  file.close();

  std::filesystem::rename(tmp_path, event_path, ec);
  if (ec) {
    std::cerr << "[FootTrajectoryEvent] Failed to rename " << tmp_path
              << " to " << event_path << ": " << ec.message() << std::endl;
    return false;
  }

  return true;
}

}  // namespace FootTrajectoryEvent

#endif  // FOOT_TRAJECTORY_EVENT_HPP
