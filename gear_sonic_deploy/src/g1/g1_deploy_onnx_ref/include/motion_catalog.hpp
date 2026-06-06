/**
 * @file motion_catalog.hpp
 * @brief YAML-backed catalog for externally selectable reference motions.
 *
 * The catalog is an operator-facing index over motion directories.  Motion CSV
 * loading remains in MotionDataReader; this file only resolves stable command
 * names, aliases, and paths into the indices loaded by MotionDataReader.
 */

#ifndef MOTION_CATALOG_HPP
#define MOTION_CATALOG_HPP

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "motion_data_reader.hpp"

struct MotionCatalogEntry {
  std::string id;
  std::string display_name;
  std::string raw_path;
  std::string resolved_path;
  bool enabled = true;
  std::vector<std::string> aliases;
  int motion_index = -1;
};

class MotionCatalog {
 public:
  bool LoadFromYaml(const std::string& catalog_path) {
    this->entries_.clear();
    this->lookup_.clear();
    this->catalog_path_ = catalog_path;

    YAML::Node root;
    try {
      root = YAML::LoadFile(catalog_path);
    } catch (const std::exception& e) {
      std::cerr << "[MotionCatalog] Failed to load " << catalog_path << ": " << e.what() << std::endl;
      return false;
    }

    if (!root["motions"] || !root["motions"].IsSequence()) {
      std::cerr << "[MotionCatalog] Missing required 'motions' sequence in " << catalog_path << std::endl;
      return false;
    }

    std::set<std::string> seen_keys;
    std::filesystem::path catalog_dir = std::filesystem::path(catalog_path).parent_path();
    for (const auto& node : root["motions"]) {
      MotionCatalogEntry entry;
      entry.id = this->ReadRequiredString(node, "id");
      entry.display_name = this->ReadOptionalString(node, "display_name", entry.id);
      entry.raw_path = this->ReadRequiredString(node, "path");
      entry.enabled = this->ReadOptionalBool(node, "enabled", true);

      if (entry.id.empty() || entry.raw_path.empty()) {
        std::cerr << "[MotionCatalog] Skipping entry with empty id/path" << std::endl;
        continue;
      }

      if (node["aliases"] && node["aliases"].IsSequence()) {
        for (const auto& alias_node : node["aliases"]) {
          std::string alias = alias_node.as<std::string>("");
          if (!alias.empty()) { entry.aliases.push_back(alias); }
        }
      }

      if (!entry.enabled) {
        this->entries_.push_back(entry);
        continue;
      }

      std::filesystem::path motion_path(entry.raw_path);
      if (!motion_path.is_absolute()) { motion_path = catalog_dir / motion_path; }
      std::error_code path_error;
      std::filesystem::path resolved_path = std::filesystem::weakly_canonical(motion_path, path_error);
      if (path_error) {
        std::cerr << "[MotionCatalog] Invalid motion path for " << entry.id << ": " << motion_path
                  << " (" << path_error.message() << ")" << std::endl;
        return false;
      }
      entry.resolved_path = resolved_path.string();

      std::vector<std::string> keys = this->LookupKeysForEntry(entry);
      bool duplicate = false;
      for (const auto& key : keys) {
        if (seen_keys.count(key) > 0) {
          std::cerr << "[MotionCatalog] Duplicate id/alias/display key: " << key << std::endl;
          duplicate = true;
        }
      }
      if (duplicate) { return false; }
      seen_keys.insert(keys.begin(), keys.end());
      this->entries_.push_back(entry);
    }

    std::cout << "[MotionCatalog] Loaded " << this->entries_.size() << " catalog entries from " << catalog_path << std::endl;
    return !this->entries_.empty();
  }

  void BuildFromLoadedMotions(const std::vector<std::shared_ptr<MotionSequence>>& motions) {
    this->entries_.clear();
    this->lookup_.clear();
    this->catalog_path_.clear();
    for (size_t i = 0; i < motions.size(); ++i) {
      if (!motions[i]) { continue; }
      MotionCatalogEntry entry;
      entry.id = motions[i]->name;
      entry.display_name = motions[i]->name;
      entry.raw_path = motions[i]->name;
      entry.resolved_path = motions[i]->name;
      entry.enabled = true;
      entry.motion_index = static_cast<int>(i);
      this->entries_.push_back(entry);
    }
    this->RebuildLookup();
  }

  std::vector<MotionCatalogEntry>& Entries() { return this->entries_; }
  const std::vector<MotionCatalogEntry>& Entries() const { return this->entries_; }

  void BindMotionIndex(size_t catalog_index, int motion_index) {
    if (catalog_index >= this->entries_.size()) { return; }
    this->entries_[catalog_index].motion_index = motion_index;
    this->RebuildLookup();
  }

  bool Resolve(const std::string& request, int& motion_index, MotionCatalogEntry& entry) const {
    std::string normalized = this->NormalizeKey(request);
    if (normalized.empty()) { return false; }

    if (this->IsInteger(normalized)) {
      try {
        int catalog_index = std::stoi(normalized);
        if (catalog_index >= 0 && catalog_index < static_cast<int>(this->entries_.size())) {
          const auto& candidate = this->entries_[static_cast<size_t>(catalog_index)];
          if (candidate.enabled && candidate.motion_index >= 0) {
            motion_index = candidate.motion_index;
            entry = candidate;
            return true;
          }
        }
      } catch (const std::exception&) {
        return false;
      }
    }

    auto it = this->lookup_.find(normalized);
    if (it == this->lookup_.end()) { return false; }
    const auto& candidate = this->entries_[it->second];
    if (!candidate.enabled || candidate.motion_index < 0) { return false; }
    motion_index = candidate.motion_index;
    entry = candidate;
    return true;
  }

  std::string ToJson() const {
    std::ostringstream out;
    out << "{\"motions\":[";
    bool first = true;
    for (size_t i = 0; i < this->entries_.size(); ++i) {
      const auto& entry = this->entries_[i];
      if (!entry.enabled || entry.motion_index < 0) { continue; }
      if (!first) { out << ","; }
      first = false;
      out << "{";
      out << "\"index\":" << i << ",";
      out << "\"motion_index\":" << entry.motion_index << ",";
      out << "\"id\":\"" << this->JsonEscape(entry.id) << "\",";
      out << "\"display_name\":\"" << this->JsonEscape(entry.display_name) << "\",";
      out << "\"path\":\"" << this->JsonEscape(entry.resolved_path) << "\",";
      out << "\"aliases\":[";
      for (size_t a = 0; a < entry.aliases.size(); ++a) {
        if (a > 0) { out << ","; }
        out << "\"" << this->JsonEscape(entry.aliases[a]) << "\"";
      }
      out << "]}";
    }
    out << "]}";
    return out.str();
  }

 private:
  std::vector<MotionCatalogEntry> entries_;
  std::map<std::string, size_t> lookup_;
  std::string catalog_path_;

  std::string ReadRequiredString(const YAML::Node& node, const std::string& key) const {
    if (!node[key]) { return ""; }
    return node[key].as<std::string>("");
  }

  std::string ReadOptionalString(const YAML::Node& node, const std::string& key, const std::string& fallback) const {
    if (!node[key]) { return fallback; }
    return node[key].as<std::string>(fallback);
  }

  bool ReadOptionalBool(const YAML::Node& node, const std::string& key, bool fallback) const {
    if (!node[key]) { return fallback; }
    return node[key].as<bool>(fallback);
  }

  void RebuildLookup() {
    this->lookup_.clear();
    for (size_t i = 0; i < this->entries_.size(); ++i) {
      const auto& entry = this->entries_[i];
      if (!entry.enabled || entry.motion_index < 0) { continue; }
      for (const auto& key : this->LookupKeysForEntry(entry)) {
        this->lookup_[key] = i;
      }
    }
  }

  std::vector<std::string> LookupKeysForEntry(const MotionCatalogEntry& entry) const {
    std::vector<std::string> keys;
    this->AppendLookupKey(keys, entry.id);
    this->AppendLookupKey(keys, entry.display_name);
    this->AppendLookupKey(keys, std::filesystem::path(entry.raw_path).filename().string());
    this->AppendLookupKey(keys, std::filesystem::path(entry.resolved_path).filename().string());
    for (const auto& alias : entry.aliases) { this->AppendLookupKey(keys, alias); }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
  }

  void AppendLookupKey(std::vector<std::string>& keys, const std::string& raw) const {
    std::string key = this->NormalizeKey(raw);
    if (!key.empty()) { keys.push_back(key); }
  }

  std::string NormalizeKey(const std::string& raw) const {
    std::string trimmed = raw;
    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
      return !std::isspace(ch);
    }));
    trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
      return !std::isspace(ch);
    }).base(), trimmed.end());
    std::transform(trimmed.begin(), trimmed.end(), trimmed.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    return trimmed;
  }

  bool IsInteger(const std::string& value) const {
    if (value.empty()) { return false; }
    size_t start = (value[0] == '-') ? 1 : 0;
    if (start >= value.size()) { return false; }
    return std::all_of(value.begin() + static_cast<std::string::difference_type>(start), value.end(), [](unsigned char ch) {
      return std::isdigit(ch);
    });
  }

  std::string JsonEscape(const std::string& value) const {
    std::ostringstream escaped;
    for (char ch : value) {
      switch (ch) {
        case '\\': escaped << "\\\\"; break;
        case '"': escaped << "\\\""; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default: escaped << ch; break;
      }
    }
    return escaped.str();
  }
};

#endif // MOTION_CATALOG_HPP
