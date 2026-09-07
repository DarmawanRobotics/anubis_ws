#ifndef GLOBAL_LOCALIZATION_CSV_HPP
#define GLOBAL_LOCALIZATION_CSV_HPP

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>

#include "localization/global_localization_types.hpp"

namespace localization {

struct GLCandidateCsvContext {
  std::string episode_id;
  int64_t scan_stamp_ns = 0;
  double processing_ms = 0.0;
};

// Appends evaluator-compatible candidate snapshots without participating in
// localization decisions. The required `region` column is intentionally left
// empty: semantic regions are resolved from independent map polygons by the
// offline evaluator, never copied from the episode ground truth.
class GLCandidateCsvWriter {
public:
  explicit GLCandidateCsvWriter(std::filesystem::path output_path);

  bool appendProbe(const GLCandidateCsvContext& context,
                   const GlobalLocalizationResult& result,
                   const GLSummaryCounts& summary,
                   bool include_decision,
                   std::string& error);

  bool appendDecision(const GLCandidateCsvContext& context,
                      const GlobalLocalizationResult& result,
                      const GLSummaryCounts& summary,
                      std::string& error);

  const std::filesystem::path& outputPath() const;
  static const char* header();

private:
  bool appendPayloadLocked(const std::string& payload, std::string& error);

  std::filesystem::path output_path_;
  std::mutex mutex_;
  std::set<std::string> probe_episodes_;
  std::set<std::string> decision_episodes_;
};

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_CSV_HPP
