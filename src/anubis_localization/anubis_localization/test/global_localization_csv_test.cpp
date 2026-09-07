#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "localization/global_localization_csv.hpp"

namespace localization {
namespace {

struct TemporaryCsv {
  TemporaryCsv() {
    const auto suffix = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
        ("global_localization_csv_test_" + std::to_string(suffix) + ".csv");
  }

  ~TemporaryCsv() {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }

  std::filesystem::path path;
};

std::vector<std::string> parseCsvRow(const std::string& row) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (size_t index = 0; index < row.size(); ++index) {
    const char character = row[index];
    if (quoted) {
      if (character == '"' && index + 1U < row.size() &&
          row[index + 1U] == '"') {
        field.push_back('"');
        ++index;
      } else if (character == '"') {
        quoted = false;
      } else {
        field.push_back(character);
      }
    } else if (character == '"') {
      quoted = true;
    } else if (character == ',') {
      fields.push_back(field);
      field.clear();
    } else {
      field.push_back(character);
    }
  }
  fields.push_back(field);
  return fields;
}

std::vector<std::vector<std::string>> readCsv(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    rows.push_back(parseCsvRow(line));
  }
  return rows;
}

size_t columnIndex(const std::vector<std::string>& header,
                   const std::string& name) {
  for (size_t index = 0; index < header.size(); ++index) {
    if (header[index] == name) {
      return index;
    }
  }
  return header.size();
}

GlobalPoseCandidate makeCandidate(double x, const std::string& source) {
  GlobalPoseCandidate candidate;
  candidate.seed_pose(0, 3) = x;
  candidate.final_pose(0, 3) = x + 0.1;
  candidate.source_type = source;
  candidate.grid_cell_id = 7;
  candidate.yaw_bin = 2;
  candidate.rotation_batch = 1;
  candidate.support_group = "grid:7:2,group";
  candidate.evidence_id = "anchor:1";
  candidate.evidence_role = "anchor_recall";
  candidate.coarse_score = 0.2;
  candidate.fitness_score = 0.1;
  candidate.converged = true;
  return candidate;
}

TEST(GLCandidateCsvWriter, WritesCompatibleRowsOnceAndEscapesStrings) {
  TemporaryCsv temporary;
  GLCandidateCsvWriter writer(temporary.path);
  GLCandidateCsvContext context;
  context.episode_id = "fixture,\"room_a\"";
  context.scan_stamp_ns = 123456789;
  context.processing_ms = 12.5;

  GlobalLocalizationResult result;
  result.status = GLStatus::PendingVote;
  result.seed_candidates.push_back(makeCandidate(1.0, "grid"));
  result.level0_candidates.push_back(
      makeCandidate(2.0, "level0_region"));
  result.candidates.push_back(makeCandidate(2.5, "grid"));
  result.passed_candidates = result.candidates;
  GlobalCluster cluster;
  cluster.member_indices = {0};
  cluster.medoid_pose(0, 3) = 3.0;
  cluster.size = 2;
  cluster.support_group_count = 2;
  cluster.best_score = 0.1;
  cluster.mean_score = 0.12;
  result.clusters.push_back(cluster);
  GLSummaryCounts summary;
  summary.seed_total = 1;
  summary.coarse_kept = 1;
  summary.refined = 1;
  summary.passed = 1;
  summary.cluster_count = 1;

  std::string error;
  ASSERT_TRUE(writer.appendProbe(context, result, summary, true, error))
      << error;
  ASSERT_TRUE(writer.appendProbe(context, result, summary, true, error))
      << error;

  const auto rows = readCsv(temporary.path);
  ASSERT_EQ(rows.size(), 6U);
  const auto& header = rows.front();
  ASSERT_EQ(header.size(), 68U);
  for (const auto& row : rows) {
    EXPECT_EQ(row.size(), header.size());
  }
  const size_t episode = columnIndex(header, "episode_id");
  const size_t stage = columnIndex(header, "stage");
  const size_t region = columnIndex(header, "region");
  const size_t status = columnIndex(header, "status");
  const size_t cluster_id = columnIndex(header, "cluster_id");
  ASSERT_LT(status, header.size());
  EXPECT_EQ(rows[1][episode], context.episode_id);
  EXPECT_EQ(rows[1][stage], "seed");
  EXPECT_TRUE(rows[1][region].empty());
  EXPECT_EQ(rows[2][stage], "coarse");
  EXPECT_EQ(rows[3][stage], "refined");
  EXPECT_EQ(rows[3][status], "Passed");
  EXPECT_EQ(rows[3][cluster_id], "0");
  EXPECT_EQ(rows[4][stage], "cluster");
  EXPECT_EQ(rows[5][stage], "decision");
  EXPECT_EQ(rows[5][status], "PendingVote");
}

TEST(GLCandidateCsvWriter, AppendsOneConfirmedDecisionForAnotherEpisode) {
  TemporaryCsv temporary;
  GLCandidateCsvWriter writer(temporary.path);
  GLCandidateCsvContext context;
  context.episode_id = "confirmed";
  context.scan_stamp_ns = 99;

  GlobalLocalizationResult result;
  result.status = GLStatus::Confirmed;
  result.success = true;
  result.final_pose(0, 3) = 4.0;
  GLSummaryCounts summary;
  std::string error;
  ASSERT_TRUE(writer.appendDecision(context, result, summary, error)) << error;
  ASSERT_TRUE(writer.appendDecision(context, result, summary, error)) << error;

  const auto rows = readCsv(temporary.path);
  ASSERT_EQ(rows.size(), 2U);
  const size_t status = columnIndex(rows[0], "status");
  const size_t x = columnIndex(rows[0], "x");
  EXPECT_EQ(rows[1][status], "Confirmed");
  EXPECT_EQ(rows[1][x], "4");
}

TEST(GLCandidateCsvWriter, RefusesAnExistingIncompatibleHeader) {
  TemporaryCsv temporary;
  {
    std::ofstream output(temporary.path);
    output << "wrong,header\n";
  }
  GLCandidateCsvWriter writer(temporary.path);
  GLCandidateCsvContext context;
  context.episode_id = "episode";
  context.scan_stamp_ns = 1;
  GlobalLocalizationResult result;
  GLSummaryCounts summary;
  std::string error;

  EXPECT_FALSE(writer.appendDecision(context, result, summary, error));
  EXPECT_NE(error.find("header mismatch"), std::string::npos);
}

}  // namespace
}  // namespace localization
