#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

#include "localization/global_localization_spatial.hpp"

namespace localization {
namespace {

constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix4d makePose(double x,
                         double y = 0.0,
                         double yaw_deg = 0.0,
                         double roll_deg = 0.0,
                         double pitch_deg = 0.0) {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  const Eigen::AngleAxisd yaw(yaw_deg * kPi / 180.0, Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd pitch(
      pitch_deg * kPi / 180.0, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd roll(
      roll_deg * kPi / 180.0, Eigen::Vector3d::UnitX());
  pose.block<3, 3>(0, 0) = (yaw * pitch * roll).toRotationMatrix();
  pose(0, 3) = x;
  pose(1, 3) = y;
  return pose;
}

GlobalPoseCandidate validCandidate(double final_x,
                                   double seed_x,
                                   double fitness = 0.10) {
  GlobalPoseCandidate candidate;
  candidate.seed_pose = makePose(seed_x);
  candidate.final_pose = makePose(final_x);
  candidate.fitness_score = fitness;
  candidate.overlap_ratio = 0.8;
  candidate.inlier_count = 200;
  candidate.correction_xy = std::abs(final_x - seed_x);
  candidate.correction_z = 0.0;
  candidate.yaw_delta_deg = 0.0;
  candidate.roll_delta_deg = 0.0;
  candidate.pitch_delta_deg = 0.0;
  candidate.source_type = "grid";
  candidate.support_group = "grid:test";
  candidate.converged = true;
  return candidate;
}

TEST(GlobalLocalizationSpatial, FilterAccountsForEveryRejectRule) {
  const GlobalPoseCandidate valid = validCandidate(0.1, 0.0);
  std::vector<GlobalPoseCandidate> candidates(8, valid);
  candidates[1].converged = false;
  candidates[2].fitness_score = 0.30;
  candidates[3].correction_xy = 3.0;
  candidates[4].correction_z = 2.0;
  candidates[5].yaw_delta_deg = 50.0;
  candidates[6].roll_delta_deg = 11.0;
  candidates[7].final_pose = makePose(0.1, 0.0, 0.0, 16.0, 0.0);

  std::vector<GlobalPoseCandidate> passed;
  GLSummaryCounts summary;
  ASSERT_TRUE(filterRefinedCandidates(
      candidates, GLParams{}, passed, summary));

  ASSERT_EQ(passed.size(), 1U);
  EXPECT_EQ(summary.refined, 8);
  EXPECT_EQ(summary.passed, 1);
  EXPECT_EQ(summary.rejected_conv, 1);
  EXPECT_EQ(summary.rejected_score, 1);
  EXPECT_EQ(summary.rejected_xy, 1);
  EXPECT_EQ(summary.rejected_z, 1);
  EXPECT_EQ(summary.rejected_yaw, 1);
  EXPECT_EQ(summary.rejected_rp, 2);
}

TEST(GlobalLocalizationSpatial, InvalidFilterThresholdFailsClosed) {
  GLParams params;
  params.max_correction_xy = std::numeric_limits<double>::quiet_NaN();
  std::vector<GlobalPoseCandidate> passed;
  GLSummaryCounts summary;

  EXPECT_FALSE(filterRefinedCandidates(
      {validCandidate(0.1, 0.0)}, params, passed, summary));
  EXPECT_TRUE(passed.empty());
}

TEST(GlobalLocalizationSpatial, ConnectedComponentsAreOrderIndependent) {
  std::vector<GlobalPoseCandidate> candidates = {
      validCandidate(0.0, 0.0, 0.12),
      validCandidate(0.1, 0.5, 0.08),
      validCandidate(0.4, 2.0, 0.10),
      validCandidate(5.0, 5.0, 0.09)};
  GLSummaryCounts first_summary;
  std::vector<GlobalCluster> first_clusters;
  ASSERT_TRUE(clusterCandidates(
      candidates, GLParams{}, first_clusters, first_summary));

  std::reverse(candidates.begin(), candidates.end());
  GLSummaryCounts reversed_summary;
  std::vector<GlobalCluster> reversed_clusters;
  ASSERT_TRUE(clusterCandidates(
      candidates, GLParams{}, reversed_clusters, reversed_summary));

  ASSERT_EQ(first_clusters.size(), 2U);
  ASSERT_EQ(reversed_clusters.size(), first_clusters.size());
  for (size_t index = 0; index < first_clusters.size(); ++index) {
    EXPECT_EQ(reversed_clusters[index].size, first_clusters[index].size);
    EXPECT_EQ(reversed_clusters[index].support_group_count,
              first_clusters[index].support_group_count);
    EXPECT_TRUE(reversed_clusters[index].medoid_pose.isApprox(
        first_clusters[index].medoid_pose, 1e-12));
  }
  EXPECT_NEAR(first_clusters.front().medoid_pose(0, 3), 0.1, 1e-12);
  EXPECT_EQ(first_clusters.front().support_group_count, 2);
}

TEST(GlobalLocalizationSpatial, OversizedBridgeComponentIsSplitByMst) {
  std::vector<GlobalPoseCandidate> candidates = {
      validCandidate(0.0, 0.0), validCandidate(0.4, 2.0),
      validCandidate(0.8, 4.0), validCandidate(1.2, 6.0)};
  std::vector<GlobalCluster> clusters;
  GLSummaryCounts summary;

  ASSERT_TRUE(clusterCandidates(candidates, GLParams{}, clusters, summary));
  ASSERT_GE(clusters.size(), 2U);
  for (const GlobalCluster& cluster : clusters) {
    for (size_t first = 0; first < cluster.member_indices.size(); ++first) {
      for (size_t second = first + 1;
           second < cluster.member_indices.size(); ++second) {
        const double first_x = candidates[cluster.member_indices[first]].final_pose(0, 3);
        const double second_x = candidates[cluster.member_indices[second]].final_pose(0, 3);
        EXPECT_LE(std::abs(first_x - second_x), 1.0 + 1e-12);
      }
    }
  }
}

TEST(GlobalLocalizationSpatial, CircularYawKeepsWrapBoundaryNeighborsTogether) {
  GlobalPoseCandidate first = validCandidate(0.0, 0.0, 0.10);
  GlobalPoseCandidate second = validCandidate(0.1, 2.0, 0.11);
  first.final_pose = makePose(0.0, 0.0, 179.0);
  second.final_pose = makePose(0.1, 0.0, -179.0);
  std::vector<GlobalCluster> clusters;
  GLSummaryCounts summary;

  ASSERT_TRUE(clusterCandidates(
      {first, second}, GLParams{}, clusters, summary));
  ASSERT_EQ(clusters.size(), 1U);
  EXPECT_NEAR(std::abs(clusters.front().mean_yaw_deg), 180.0, 1e-9);
}

TEST(GlobalLocalizationSpatial, MedoidTieUsesDocumentedQualityOrder) {
  GlobalPoseCandidate first = validCandidate(0.0, 0.0, 0.12);
  GlobalPoseCandidate second = validCandidate(0.4, 2.0, 0.08);
  first.overlap_ratio = 0.9;
  second.overlap_ratio = 0.7;
  std::vector<GlobalCluster> clusters;
  GLSummaryCounts summary;

  ASSERT_TRUE(clusterCandidates(
      {first, second}, GLParams{}, clusters, summary));
  ASSERT_EQ(clusters.size(), 1U);
  EXPECT_NEAR(clusters.front().medoid_pose(0, 3), 0.4, 1e-12);
}

TEST(GlobalLocalizationSpatial, LocalStateSourcesCountAsOneSupportGroup) {
  GlobalPoseCandidate ukf = validCandidate(0.0, 0.0);
  ukf.source_type = "ukf";
  ukf.support_group = "local_state";
  GlobalPoseCandidate odom = validCandidate(0.1, 2.0);
  odom.source_type = "odom";
  odom.support_group = "odom";
  GlobalPoseCandidate rviz = validCandidate(0.2, 4.0);
  rviz.source_type = "rviz";
  rviz.support_group = "rviz";
  std::vector<GlobalCluster> clusters;
  GLSummaryCounts summary;

  ASSERT_TRUE(clusterCandidates(
      {ukf, odom, rviz}, GLParams{}, clusters, summary));
  ASSERT_EQ(clusters.size(), 1U);
  EXPECT_EQ(clusters.front().support_group_count, 2);
}

TEST(GlobalLocalizationSpatial, RetrievalSupportGroupsUseGridSpatialMerge) {
  GLParams params;
  std::vector<GlobalPoseCandidate> candidates;
  for (int index = 0; index < 3; ++index) {
    GlobalPoseCandidate candidate = validCandidate(0.2 * index, 0.2 * index);
    candidate.source_type = "retrieval";
    candidate.support_group = "grid:kf7:xy:" + std::to_string(index);
    candidates.push_back(candidate);
  }
  std::vector<GlobalCluster> clusters;
  GLSummaryCounts summary;
  ASSERT_TRUE(clusterCandidates(candidates, params, clusters, summary));
  ASSERT_EQ(clusters.size(), 1U);
  EXPECT_EQ(clusters.front().support_group_count, 1);
}

TEST(GlobalLocalizationSpatial, DecisionDistinguishesAmbiguousAndLowConsensus) {
  GlobalCluster first;
  first.size = 3;
  first.support_group_count = 2;
  first.best_score = 0.10;
  first.medoid_pose = makePose(0.0);
  GlobalCluster second = first;
  second.best_score = 0.11;
  second.medoid_pose = makePose(5.0);
  GLSummaryCounts summary;

  GlobalSpatialDecision decision = selectBestCluster(
      {first, second}, GLParams{}, summary);
  EXPECT_EQ(decision.status, GLStatus::Ambiguous);
  EXPECT_EQ(summary.best_support_group_count, 2);

  second.best_score = 0.20;
  decision = selectBestCluster({first, second}, GLParams{}, summary);
  EXPECT_EQ(decision.status, GLStatus::PendingVote);

  first.support_group_count = 1;
  decision = selectBestCluster({first}, GLParams{}, summary);
  EXPECT_EQ(decision.status, GLStatus::LowConsensus);
}

// Regression for the measured P2 dataset (2026-08 field logs): the correct
// cluster refines to a better score than the wrong repeated-structure cluster
// but only has one independent support group. Ranking must be quality-first
// (correct cluster = rank 1) and the distant near-score competitor must
// produce Ambiguous instead of letting either cluster enter temporal voting.
TEST(GlobalLocalizationSpatial, QualityFirstRankingReportsMeasuredP2Ambiguity) {
  GlobalCluster correct;
  correct.size = 1;
  correct.support_group_count = 1;
  correct.best_score = 0.020859;
  correct.medoid_pose = makePose(-4.314, 0.664, -176.586);
  GlobalCluster wrong;
  wrong.size = 10;
  wrong.support_group_count = 4;
  wrong.best_score = 0.022061;
  wrong.medoid_pose = makePose(0.475, -0.534, 1.761);
  GLSummaryCounts summary;

  const GlobalSpatialDecision decision = selectBestCluster(
      {wrong, correct}, GLParams{}, summary);
  EXPECT_EQ(decision.best_cluster_index, 1);
  EXPECT_EQ(decision.second_cluster_index, 0);
  EXPECT_EQ(decision.status, GLStatus::Ambiguous);
  EXPECT_EQ(summary.best_support_group_count, 1);
}

// Measured repeated-view pair (top-2 distance 1.655 m, score ratio 3.691):
// the score gap is decisive and the medoids are within same-site range, so
// this is not spatial ambiguity — the quality-ranked winner proceeds to the
// temporal vote, which remains the guard against a wrong dominant score.
TEST(GlobalLocalizationSpatial, DominantScoreNearbyPairIsNotAmbiguous) {
  GlobalCluster dominant;
  dominant.size = 3;
  dominant.support_group_count = 2;
  dominant.best_score = 0.020;
  dominant.medoid_pose = makePose(0.0);
  GlobalCluster runner_up = dominant;
  runner_up.best_score = 0.020 * 3.691;
  runner_up.medoid_pose = makePose(1.655);
  GLSummaryCounts summary;

  const GlobalSpatialDecision decision = selectBestCluster(
      {runner_up, dominant}, GLParams{}, summary);
  EXPECT_EQ(decision.best_cluster_index, 1);
  EXPECT_EQ(decision.status, GLStatus::PendingVote);
}

// A distant single-support cluster scoring within the ambiguity ratio must
// still block confirmation: lack of independent support is not a reason to
// ignore a materially different competing conclusion.
TEST(GlobalLocalizationSpatial, SingleSupportCompetitorStillTriggersAmbiguous) {
  GlobalCluster best;
  best.size = 4;
  best.support_group_count = 2;
  best.best_score = 0.10;
  best.medoid_pose = makePose(0.0);
  GlobalCluster competitor;
  competitor.size = 1;
  competitor.support_group_count = 1;
  competitor.best_score = 0.105;
  competitor.medoid_pose = makePose(5.0);
  GLSummaryCounts summary;

  const GlobalSpatialDecision decision = selectBestCluster(
      {best, competitor}, GLParams{}, summary);
  EXPECT_EQ(decision.status, GLStatus::Ambiguous);
}

// Non-finite scores must sort last and never win the ranking.
TEST(GlobalLocalizationSpatial, NonFiniteScoreClusterRanksLast) {
  GlobalCluster healthy;
  healthy.size = 1;
  healthy.support_group_count = 2;
  healthy.best_score = 0.12;
  healthy.medoid_pose = makePose(3.0);
  GlobalCluster broken;
  broken.size = 5;
  broken.support_group_count = 4;
  broken.best_score = std::numeric_limits<double>::quiet_NaN();
  broken.medoid_pose = makePose(0.0);
  GLSummaryCounts summary;

  const GlobalSpatialDecision decision = selectBestCluster(
      {broken, healthy}, GLParams{}, summary);
  EXPECT_EQ(decision.best_cluster_index, 1);
  EXPECT_EQ(decision.status, GLStatus::PendingVote);
}

}  // namespace
}  // namespace localization
