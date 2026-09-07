#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <utility>

#include "keyframe_store.h"

TEST(KeyframeStoreTest, TriggerUsesTranslationOrYaw) {
    Eigen::Matrix4d origin = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d moved = origin;
    moved(0, 3) = 0.5;
    EXPECT_TRUE(anubis_mapping::shouldCreateKeyframe(origin, moved, 0.5, 0.2));
    Eigen::Matrix4d rotated = origin;
    rotated.block<3, 3>(0, 0) = Eigen::AngleAxisd(
        0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    EXPECT_TRUE(anubis_mapping::shouldCreateKeyframe(origin, rotated, 0.5, 0.2));
    moved(0, 3) = 0.49;
    EXPECT_FALSE(anubis_mapping::shouldCreateKeyframe(origin, moved, 0.5, 0.2));
}

TEST(KeyframeStoreTest, RequiresCoarseAndDenseClouds)
{
    anubis_mapping::KeyframeStore store;
    anubis_mapping::Keyframe missing_dense;
    auto coarse = anubis_mapping::PointCloudType::Ptr(
        new anubis_mapping::PointCloudType());
    coarse->push_back(anubis_mapping::PointType{});
    missing_dense.cloud_lidar = coarse;
    EXPECT_FALSE(store.append(std::move(missing_dense), 10U));

    anubis_mapping::Keyframe complete;
    complete.cloud_lidar = coarse;
    auto dense = anubis_mapping::PointCloudType::Ptr(
        new anubis_mapping::PointCloudType());
    dense->push_back(anubis_mapping::PointType{});
    complete.dense_cloud_lidar = dense;
    EXPECT_TRUE(store.append(std::move(complete), 10U));
    ASSERT_EQ(store.snapshot().size(), 1U);
    EXPECT_EQ(store.snapshot().front().cloud_lidar->size(), 1U);
    EXPECT_EQ(store.snapshot().front().dense_cloud_lidar->size(), 1U);
}
