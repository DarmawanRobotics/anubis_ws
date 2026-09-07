#include <gtest/gtest.h>

#include <localization/sensor_data_validity.hpp>

namespace localization {
namespace {

TEST(SensorDataValidity, RequiresFreshLidarAndConfiguredImu)
{
  EXPECT_TRUE(sensorDataValid(true, true, true));
  EXPECT_FALSE(sensorDataValid(true, false, true));
  EXPECT_FALSE(sensorDataValid(true, true, false));
  EXPECT_TRUE(sensorDataValid(false, true, false));
  EXPECT_FALSE(sensorDataValid(false, false, true));
}

}  // namespace
}  // namespace localization
