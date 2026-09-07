/**
 * @file mapping_node_main.cpp
 * @brief Entry point for the mapping node. Ported from darmawan_ws's
 * test/mapping_demo.cpp (renamed -- this is the real entry point, not a
 * demo). Runs the mapping loop at 5kHz via spin_some() + a manual run()
 * call rather than rclcpp::spin(), matching the original design's
 * comment about keeping the mapping loop free of ROS executor latency.
 */

#include "mapping_alg.h"

static std::atomic<bool> flag{ false };

void SigHandle(int sig)
{
    (void)sig;
    if (!flag.exchange(true, std::memory_order_relaxed))
    {
        rclcpp::shutdown();
    }
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::signal(SIGINT, SigHandle);
    std::signal(SIGTERM, SigHandle);
    auto         node = std::make_shared<anubis_mapping::MappingAlg>();
    rclcpp::Rate rate(5000);
    while (rclcpp::ok())
    {
        if (flag)
            break;
        rclcpp::spin_some(node);
        node->run();
        rate.sleep();
    }
    node->finish();
    return 0;
}
