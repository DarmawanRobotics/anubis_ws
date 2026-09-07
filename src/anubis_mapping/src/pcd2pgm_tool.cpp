/**
 * @brief 离线把 PCD 点云重制成 2D 栅格图（PGM + YAML）。
 *
 * [2026-08-14] 建图保存时 pcd2pgm 的 z 过滤下限是 -0.20，把地面点
 * 也计入了栅格 —— 高密度地面点让每个格子都 ≥2 点，地板被画成"墙"，
 * 栅格图 82% 占用、自由空间为 0、导航不可用。本工具用抬高后的下限
 * （默认 0.10；2026-08-26 前为 0.05）离线重制，无需重新走图。
 *
 * 用法:
 *   pcd2pgm_tool <map.pcd> [输出前缀, 默认与输入同目录 "map"]
 *   pcd2pgm_tool <map.pcd> <输出前缀> <离地z最小> <离地z最大>
 *                <悬空净高> <floor_z_map> [自适应地面 true|false]
 *                [地面模型残差P95上限] [栅格分辨率]
 *                [占用最少点数] [半径滤波半径] [半径滤波最少邻居]
 *                [地面拟合失败是否拒绝 true|false]
 *
 * 输出: <输出前缀>.pgm + <输出前缀>.yaml（与 map_server 格式一致）。
 */
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>

#include "common.h"
#include "pcd2grid.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: pcd2pgm_tool <map.pcd> [输出前缀] [离地z最小] "
              << "[离地z最大] [悬空净高] [floor_z_map] "
              << "[自适应地面 true|false] [地面模型残差P95上限] "
                 "[栅格分辨率] [占用最少点数] [半径滤波半径] "
                 "[半径滤波最少邻居] [地面拟合失败是否拒绝 true|false]\n"
              << "默认: floor_z_map=-0.304, 离地 z 过滤 [0.10, 3.00]，"
              << "悬空净高=0.55m，栅格分辨率=0.05m，"
              << "输出前缀 = 输入同名\n";
    return 1;
  }
  const std::string pcd_path = argv[1];
  const std::string out_prefix =
      argc > 2 ? argv[2] : pcd_path.substr(0, pcd_path.rfind(".pcd"));

  anubis_mapping::Pcd2GridOptions options;
  options.file_name = out_prefix.substr(out_prefix.find_last_of("/\\") + 1);
  options.floor_z_map = argc > 6 ? std::stod(argv[6]) : -0.304;
  // [2026-08-14] 步态颠簸让墙下部点 z 散布，1.00 上限切掉太多（实测占用仅
  // 5%），放宽高度带后占用率回到 20%（正常室内图）。
  options.thre_z_min = argc > 3 ? std::stod(argv[3]) : 0.10;
  // 上限放宽到 3.0：让门楣/横梁点完整进入，靠 overhead_clearance_z 的
  // "最低点高度"判定区分悬空结构与实心墙（各房间门楣高度 1.95~2.45m 不同，
  // 固定上限切不干净 —— 实测隔墙1/2 的门被误判占用、隔墙3 恰好躲过）。
  options.thre_z_max = argc > 4 ? std::stod(argv[4]) : 3.00;
  options.overhead_clearance_z = argc > 5 ? std::stod(argv[5]) : 0.55;
  options.flag_pass_through = 0;
  options.map_resolution = argc > 9 ? std::stod(argv[9]) : 0.05;
  options.min_points_occupied = argc > 10 ? std::stoi(argv[10]) : 2;
  options.thre_radius = argc > 11 ? std::stod(argv[11]) : 0.15;
  options.thres_point_count = argc > 12 ? std::stoi(argv[12]) : 2;
  options.ground_free_enabled = true;
  options.ground_free_min_height = -0.15;
  // 离线 PCD 不含逐关键帧点云，无法安全恢复每条射线；仍可用聚合地面点
  // 填补已观测空闲格。在线建图保存会额外传入关键帧并启用射线清空。
  options.raytrace_free_enabled = false;
  options.raytrace_max_range = 10.0;
  options.low_obstacle_diagnostic_height = 0.25;
  options.ground_plane_enabled = argc <= 7 || std::string(argv[7]) != "false";
  options.ground_plane_required =
      argc <= 13 || std::string(argv[13]) != "false";
  options.ground_plane_max_residual_p95 =
      argc > 8 ? std::stod(argv[8]) : 0.08;
  if (!std::isfinite(options.map_resolution) ||
      options.map_resolution <= 0.0 ||
      !std::isfinite(options.thre_radius) || options.thre_radius < 0.0 ||
      options.min_points_occupied < 1 || options.thres_point_count < 0)
  {
    std::cerr << "无效的量化参数: resolution=" << options.map_resolution
              << " min_points_occupied=" << options.min_points_occupied
              << " radius=" << options.thre_radius
              << " thres_point_count=" << options.thres_point_count << "\n";
    return 1;
  }

  anubis_mapping::CloudPtr cloud(new anubis_mapping::PointCloudType());
  if (pcl::io::loadPCDFile<anubis_mapping::PointType>(pcd_path, *cloud) != 0 ||
      cloud->empty()) {
    std::cerr << "无法加载 PCD: " << pcd_path << "\n";
    return 1;
  }

  // [2026-08-14] 按描述子库的位姿范围 ±3m 裁剪：窗外远处杂点会把栅格图
  // 撑大（实测 53m 长）并稀释墙点，裁剪后占用率才正常。
  const std::string db_path =
      pcd_path.substr(0, pcd_path.rfind(".pcd")) + "_scd.bin";
  std::ifstream db(db_path, std::ios::binary);
  if (db.is_open())
  {
    char magic[4];
    db.read(magic, 4);
    if (std::memcmp(magic, "SCDB", 4) == 0)
    {
      db.seekg(4 * 3, std::ios::cur);  // version/rings/sectors (u32 x3)
      db.seekg(4 * 8, std::ios::cur);  // range/z config
      db.seekg(4, std::ios::cur);      // normalization
      uint64_t kf_count = 0;
      db.read(reinterpret_cast<char*>(&kf_count), 8);
      double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
      for (uint64_t i = 0; i < kf_count; ++i)
      {
        db.seekg(16, std::ios::cur);  // id + stamp
        double pose[16];
        db.read(reinterpret_cast<char*>(pose), 128);
        uint64_t gs, ks;
        db.read(reinterpret_cast<char*>(&gs), 8);
        db.read(reinterpret_cast<char*>(&ks), 8);
        db.seekg(static_cast<std::streamoff>((gs + ks) * 4), std::ios::cur);
        x_min = std::min(x_min, pose[3]);
        x_max = std::max(x_max, pose[3]);
        y_min = std::min(y_min, pose[7]);
        y_max = std::max(y_max, pose[7]);
      }
      anubis_mapping::CloudPtr cropped(new anubis_mapping::PointCloudType());
      cropped->reserve(cloud->size());
      for (const auto& point : cloud->points)
      {
        if (point.x > x_min - 3.0 && point.x < x_max + 3.0 &&
            point.y > y_min - 3.0 && point.y < y_max + 3.0)
        {
          cropped->push_back(point);
        }
      }
      std::cout << "轨迹裁剪(±3m): " << cloud->size() << " -> " << cropped->size()
                << " 点 (" << kf_count << " 关键帧)\n";
      cloud = cropped;
    }
  }

  std::cout << "加载 " << pcd_path << ": " << cloud->size() << " 点\n"
            << "floor_z_map=" << options.floor_z_map << "m，离地 z 过滤 ["
            << options.thre_z_min << ", " << options.thre_z_max
            << "]m，地面模型="
            << (options.ground_plane_enabled
                    ? (options.ground_plane_required
                           ? "自适应平面(拟合失败拒绝保存)"
                           : "自适应平面(拟合失败回退固定地面)")
                    : "固定 floor_z_map")
            << "，残差P95上限 "
            << options.ground_plane_max_residual_p95 << "m"
            << "，分辨率 "
            << options.map_resolution << "m"
            << "，占用最少点数 " << options.min_points_occupied
            << "，半径滤波 [" << options.thre_radius << "m, "
            << options.thres_point_count << " neighbors]\n";

  anubis_mapping::Pcd2Grid pcd2grid(options);
  if (!pcd2grid.run(cloud, out_prefix)) {
    std::cerr << "栅格图生成失败\n";
    return 1;
  }
  std::cout << "已生成: " << out_prefix << ".pgm / " << out_prefix << ".yaml\n";
  return 0;
}
