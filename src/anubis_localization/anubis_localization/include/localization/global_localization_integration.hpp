#ifndef GLOBAL_LOCALIZATION_INTEGRATION_HPP
#define GLOBAL_LOCALIZATION_INTEGRATION_HPP

#include <string>

#include <rclcpp/time.hpp>

#include "localization/global_localization_episode.hpp"
#include "localization/global_localization_types.hpp"

namespace localization {

struct GLIntegrationAction {
  bool accept_pose_for_init = false;
  bool freeze_pose_estimator = true;
  bool run_init_verification = false;
  bool keep_episode_active = false;
  bool request_initialpose = false;
  bool decision_shadow = false;
  bool invalid_confirmed_result = false;
  bool stale_confirmed_result = false;
};

struct GLOutputGate {
  bool localization_valid = false;
  bool publish_localization_odom = false;
  bool publish_map_tf = false;
};

const char* glStatusName(GLStatus status);
const char* glEpisodePhaseName(GLEpisodePhase phase);

// Returns true only when the requested automatic-confirmation contract is
// complete.  A false result keeps the episode in decision-shadow mode; it does
// not disable global-localization recall itself.
bool automaticApprovalContractSatisfied(const GLParams& params);

GLIntegrationAction decideGLIntegrationAction(
    const GlobalLocalizationResult& result,
    const GLParams& params,
    bool episode_active,
    bool episode_timed_out);

GLOutputGate decideGLOutputGate(bool episode_active,
                                bool init_verification_complete);

GLOutputGate decideGLOutputGate(const GLEpisodeState& episode,
                                bool init_verification_complete);

bool shouldDropScanAfterGL(const rclcpp::Time& now,
                           const rclcpp::Time& scan_stamp,
                           const GLParams& params);

std::string formatGLSummary(const GlobalLocalizationResult& result,
                            const GLSummaryCounts& summary,
                            const GLEpisodeState& episode,
                            double cost_ms);

}  // namespace localization

#endif  // GLOBAL_LOCALIZATION_INTEGRATION_HPP
