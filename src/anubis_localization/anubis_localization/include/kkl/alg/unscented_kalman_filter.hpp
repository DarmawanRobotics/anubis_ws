/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl {
  namespace alg {

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class UnscentedKalmanFilterX {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param input_dim            input vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  UnscentedKalmanFilterX(const System& system, int state_dim, int input_dim, int measurement_dim, const MatrixXt& process_noise, const MatrixXt& measurement_noise, const VectorXt& mean, const MatrixXt& cov)
    : state_dim(state_dim),
    input_dim(input_dim),
    measurement_dim(measurement_dim),
    N(state_dim),
    M(input_dim),
    K(measurement_dim),
    S(2 * state_dim + 1),
    mean(mean),
    cov(cov),
    system(system),
    process_noise(process_noise),
    measurement_noise(measurement_noise),
    lambda(1),
    normal_dist(0.0, 1.0)
  {
    weights.resize(S, 1);
    sigma_points.resize(S, N);
    ext_weights.resize(2 * (N + K) + 1, 1);
    ext_sigma_points.resize(2 * (N + K) + 1, N + K);
    expected_measurements.resize(2 * (N + K) + 1, K);
    // [2026-08-16] Eigen 的 resize 不做值初始化。sigma_points 只有 predict() 会写，
    // 若外部在首次 predict 之前读它（诊断代码就这么干过）即为未定义行为，实测读出
    // inf。显式清零。
    sigma_points.setZero();
    ext_sigma_points.setZero();
    expected_measurements.setZero();

    // initialize weights for unscented filter
    weights[0] = lambda / (N + lambda);
    for (int i = 1; i < 2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // weights for extended state space which includes error variances
    ext_weights[0] = lambda / (N + K + lambda);
    for (int i = 1; i < 2 * (N + K) + 1; i++) {
      ext_weights[i] = 1 / (2 * (N + K + lambda));
    }
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict() {
    // calculate sigma points
    predict_count++;
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i));
    }
    // 流形修正：半球对齐（详见带控制量的 predict 重载）
    {
      VectorXt reference = sigma_points.row(0).transpose();
      for (int i = 1; i < S; i++) {
        VectorXt sp = sigma_points.row(i).transpose();
        system.alignStateToReference(sp, reference);
        sigma_points.row(i) = sp.transpose();
      }
    }

    const auto& R = process_noise;

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    system.normalizeState(mean_pred);
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += R;

    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict(const VectorXt& control) {
    // calculate sigma points
    predict_count++;
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i), control);
    }
    // [2026-08-16] 流形修正：q 与 -q 同一旋转，混在一起做算术平均会相消。
    // 传播后先统一对齐到 row 0（均值传播的结果），再参与平均。
    {
      VectorXt reference = sigma_points.row(0).transpose();
      for (int i = 1; i < S; i++) {
        VectorXt sp = sigma_points.row(i).transpose();
        system.alignStateToReference(sp, reference);
        sigma_points.row(i) = sp.transpose();
      }
    }

    const auto& R = process_noise;

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    // 线性加权平均会把单位四元数拉进球内；先归一化再算协方差，
    // 否则协方差是相对一个非法均值算的。
    system.normalizeState(mean_pred);
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += R;

    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  void correct(const VectorXt& measurement) {
    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    ext_cov_pred.bottomRightCorner(K, K) = measurement_noise;

    ensurePositiveFinite(ext_cov_pred);
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points);

    // [2026-08-16] 流形修正（状态侧）：sigma 点扰动实测达 0.28~0.73，足以让部分
    // 点落到对映半球；不对齐就做算术平均会相消。
    for (int i = 1; i < ext_sigma_points.rows(); i++) {
      VectorXt sp = ext_sigma_points.row(i).transpose();
      system.alignStateToReference(sp, ext_mean_pred);
      ext_sigma_points.row(i) = sp.transpose();
    }

    // unscented transform
    expected_measurements.setZero();
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }

    // 流形修正（观测侧）：h() 输出的是单位四元数，同样要统一半球后再平均，
    // 否则 expected_measurement_mean 会被相消拉短，innovation 虚高。
    for (int i = 0; i < expected_measurements.rows(); i++) {
      VectorXt em = expected_measurements.row(i).transpose();
      system.alignObservationToReference(em, measurement);
      expected_measurements.row(i) = em.transpose();
    }

    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    MatrixXt sigma = MatrixXt::Zero(N + K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    // [2026-08-16] 原为 `sigma * expected_measurement_cov.inverse()`。显式求逆在
    // 协方差趋零时数值极脆（本项目实测过 cov 塌缩到 4e-4），改用 LDLT 求解：
    //   Kᵀ = S⁻¹·sigmaᵀ  （S 对称）
    kalman_gain =
      expected_measurement_cov.ldlt().solve(sigma.transpose()).transpose();
    const auto& K = kalman_gain;

    VectorXt ext_mean = ext_mean_pred + K * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();

    mean = ext_mean.topLeftCorner(N, 1);
    // 关键：mean += K·innovation 会把四元数推离单位球，而 h() 把观测侧归一化掉，
    // 模长因此成为**不可观测且无阻尼**的自由度。predict 一停（回调阻塞时确实会
    // 停），模长就自由落体 —— 实测塌到 0.1424，随后 cov 衰减、增益归零、
    // Mahalanobis 门锁死，状态彻底冻结。这一行是止血点。
    system.normalizeState(mean);
    cov = ext_cov.topLeftCorner(N, N);
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return sigma_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m) { mean = m;			return *this; }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) { cov = s;			return *this; }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) { process_noise = p;			return *this; }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;	return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int input_dim;
  const int measurement_dim;

  const int N;
  const int M;
  const int K;
  const int S;

public:
  VectorXt mean;
  MatrixXt cov;

  System system;
  MatrixXt process_noise;		//
  MatrixXt measurement_noise;	//

  T lambda;
  VectorXt weights;

  MatrixXt sigma_points;

  VectorXt ext_weights;
  MatrixXt ext_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    Eigen::LLT<MatrixXt> llt;
    llt.compute((n + lambda) * cov);
    if (llt.info() != Eigen::Success) {
      // [2026-08-16] 原实现不检查 info()：分解失败时 matrixL() 的内容是未定义的，
      // 会被当作合法 sigma 点用下去，污染整个滤波器且毫无征兆。
      // 回退策略：对角加载后重试；仍失败则让全部 sigma 点退化为均值（等价于
      // 本帧不做无迹变换，只做确定性传播），这比用垃圾值安全。
      MatrixXt regularized = (n + lambda) * cov;
      regularized.diagonal().array() += static_cast<T>(1e-6);
      llt.compute(regularized);
      if (llt.info() != Eigen::Success) {
        llt_failure_count++;
        for (int i = 0; i < 2 * n + 1; i++) {
          sigma_points.row(i) = mean;
        }
        return;
      }
      llt_failure_count++;
    }
    MatrixXt l = llt.matrixL();

    sigma_points.row(0) = mean;
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
    }
  }

  /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   *
   * [2026-08-16] 原实现整个函数体是死代码（首行就是 return），协方差退化时没有
   * 任何保护 —— computeSigmaPoints 里的 LLT 会静默产出垃圾 sigma 点。
   * 原实现本身也不合适：用非对称的 EigenSolver 处理协方差（应为
   * SelfAdjointEigenSolver），且 V*D*V.inverse() 数值稳定性差，还是每帧 O(n³)
   * 特征分解（40Hz 调用，很可能正是当初被禁用的原因）。
   * 换成便宜且对症的版本：对称化 + 对角下限，O(n²)，覆盖实际的退化路径。
   */
  void ensurePositiveFinite(MatrixXt& cov) {
    // 协方差按定义对称，但浮点累积会引入不对称，进而让 LLT 失败
    cov = (T(0.5) * (cov + cov.transpose())).eval();
    // 对角下限：防止数值下溢导致分解失败
    const T eps = static_cast<T>(1e-9);
    for (int i = 0; i < cov.rows(); i++) {
      if (!(cov(i, i) > eps)) {
        cov(i, i) = eps;
      }
    }
  }

public:
  MatrixXt kalman_gain;

  /// LLT 分解失败次数（协方差退化的直接指标，正常应恒为 0）
  long llt_failure_count = 0;
  /// predict() 调用累计次数。诊断用：实测出现过"整个滤波器实例生命周期内
  /// predict 一次都没被调用"的情况（点云回调阻塞 3~4s 所致），而 correct 仍在跑，
  /// 导致四元数模长无阻尼塌缩。没有这个计数只能靠间接推断。
  long predict_count = 0;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};

  }
}


#endif
