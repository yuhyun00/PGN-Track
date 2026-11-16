# Adapted from ByteTrack (MIT) / BoT-SORT (MIT) as used by McByte (MIT): constant-velocity Kalman filter on
# (cx, cy, w, h), plus a batched 2-D position update with a free measurement covariance (PGN-Track's bridge).
import numpy as np
import scipy.linalg


class KalmanFilter:
    """State (cx, cy, w, h, vcx, vcy, vw, vh); noise proportional to the box size."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """New track from an (cx, cy, w, h) measurement."""
        mean = np.r_[measurement, np.zeros_like(measurement)]
        w, h = measurement[2], measurement[3]
        std = [2 * self._std_weight_position * w, 2 * self._std_weight_position * h,
               2 * self._std_weight_position * w, 2 * self._std_weight_position * h,
               10 * self._std_weight_velocity * w, 10 * self._std_weight_velocity * h,
               10 * self._std_weight_velocity * w, 10 * self._std_weight_velocity * h]
        return mean, np.diag(np.square(std))

    def project(self, mean, covariance):
        """Measurement-space mean and covariance (detection noise = box size / 20)."""
        std = [self._std_weight_position * mean[2], self._std_weight_position * mean[3],
               self._std_weight_position * mean[2], self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Batched prediction: mean (n, 8), covariance (n, 8, 8)."""
        std_pos = [self._std_weight_position * mean[:, 2], self._std_weight_position * mean[:, 3],
                   self._std_weight_position * mean[:, 2], self._std_weight_position * mean[:, 3]]
        std_vel = [self._std_weight_velocity * mean[:, 2], self._std_weight_velocity * mean[:, 3],
                   self._std_weight_velocity * mean[:, 2], self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = np.zeros((len(mean), 8, 8))
        idx = np.arange(8)
        motion_cov[:, idx, idx] = sqr
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Update with an (cx, cy, w, h) detection."""
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    @staticmethod
    def multi_update_position(mean, covariance, z, R):
        """Batched update with a centre measurement z ~ N(H x, R), H = [I_2 0]; shapes (K,8), (K,8,8), (K,2), (K,2,2)."""
        PHt = covariance[:, :, :2]
        S = covariance[:, :2, :2] + R
        det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
        Sinv = np.empty_like(S)
        Sinv[:, 0, 0] = S[:, 1, 1] / det
        Sinv[:, 1, 1] = S[:, 0, 0] / det
        Sinv[:, 0, 1] = -S[:, 0, 1] / det
        Sinv[:, 1, 0] = -S[:, 1, 0] / det
        G = PHt @ Sinv
        innov = z - mean[:, :2]
        new_mean = mean + np.einsum('kij,kj->ki', G, innov)
        new_cov = covariance - G @ np.transpose(PHt, (0, 2, 1))
        new_cov = 0.5 * (new_cov + np.transpose(new_cov, (0, 2, 1)))
        return new_mean, new_cov
