/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifndef FLATNESS_HPP
#define FLATNESS_HPP

#include <Eigen/Eigen>
#include <cmath>

struct InertialAndDragParams {
  double mass{1};
  double grav{-9.81};
  double dh{1};
  double dv{1};
  double cp{1};
  double veps{0.01};
};

struct InertialParams {
  double mass{1};
  double grav{-9.81};
  double veps{0.01};
};

inline void Forward(const Eigen::Vector3d &vel, const Eigen::Vector3d &acc,
                    const Eigen::Vector3d &jer, double psi, double dpsi,
                    double &thr, Eigen::Quaterniond &quat, Eigen::Vector3d &omg,
                    const InertialAndDragParams &params = {}) {
  static const Eigen::Vector3d kGravVector =
      Eigen::Vector3d::UnitZ() * params.grav;
  // Parasitic drag
  const double cp_term = sqrt(vel.squaredNorm() + params.veps);
  const double w_term = 1.0 + params.cp * cp_term;
  const double v_dot_a = vel.dot(acc);
  const double dw_term = params.cp * v_dot_a / cp_term;

  auto dw = w_term * acc + dw_term * vel;
  const Eigen::Vector3d w = w_term * vel;
  const double dh_over_m = params.dh / params.mass;
  Eigen::Vector3d z = acc + dh_over_m * w + kGravVector;
  const double z_nrm = z.norm();
  z /= z_nrm;

  const Eigen::Vector3d dz = -z.cross(z.cross(jer + dh_over_m * dw)) / z_nrm;

  thr = z.dot(params.mass * (acc + kGravVector) + params.dv * w);
  const double tilt_den = sqrt(2.0 * (1.0 + z.z()));
  const double tilt0 = 0.5 * tilt_den;
  const double tilt1 = -z.y() / tilt_den;
  const double tilt2 = z.x() / tilt_den;
  const double c_half_psi = cos(0.5 * psi);
  const double s_half_psi = sin(0.5 * psi);
  quat.x() = tilt1 * c_half_psi + tilt2 * s_half_psi;
  quat.y() = tilt2 * c_half_psi - tilt1 * s_half_psi;
  quat.z() = tilt0 * s_half_psi;
  quat.w() = tilt0 * c_half_psi;
  const double c_psi = cos(psi);
  const double s_psi = sin(psi);
  const double omg_den = z[2] + 1.0;
  const double omg_term = dz.z() / omg_den;
  omg(0) =
      dz[0] * s_psi - dz[1] * c_psi - (z[0] * s_psi - z[1] * c_psi) * omg_term;
  omg(1) =
      dz[0] * c_psi + dz[1] * s_psi - (z[0] * c_psi + z[1] * s_psi) * omg_term;
  omg(2) = (z[1] * dz[0] - z[0] * dz[1]) / omg_den + dpsi;

  return;
}

inline void Forward(const Eigen::Vector3d &vel, const Eigen::Vector3d &acc,
                    const Eigen::Vector3d &jer, double psi, double dpsi,
                    double &thr, Eigen::Quaterniond &quat, Eigen::Vector3d &omg,
                    const InertialParams &params = {}) {
  static const Eigen::Vector3d kGravVector =
      Eigen::Vector3d::UnitZ() * params.grav;
  // Parasitic drag
  const double v_dot_a = vel.dot(acc);

  Eigen::Vector3d z = acc + kGravVector;
  const double z_nrm = z.norm();
  z /= z_nrm;

  const Eigen::Vector3d dz = -z.cross(z.cross(jer)) / z_nrm;

  thr = z.dot(params.mass * (acc + kGravVector));
  const double tilt_den = sqrt(2.0 * (1.0 + z.z()));
  const double tilt0 = 0.5 * tilt_den;
  const double tilt1 = -z.y() / tilt_den;
  const double tilt2 = z.x() / tilt_den;
  const double c_half_psi = cos(0.5 * psi);
  const double s_half_psi = sin(0.5 * psi);
  quat.x() = tilt1 * c_half_psi + tilt2 * s_half_psi;
  quat.y() = tilt2 * c_half_psi - tilt1 * s_half_psi;
  quat.z() = tilt0 * s_half_psi;
  quat.w() = tilt0 * c_half_psi;
  const double c_psi = cos(psi);
  const double s_psi = sin(psi);
  const double omg_den = z[2] + 1.0;
  const double omg_term = dz.z() / omg_den;
  omg(0) =
      dz[0] * s_psi - dz[1] * c_psi - (z[0] * s_psi - z[1] * c_psi) * omg_term;
  omg(1) =
      dz[0] * c_psi + dz[1] * s_psi - (z[0] * c_psi + z[1] * s_psi) * omg_term;
  omg(2) = (z[1] * dz[0] - z[0] * dz[1]) / omg_den + dpsi;

  return;
}

namespace flatness {
class
    FlatnessMap  // See
                 // https://github.com/ZJU-FAST-Lab/GCOPTER/blob/main/misc/flatness.pdf
{
 public:
  inline void reset(double vehicle_mass, double gravitational_acceleration,
                    double horitonral_drag_coeff, double vertical_drag_coeff,
                    double parasitic_drag_coeff, double speed_smooth_factor) {
    mass = vehicle_mass;
    grav = gravitational_acceleration;
    dh = horitonral_drag_coeff;
    dv = vertical_drag_coeff;
    cp = parasitic_drag_coeff;
    veps = speed_smooth_factor;

    return;
  }

  inline void forward(const Eigen::Vector3d &vel, const Eigen::Vector3d &acc,
                      const Eigen::Vector3d &jer, double psi, double dpsi,
                      double &thr, Eigen::Quaterniond &quat,
                      Eigen::Vector3d &omg) {
    double v0, v1, v2, a0, a1, a2, v_dot_a;
    double z0, z1, z2, dz0, dz1, dz2;
    double cp_term, w_term, dh_over_m;
    double zu_sqr_norm, zu_norm, zu0, zu1, zu2;
    double zu_sqr0, zu_sqr1, zu_sqr2, zu01, zu12, zu02;
    double ng00, ng01, ng02, ng11, ng12, ng22, ng_den;
    double dw_term, dz_term0, dz_term1, dz_term2, f_term0, f_term1, f_term2;
    double tilt_den, tilt0, tilt1, tilt2, c_half_psi, s_half_psi;
    double c_psi, s_psi, omg_den, omg_term;
    double w0, w1, w2, dw0, dw1, dw2;

    v0 = vel(0);
    v1 = vel(1);
    v2 = vel(2);
    a0 = acc(0);
    a1 = acc(1);
    a2 = acc(2);
    cp_term = sqrt(v0 * v0 + v1 * v1 + v2 * v2 + veps);
    w_term = 1.0 + cp * cp_term;
    w0 = w_term * v0;
    w1 = w_term * v1;
    w2 = w_term * v2;
    dh_over_m = dh / mass;
    zu0 = a0 + dh_over_m * w0;
    zu1 = a1 + dh_over_m * w1;
    zu2 = a2 + dh_over_m * w2 + grav;
    zu_sqr0 = zu0 * zu0;
    zu_sqr1 = zu1 * zu1;
    zu_sqr2 = zu2 * zu2;
    zu01 = zu0 * zu1;
    zu12 = zu1 * zu2;
    zu02 = zu0 * zu2;
    zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2;
    zu_norm = sqrt(zu_sqr_norm);
    z0 = zu0 / zu_norm;
    z1 = zu1 / zu_norm;
    z2 = zu2 / zu_norm;
    ng_den = zu_sqr_norm * zu_norm;
    ng00 = (zu_sqr1 + zu_sqr2) / ng_den;
    ng01 = -zu01 / ng_den;
    ng02 = -zu02 / ng_den;
    ng11 = (zu_sqr0 + zu_sqr2) / ng_den;
    ng12 = -zu12 / ng_den;
    ng22 = (zu_sqr0 + zu_sqr1) / ng_den;
    v_dot_a = v0 * a0 + v1 * a1 + v2 * a2;
    dw_term = cp * v_dot_a / cp_term;
    dw0 = w_term * a0 + dw_term * v0;
    dw1 = w_term * a1 + dw_term * v1;
    dw2 = w_term * a2 + dw_term * v2;
    dz_term0 = jer(0) + dh_over_m * dw0;
    dz_term1 = jer(1) + dh_over_m * dw1;
    dz_term2 = jer(2) + dh_over_m * dw2;
    dz0 = ng00 * dz_term0 + ng01 * dz_term1 + ng02 * dz_term2;
    dz1 = ng01 * dz_term0 + ng11 * dz_term1 + ng12 * dz_term2;
    dz2 = ng02 * dz_term0 + ng12 * dz_term1 + ng22 * dz_term2;
    f_term0 = mass * a0 + dv * w0;
    f_term1 = mass * a1 + dv * w1;
    f_term2 = mass * (a2 + grav) + dv * w2;
    thr = z0 * f_term0 + z1 * f_term1 + z2 * f_term2;
    tilt_den = sqrt(2.0 * (1.0 + z2));
    tilt0 = 0.5 * tilt_den;
    tilt1 = -z1 / tilt_den;
    tilt2 = z0 / tilt_den;
    c_half_psi = cos(0.5 * psi);
    s_half_psi = sin(0.5 * psi);
    quat = Eigen::Quaterniond(tilt0 * c_half_psi,                       // w
                              tilt1 * c_half_psi + tilt2 * s_half_psi,  // x
                              tilt2 * c_half_psi - tilt1 * s_half_psi,  // y
                              tilt0 * s_half_psi);
    c_psi = cos(psi);
    s_psi = sin(psi);
    omg_den = z2 + 1.0;
    omg_term = dz2 / omg_den;
    omg(0) = dz0 * s_psi - dz1 * c_psi - (z0 * s_psi - z1 * c_psi) * omg_term;
    omg(1) = dz0 * c_psi + dz1 * s_psi - (z0 * c_psi + z1 * s_psi) * omg_term;
    omg(2) = (z1 * dz0 - z0 * dz1) / omg_den + dpsi;

    return;
  }

  inline void newForward(const Eigen::Vector3d &vel, const Eigen::Vector3d &acc,
                         const Eigen::Vector3d &jer, double psi, double dpsi,
                         double &thr, Eigen::Quaterniond &quat,
                         Eigen::Vector3d &omg) {
    static const Eigen::Vector3d kGravVector = Eigen::Vector3d::UnitZ() * grav;
    // Parasitic drag
    const double cp_term = sqrt(vel.squaredNorm() + veps);
    const double w_term = 1.0 + cp * cp_term;
    const double v_dot_a = vel.dot(acc);
    const double dw_term = cp * v_dot_a / cp_term;

    auto dw = w_term * acc + dw_term * vel;
    const Eigen::Vector3d w = w_term * vel;
    const double dh_over_m = dh / mass;
    const Eigen::Vector3d zu = acc + dh_over_m * w + kGravVector;
    const double z_nrm = zu.norm();
    const Eigen::Vector3d z = zu / z_nrm;

    const Eigen::Vector3d dz = -z.cross(z.cross(jer + dh_over_m * dw)) / z_nrm;

    thr = z.dot(mass * (acc + kGravVector) + dv * w);
    quat = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ());
    const double tilt_den = sqrt(2.0 * (1.0 + z.z()));
    quat *= Eigen::Quaterniond(0.5 * tilt_den, -z.y() / tilt_den,
                               z.x() / tilt_den, 0.0);

    const double c_psi = cos(psi);
    const double s_psi = sin(psi);
    const double omg_den = z[2] + 1.0;
    const double omg_term = dz.z() / omg_den;
    omg(0) = dz[0] * s_psi - dz[1] * c_psi -
             (z[0] * s_psi - z[1] * c_psi) * omg_term;
    omg(1) = dz[0] * c_psi + dz[1] * s_psi -
             (z[0] * c_psi + z[1] * s_psi) * omg_term;
    omg(2) = (z[1] * dz[0] - z[0] * dz[1]) / omg_den + dpsi;

    return;
  }

  // inline void backward(const Eigen::Vector3d &pos_grad,
  //                      const Eigen::Vector3d &vel_grad, const double
  //                      &thr_grad, const Eigen::Vector4d &quat_grad, const
  //                      Eigen::Vector3d &omg_grad, Eigen::Vector3d
  //                      &pos_total_grad, Eigen::Vector3d &vel_total_grad,
  //                      Eigen::Vector3d &acc_total_grad,
  //                      Eigen::Vector3d &jer_total_grad, double
  //                      &psi_total_grad, double &dpsi_total_grad) const {
  //   double v0, v1, v2, a0, a1, a2, v_dot_a;
  //   double z0, z1, z2, dz0, dz1, dz2;
  //   double cp_term, w_term, dh_over_m;
  //   double zu_sqr_norm, zu_norm, zu0, zu1, zu2;
  //   double zu_sqr0, zu_sqr1, zu_sqr2, zu01, zu12, zu02;
  //   double ng00, ng01, ng02, ng11, ng12, ng22, ng_den;
  //   double dw_term, dz_term0, dz_term1, dz_term2, f_term0, f_term1, f_term2;
  //   double tilt_den, tilt0, tilt1, tilt2, c_half_psi, s_half_psi;
  //   double c_psi, s_psi, omg_den, omg_term;
  //   double w0, w1, w2, dw0, dw1, dw2;

  //   tilt0b = s_half_psi * (quat_grad(3)) + c_half_psi * (quat_grad(0));
  //   head3b = tilt0 * (quat_grad(3)) + tilt2 * (quat_grad(1)) -
  //            tilt1 * (quat_grad(2));
  //   tilt2b = c_half_psi * (quat_grad(2)) + s_half_psi * (quat_grad(1));
  //   head0b = tilt2 * (quat_grad(2)) + tilt1 * (quat_grad(1)) +
  //            tilt0 * (quat_grad(0));
  //   tilt1b = c_half_psi * (quat_grad(1)) - s_half_psi * (quat_grad(2));
  //   tilt_den_sqr = tilt_den * tilt_den;
  //   tilt_denb = (z1 * tilt1b - z0 * tilt2b) / tilt_den_sqr + 0.5 * tilt0b;
  //   omg_termb = -((z0 * c_psi + z1 * s_psi) * (omg_grad(1))) -
  //               (z0 * s_psi - z1 * c_psi) * (omg_grad(0));
  //   tempb = omg_grad(2) / omg_den;
  //   dpsi_total_grad = omg_grad(2);
  //   z1b = dz0 * tempb;
  //   dz0b = z1 * tempb + c_psi * (omg_grad(1)) + s_psi * (omg_grad(0));
  //   z0b = -(dz1 * tempb);
  //   dz1b = s_psi * (omg_grad(1)) - z0 * tempb - c_psi * (omg_grad(0));
  //   omg_denb = -((z1 * dz0 - z0 * dz1) * tempb / omg_den) -
  //              dz2 * omg_termb / (omg_den * omg_den);
  //   tempb = -(omg_term * (omg_grad(1)));
  //   cpsib = dz0 * (omg_grad(1)) + z0 * tempb;
  //   spsib = dz1 * (omg_grad(1)) + z1 * tempb;
  //   z0b += c_psi * tempb;
  //   z1b += s_psi * tempb;
  //   tempb = -(omg_term * (omg_grad(0)));
  //   spsib += dz0 * (omg_grad(0)) + z0 * tempb;
  //   cpsib += -dz1 * (omg_grad(0)) - z1 * tempb;
  //   z0b += s_psi * tempb + tilt2b / tilt_den + f_term0 * (thr_grad);
  //   z1b += -c_psi * tempb - tilt1b / tilt_den + f_term1 * (thr_grad);
  //   dz2b = omg_termb / omg_den;
  //   z2b = omg_denb + tilt_denb / tilt_den + f_term2 * (thr_grad);
  //   psi_total_grad = c_psi * spsib + 0.5 * c_half_psi * head3b - s_psi *
  //   cpsib -
  //                    0.5 * s_half_psi * head0b;
  //   f_term0b = z0 * (thr_grad);
  //   f_term1b = z1 * (thr_grad);
  //   f_term2b = z2 * (thr_grad);
  //   ng02b = dz_term0 * dz2b + dz_term2 * dz0b;
  //   dz_term0b = ng02 * dz2b + ng01 * dz1b + ng00 * dz0b;
  //   ng12b = dz_term1 * dz2b + dz_term2 * dz1b;
  //   dz_term1b = ng12 * dz2b + ng11 * dz1b + ng01 * dz0b;
  //   ng22b = dz_term2 * dz2b;
  //   dz_term2b = ng22 * dz2b + ng12 * dz1b + ng02 * dz0b;
  //   ng01b = dz_term0 * dz1b + dz_term1 * dz0b;
  //   ng11b = dz_term1 * dz1b;
  //   ng00b = dz_term0 * dz0b;
  //   jer_total_grad(2) = dz_term2b;
  //   dw2b = dh_over_m * dz_term2b;
  //   jer_total_grad(1) = dz_term1b;
  //   dw1b = dh_over_m * dz_term1b;
  //   jer_total_grad(0) = dz_term0b;
  //   dw0b = dh_over_m * dz_term0b;
  //   tempb = cp * (v2 * dw2b + v1 * dw1b + v0 * dw0b) / cp_term;
  //   acc_total_grad(2) = mass * f_term2b + w_term * dw2b + v2 * tempb;
  //   acc_total_grad(1) = mass * f_term1b + w_term * dw1b + v1 * tempb;
  //   acc_total_grad(0) = mass * f_term0b + w_term * dw0b + v0 * tempb;
  //   vel_total_grad(2) = dw_term * dw2b + a2 * tempb;
  //   vel_total_grad(1) = dw_term * dw1b + a1 * tempb;
  //   vel_total_grad(0) = dw_term * dw0b + a0 * tempb;
  //   cp_termb = -(v_dot_a * tempb / cp_term);
  //   tempb = ng22b / ng_den;
  //   zu_sqr0b = tempb;
  //   zu_sqr1b = tempb;
  //   ng_denb = -((zu_sqr0 + zu_sqr1) * tempb / ng_den);
  //   zu12b = -(ng12b / ng_den);
  //   tempb = ng11b / ng_den;
  //   ng_denb +=
  //       zu12 * ng12b / (ng_den * ng_den) - (zu_sqr0 + zu_sqr2) * tempb /
  //       ng_den;
  //   zu_sqr0b += tempb;
  //   zu_sqr2b = tempb;
  //   zu02b = -(ng02b / ng_den);
  //   zu01b = -(ng01b / ng_den);
  //   tempb = ng00b / ng_den;
  //   ng_denb += zu02 * ng02b / (ng_den * ng_den) +
  //              zu01 * ng01b / (ng_den * ng_den) -
  //              (zu_sqr1 + zu_sqr2) * tempb / ng_den;
  //   zu_normb = zu_sqr_norm * ng_denb -
  //              (zu2 * z2b + zu1 * z1b + zu0 * z0b) / zu_sqr_norm;
  //   zu_sqr_normb = zu_norm * ng_denb + zu_normb / (2.0 * zu_norm);
  //   tempb += zu_sqr_normb;
  //   zu_sqr1b += tempb;
  //   zu_sqr2b += tempb;
  //   zu2b = z2b / zu_norm + zu0 * zu02b + zu1 * zu12b + 2 * zu2 * zu_sqr2b;
  //   w2b = dv * f_term2b + dh_over_m * zu2b;
  //   zu1b = z1b / zu_norm + zu2 * zu12b + zu0 * zu01b + 2 * zu1 * zu_sqr1b;
  //   w1b = dv * f_term1b + dh_over_m * zu1b;
  //   zu_sqr0b += zu_sqr_normb;
  //   zu0b = z0b / zu_norm + zu2 * zu02b + zu1 * zu01b + 2 * zu0 * zu_sqr0b;
  //   w0b = dv * f_term0b + dh_over_m * zu0b;
  //   w_termb =
  //       a2 * dw2b + a1 * dw1b + a0 * dw0b + v2 * w2b + v1 * w1b + v0 * w0b;
  //   acc_total_grad(2) += zu2b;
  //   acc_total_grad(1) += zu1b;
  //   acc_total_grad(0) += zu0b;
  //   cp_termb += cp * w_termb;
  //   v_sqr_normb = cp_termb / (2.0 * cp_term);
  //   vel_total_grad(2) += w_term * w2b + 2 * v2 * v_sqr_normb + vel_grad(2);
  //   vel_total_grad(1) += w_term * w1b + 2 * v1 * v_sqr_normb + vel_grad(1);
  //   vel_total_grad(0) += w_term * w0b + 2 * v0 * v_sqr_normb + vel_grad(0);
  //   pos_total_grad(2) = pos_grad(2);
  //   pos_total_grad(1) = pos_grad(1);
  //   pos_total_grad(0) = pos_grad(0);

  //   return;
  // }

 private:
  double mass{1};
  double grav{-9.81};
  double dh{1};
  double dv{1};
  double cp{1};
  double veps{0.01};
};
}  // namespace flatness

#endif
