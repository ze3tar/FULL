#include "rm65_planner/apf_rrt_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rm65_planner
{
APFRRTPlanner::APFRRTPlanner() : rng_(std::random_device{}())
{
  links_ = {{Cylinder{0.04, 0.22}, Cylinder{0.04, 0.25}, Cylinder{0.03, 0.18},
             Cylinder{0.03, 0.16}, Cylinder{0.025, 0.14}, Cylinder{0.025, 0.12}}};
}

std::array<double, 6> APFRRTPlanner::steer(
  const std::array<double, 6> & from, const std::array<double, 6> & to, double step_size) const
{
  std::array<double, 6> result = from;
  for (size_t i = 0; i < result.size(); ++i) {
    double delta = to[i] - from[i];
    result[i] += std::clamp(delta, -step_size, step_size);
  }
  return result;
}

double APFRRTPlanner::attractive(const std::array<double, 6> & q, const std::array<double, 6> & goal) const
{
  double sum = 0.0;
  for (size_t i = 0; i < q.size(); ++i) {
    sum += std::pow(q[i] - goal[i], 2);
  }
  return 0.5 * sum;
}

double APFRRTPlanner::repulsive(const std::array<double, 6> & q, const std::vector<Sphere> & obstacles) const
{
  // Placeholder: use joint-space distance to obstacle centers mapped to norm of joint vector.
  double norm_q = 0.0;
  for (double v : q) {
    norm_q += v * v;
  }
  norm_q = std::sqrt(norm_q);
  double repulse = 0.0;
  for (const auto & obs : obstacles) {
    repulse += 1.0 / (1.0 + obs.radius + norm_q);
  }
  return repulse;
}

bool APFRRTPlanner::is_collision_free(const std::array<double, 6> & q, const std::vector<Sphere> & obstacles) const
{
  // Minimal placeholder collision: simply ensure joint magnitudes are bounded and not too close to obstacles metric.
  double norm_q = 0.0;
  for (double v : q) {
    norm_q += v * v;
  }
  norm_q = std::sqrt(norm_q);
  for (const auto & obs : obstacles) {
    if (norm_q < obs.radius) {
      return false;
    }
  }
  return norm_q < 10.0;  // loose bound
}

std::vector<std::array<double, 6>> APFRRTPlanner::plan(
  const std::array<double, 6> & start, const std::array<double, 6> & goal, const std::vector<Sphere> & obstacles)
{
  std::vector<std::array<double, 6>> path;
  path.push_back(start);
  std::array<double, 6> current = start;

  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const double step = 0.1;
  for (int iter = 0; iter < 200 && attractive(current, goal) > 1e-3; ++iter) {
    std::array<double, 6> sample = goal;
    if (uni(rng_) > 0.2) {
      for (double & v : sample) {
        v = uni(rng_) * 2.0 * M_PI - M_PI;
      }
    }
    std::array<double, 6> guided = steer(current, sample, step + repulsive(current, obstacles));
    if (is_collision_free(guided, obstacles)) {
      current = guided;
      path.push_back(current);
    }
  }
  if (path.back() != goal) {
    path.push_back(goal);
  }
  return path;
}

}  // namespace rm65_planner
