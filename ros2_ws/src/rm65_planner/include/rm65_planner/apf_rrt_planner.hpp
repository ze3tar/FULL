#pragma once

#include <array>
#include <vector>
#include <random>
#include <Eigen/Geometry>

namespace rm65_planner
{
struct Sphere
{
  Eigen::Vector3d center;
  double radius;
};

struct Cylinder
{
  double radius;
  double length;
};

class APFRRTPlanner
{
public:
  APFRRTPlanner();

  std::vector<std::array<double, 6>> plan(
    const std::array<double, 6> & start, const std::array<double, 6> & goal,
    const std::vector<Sphere> & obstacles);

private:
  std::array<double, 6> steer(
    const std::array<double, 6> & from, const std::array<double, 6> & to,
    double step_size) const;
  double attractive(const std::array<double, 6> & q, const std::array<double, 6> & goal) const;
  double repulsive(const std::array<double, 6> & q, const std::vector<Sphere> & obstacles) const;
  bool is_collision_free(const std::array<double, 6> & q, const std::vector<Sphere> & obstacles) const;

  std::vector<Cylinder> links_;
  std::mt19937 rng_;
};

}  // namespace rm65_planner
