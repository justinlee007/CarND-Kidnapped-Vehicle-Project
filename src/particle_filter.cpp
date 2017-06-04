/**
 * particle_filter.cpp
 */

#include <random>
#include <sstream>

#include "particle_filter.h"

random_device device_;
default_random_engine engine_(device_());
normal_distribution<float> x_dist_;
normal_distribution<float> y_dist_;
normal_distribution<float> theta_dist_;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  printf("x=%f, y=%f, theta=%f\n", x, y, theta);

  num_particles = 100;

  // Initialize GPS measurement uncertainty distribution based on std[] values
  x_dist_ = normal_distribution<float>(0.0f, (float) std[0]);
  y_dist_ = normal_distribution<float>(0.0f, (float) std[1]);
  theta_dist_ = normal_distribution<float>(0.0f, (float) std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = x + x_dist_(engine_);
    particle.y = y + y_dist_(engine_);
    particle.theta = theta + theta_dist_(engine_);
    particle.weight = 1.0f;
    weights.push_back(1.0f);
    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  printf("velocity=%f, yaw_rate=%f\n", velocity, yaw_rate);
  for (int i = 0; i < num_particles; i++) {

    double theta = particles[i].theta;

    if (fabs(yaw_rate) > 0.001) {
      particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta)) + x_dist_(engine_);
      particles[i].y += -velocity / yaw_rate * (cos(theta + yaw_rate * delta_t) - cos(theta)) + y_dist_(engine_);
    } else {
      particles[i].x += velocity * cos(theta) * delta_t + x_dist_(engine_);
      particles[i].y += velocity * sin(theta) * delta_t + y_dist_(engine_);
    }
    particles[i].theta += yaw_rate * delta_t + theta_dist_(engine_);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 3.33
  //   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, vector<int> associations, vector<double> sense_x, vector<double> sense_y) {
  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
