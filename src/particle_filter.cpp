/**
 * particle_filter.cpp
 */

#include <random>
#include <sstream>

#include "particle_filter.h"

static const int NUM_PARTICLES = 100;
static const double DEFAULT_WEIGHT = 1.0f;

// GPS measurement uncertainty [x [m], y [m], theta [rad]]
static normal_distribution<float> x_dist_;
static normal_distribution<float> y_dist_;
static normal_distribution<float> theta_dist_;

// Random number generator
static default_random_engine engine_(random_device{}());

/*
* Calculates the bivariate normal pdf of a point given a mean and std and assuming zero correlation
*/
double bivariate_normal(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
  return exp(-((x - mu_x) * (x - mu_x) / (2 * sig_x * sig_x) + (y - mu_y) * (y - mu_y) / (2 * sig_y * sig_y))) / (2.0 * 3.14159 * sig_x * sig_y);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  printf("x=%f, y=%f, theta=%f\n", x, y, theta);

  num_particles = NUM_PARTICLES;

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
    particle.weight = DEFAULT_WEIGHT;
    particles.push_back(particle);
    weights.push_back(DEFAULT_WEIGHT);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  for (int i = 0; i < num_particles; i++) {
    double theta = particles[i].theta;

    // Avoid dividing by zero
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
  double dist_po;
  for (int i = 0; i < observations.size(); i++) { // loop over observations
    double min_dist = 1000000;
    int closest_id = -1;
    for (int j = 0; j < predicted.size(); j++) {
      dist_po = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (dist_po < min_dist) {
        min_dist = dist_po;
        closest_id = predicted[j].id;
      }
    }
    observations[i].id = closest_id;
  }
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

  weights.clear();

  for (int i = 0; i < particles.size(); i++) {
    vector<LandmarkObs> ground_obs;
    vector<LandmarkObs> predicted_obs;

    // Convert observations to ground frame
    for (int index = 0; index < observations.size(); index++) {
      LandmarkObs obs;
      obs.x = 0;
      obs.x += observations[index].x * cos(particles[i].theta);
      obs.x += -observations[index].y * sin(particles[i].theta);
      obs.x += particles[i].x;

      obs.y = 0;
      obs.y += observations[index].x * sin(particles[i].theta);
      obs.y += observations[index].y * cos(particles[i].theta);
      obs.y += particles[i].y;

      obs.id = -1; // Temporary ID.
      ground_obs.push_back(obs);
    }

    // Compute predicted observations
    for (int index = 0; index < map_landmarks.landmark_list.size(); index++) {
      double particle_distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[index].x_f, map_landmarks.landmark_list[index].y_f);
      if (particle_distance <= sensor_range) {
        LandmarkObs obs;
        obs.id = map_landmarks.landmark_list[index].id_i;
        obs.x = map_landmarks.landmark_list[index].x_f;
        obs.y = map_landmarks.landmark_list[index].y_f;

        predicted_obs.push_back(obs);
      }
    }

    dataAssociation(predicted_obs, ground_obs);

    double prob = DEFAULT_WEIGHT;
    double prob_i;

    for (int index = 0; index < predicted_obs.size(); index++) {
      int ind_min = -1;
      double dist_min = 1000000;

      for (int j = 0; j < ground_obs.size(); j++) {
        //Use measurement closest to predicted in case of multiple measurements assigned to the same observation
        if (predicted_obs[index].id == ground_obs[j].id) {
          double check_dist = dist(predicted_obs[index].x, predicted_obs[index].y, ground_obs[j].x, ground_obs[j].y);
          if (check_dist < dist_min) {
            ind_min = j;
            dist_min = check_dist;
          }
        }
      }
      if (ind_min != -1) {
        prob_i = bivariate_normal(predicted_obs[index].x, predicted_obs[index].y, ground_obs[ind_min].x, ground_obs[ind_min].y, std_landmark[0], std_landmark[1]);
        prob = prob * prob_i;
      }
    }

    weights.push_back(prob);
    particles[i].weight = prob;
  }
}

void ParticleFilter::resample() {
  // Initialize weight distribution for resampling
  discrete_distribution<int> weight_dist(weights.begin(), weights.end());
  vector<Particle> resampled_particles;

  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[weight_dist(engine_)];
    resampled_particles.push_back(particle);
  }
  particles = resampled_particles;
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
