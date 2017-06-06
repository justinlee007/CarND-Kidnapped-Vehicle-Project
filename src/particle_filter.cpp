/**
 * particle_filter.cpp
 */

#include <random>
#include <sstream>

#include "particle_filter.h"

using namespace std;

static const int NUM_PARTICLES = 1000;
static const double DEFAULT_WEIGHT = 1.0f;
static const double DEFAULT_DISTANCE = 10000;

// GPS measurement uncertainty [x [m], y [m], theta [rad]]
static normal_distribution<float> x_dist_;
static normal_distribution<float> y_dist_;
static normal_distribution<float> theta_dist_;

// Random number generator
static default_random_engine engine_(random_device{}());

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
  double theta;
  for (int i = 0; i < num_particles; i++) {
    theta = particles[i].theta;

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
  double obs_offset;
  double min_distance;
  int closest_id;
  for (int obs_idx = 0; obs_idx < observations.size(); obs_idx++) { // loop over observations
    min_distance = DEFAULT_DISTANCE;
    closest_id = -1;
    for (int pred_idx = 0; pred_idx < predicted.size(); pred_idx++) {
      obs_offset = dist(observations[obs_idx].x, observations[obs_idx].y, predicted[pred_idx].x, predicted[pred_idx].y);
      if (obs_offset < min_distance) {
        min_distance = obs_offset;
        closest_id = predicted[pred_idx].id;
      }
    }
    observations[obs_idx].id = closest_id;
  }
}

/**
 * Calculates the Multivariate-Gaussian probability based on the measurement, it's associated landmark, and the signal noise standard deviation.
 *
 * @param x Predicted x
 * @param y Predicted y
 * @param mu_x Landmark x
 * @param mu_y Landmark y
 * @param sigma_x Standard deviation of uncertainty for x
 * @param sigma_y Standard deviation of uncertainty for y
 * @return the Multivariate-Gaussian of two dimensions, x and y
 */
double calculate_gaussian(double x, double y, double mu_x, double mu_y, double sigma_x, double sigma_y) {
  return exp(-(pow(x - mu_x, 2) / (2.0 * pow(sigma_x, 2)) + pow(y - mu_y, 2) / (2.0 * pow(sigma_y, 2)))) / (2.0 * M_PI * sigma_x * sigma_y);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations, Map map_landmarks) {

  weights.clear();
  vector<LandmarkObs> predicted_obs;
  vector<LandmarkObs> ground_obs;
  LandmarkObs prediction;
  LandmarkObs ground_truth;
  int closest_idx;
  double particle_distance;
  double min_distance;
  double measurement_offset;
  double prob;
  double gaussian_prob;

  for (int i = 0; i < particles.size(); i++) {
    predicted_obs.clear();
    ground_obs.clear();

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
      particle_distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[index].x_f, map_landmarks.landmark_list[index].y_f);
      if (particle_distance <= sensor_range) {
        LandmarkObs obs;
        obs.id = map_landmarks.landmark_list[index].id_i;
        obs.x = map_landmarks.landmark_list[index].x_f;
        obs.y = map_landmarks.landmark_list[index].y_f;

        predicted_obs.push_back(obs);
      }
    }

    dataAssociation(predicted_obs, ground_obs);

    prob = DEFAULT_WEIGHT;

    for (int pred_idx = 0; pred_idx < predicted_obs.size(); pred_idx++) {
      closest_idx = -1;
      min_distance = DEFAULT_DISTANCE;

      for (int ground_idx = 0; ground_idx < ground_obs.size(); ground_idx++) {

        // Use the closest measurement for prediction in case multiple measurements are assigned to the same observation
        if (predicted_obs[pred_idx].id == ground_obs[ground_idx].id) {
          measurement_offset = dist(predicted_obs[pred_idx].x, predicted_obs[pred_idx].y, ground_obs[ground_idx].x, ground_obs[ground_idx].y);
          if (measurement_offset < min_distance) {
            closest_idx = ground_idx;
            min_distance = measurement_offset;
          }
        }
      }
      if (closest_idx != -1) {
        prediction = predicted_obs[pred_idx];
        ground_truth = ground_obs[closest_idx];
        gaussian_prob = calculate_gaussian(prediction.x, prediction.y, ground_truth.x, ground_truth.y, std_landmark[0], std_landmark[1]);
        prob *= gaussian_prob;
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
