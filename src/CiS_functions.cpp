//#include <Rcpp.h>
#include <RcppArmadillo.h>
#include "TDVS_header.h"
#include <random>
#include <chrono>
#include <numeric>
#include <thread>
#include <functional> // for std::hash
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
//
// [[Rcpp::export]]
arma::vec slope_error_cpp(const arma::vec& err, double nu_slo, double ga_slo) {
  arma::vec err_index = arma::conv_to<arma::vec>::from(err >= 0) * (-1) + arma::conv_to<arma::vec>::from(err < 0);
  arma::vec ga_pow = arma::pow(arma::vec(err_index.n_elem).fill(ga_slo), 2 * err_index);

  double const_term = -2 * ga_slo / (ga_slo * ga_slo + 1) *
    R::gammafn(nu_slo / 2 + 0.5) / (R::gammafn(nu_slo / 2) * std::sqrt(M_PI * nu_slo)) *
    (nu_slo + 1) / nu_slo;

  arma::vec denom = arma::pow(1 + arma::square(err) / nu_slo % ga_pow, (nu_slo + 3) / 2);
  return const_term * ga_pow % err / denom;
}

// [[Rcpp::export]]
arma::vec curvature_error_cpp(const arma::vec& err, double nu_cur, double ga_cur) {
  arma::vec err_index = arma::conv_to<arma::vec>::from(err >= 0) * (-1) + arma::conv_to<arma::vec>::from(err < 0);
  arma::vec ga_pow = arma::pow(arma::vec(err_index.n_elem).fill(ga_cur), 2 * err_index);

  double const_term = 2 * ga_cur / (ga_cur * ga_cur + 1) *
    R::gammafn(nu_cur / 2 + 0.5) / (R::gammafn(nu_cur / 2) * std::sqrt(M_PI * nu_cur)) *
    (nu_cur + 1) / nu_cur;

  arma::vec one_plus_term = 1 + arma::square(err) / nu_cur % ga_pow;
  arma::vec inner = ((nu_cur + 3) / nu_cur) * ga_pow % arma::square(err) / one_plus_term - 1;

  return const_term * ga_pow / arma::pow(one_plus_term, (nu_cur + 3) / 2) % inner;
}

// [[Rcpp::export]]
double CiS_j_fun_cpp(int test_index,
                     const arma::vec& beta_opt,
                     double beta0_opt,
                     double sigma_opt,
                     double nu_opt,
                     double ga_opt,
                     Rcpp::List dataXY,
                     double add_correc_CiS) {

  // Extract data of Y and X
  arma::vec dat_Y = dataXY["Y"];
  arma::mat dat_X = dataXY["X"];

  // Check if test_index is within valid bounds (1-based to 0-based conversion)
  if (test_index < 1 || test_index > static_cast<int>(beta_opt.n_elem)) {
    Rcpp::stop(
      "Invalid test_index (requested %d, but coefficient vector has length %d). "
      "Must be between 1 and %d.",
      test_index, beta_opt.n_elem, beta_opt.n_elem
    );
  }

  arma::vec err_full = (dat_Y - beta0_opt - dat_X * beta_opt)/sigma_opt;
  arma::vec S_Prime_obs = arma::square(slope_error_cpp(err_full, ga_opt, nu_opt));

  //Rcpp::Rcout << "err_full: " << err_full << std::endl;

  // Create a mask to exclude the test_index-th column (0-based index)
  arma::uvec all_cols = arma::regspace<arma::uvec>(0, dat_X.n_cols - 1);
  arma::uvec reduced_cols = arma::find(all_cols != (test_index-1));  // Exclude column j

  //Rcpp::Rcout << "reduced_cols: " << reduced_cols << std::endl;

  arma::vec err_reduced = (dat_Y - beta0_opt - dat_X.cols(reduced_cols) * beta_opt.elem(reduced_cols))/sigma_opt;
  arma::vec S_j_obs = arma::square(slope_error_cpp(err_reduced, ga_opt, nu_opt));
  arma::vec Delta_j_obs = arma::abs(S_j_obs - S_Prime_obs);

  //Rcpp::Rcout << "err_reduced: " << err_reduced << std::endl;

  arma::vec C_j_obs = arma::abs(curvature_error_cpp(err_reduced, ga_opt, nu_opt));
  arma::vec CiS_j_obs_i = Delta_j_obs / (C_j_obs + add_correc_CiS);

  return arma::mean(CiS_j_obs_i);
}

// [[Rcpp::export]]
double per_fun_cpp(int j_index,
                   Rcpp::List dataXY,
                   arma::vec init_beta_per,
                   double init_beta0_per,
                   double init_sigma_per,
                   double init_nu_per,
                   double init_gamma_per,
                   double init_theta_per,
                   double SS_t0_per,
                   double SS_t1_per,
                   double hyper_mu_beta0_per,
                   double hyper_sigma_beta0_per,
                   double hyper_nu_sigma_per,
                   double hyper_A_sigma_per,
                   double hyper_mu_nu_per,
                   double hyper_sigma_nu_per,
                   double hyper_c_gamma_per,
                   double hyper_d_gamma_per,
                   double hyper_a_theta_per,
                   double hyper_b_theta_per,
                   int max_iter_per,
                   double tol_per,
                   double add_correc_CiS) {

  arma::vec dat_Y = dataXY["Y"];
  arma::mat dat_X = dataXY["X"];
  int p = dat_X.n_cols;
  int n = dat_X.n_rows;

  // Check if hyper_b_theta was provided
  double hyper_b_theta_val = (hyper_b_theta_per < 0) ? p : hyper_b_theta_per;

  // Check if test_index is within valid bounds (1-based to 0-based conversion)
  if (j_index < 1 || j_index > p) {
    Rcpp::stop(
      "Invalid j_index (requested %d, but coefficient vector has length %d). "
      "Must be between 1 and %d.",
      j_index, p, p
    );
  }

  // Ensure R RNG environment
  Rcpp::RNGScope scope;

  // Set up C++ random number generator with time-based seed
  // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  unsigned long time_seed = static_cast<unsigned long>(duration.count());

  std::hash<std::thread::id> hasher;
  size_t thread_hash = hasher(std::this_thread::get_id());

  unsigned final_seed = static_cast<unsigned>(time_seed ^ thread_hash);  // Combine time and thread ID
  std::mt19937 generator(final_seed);

//  Rcpp::Rcout << "C++ seed: " << final_seed << "\n";

  // Convert 1-based to 0-based
  arma::uword j_index_cpp = j_index - 1;

  // Create a vector of indices and shuffle it
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n-1
  std::shuffle(indices.begin(), indices.end(), generator);
  // Convert to arma::uvec correctly
  arma::uvec arma_idx = arma::conv_to<arma::uvec>::from(arma::Col<int>(indices.data(), indices.size()));
  // permute the j-th column
  arma::vec col_j = dat_X.col(j_index_cpp);  // safe copy
  col_j = col_j.elem(arma_idx);              // shuffle
  dat_X.col(j_index_cpp) = col_j;

  Rcpp::List dat_per = Rcpp::List::create(
    Rcpp::Named("Y") = dat_Y,
    Rcpp::Named("X") = dat_X
  );

  Rcpp::List em_results_per_j = TDVS_EM_cpp(dat_per, init_beta_per, init_beta0_per, init_sigma_per, init_nu_per, init_gamma_per, init_theta_per,
                                           SS_t0_per, SS_t1_per, hyper_mu_beta0_per, hyper_sigma_beta0_per, hyper_nu_sigma_per, hyper_A_sigma_per,
                                           hyper_mu_nu_per, hyper_sigma_nu_per, hyper_c_gamma_per, hyper_d_gamma_per,
                                           hyper_a_theta_per, hyper_b_theta_val, max_iter_per, tol_per);

  arma::vec beta_per_j = em_results_per_j["beta"];
  double beta0_per_j = em_results_per_j["beta0"];
  double sigma_per_j = em_results_per_j["sigma"];
  double nu_per_j = em_results_per_j["nu"];
  double ga_per_j = em_results_per_j["gamma"];

  double CiS_j_per = CiS_j_fun_cpp(j_index, beta_per_j, beta0_per_j, sigma_per_j, nu_per_j, ga_per_j, dat_per, add_correc_CiS);
  return CiS_j_per;
}


// [[Rcpp::export]]
double CiS_group_fun_cpp(const arma::uvec& test_indices,
                         const arma::vec& beta_opt,
                         double beta0_opt,
                         double sigma_opt,
                         double nu_opt,
                         double ga_opt,
                         Rcpp::List dataXY,
                         double add_correc_CiS) {

  // Extract data
  arma::vec dat_Y = dataXY["Y"];
  arma::mat dat_X = dataXY["X"];
  int p = dat_X.n_cols;

  // Validate test_indices
  if (arma::any(test_indices < 1) || arma::any(test_indices > p)) {
    Rcpp::stop("Invalid index in test_indices. Must be in range 1 to %d.", p);
  }

  // Convert to 0-based indices
  arma::uvec zero_based = test_indices - 1;
  //Rcpp::Rcout << "zero_based: " << zero_based << std::endl;

  // Compute full model errors and slope
  arma::vec err_full = (dat_Y - beta0_opt - dat_X * beta_opt)/sigma_opt;
  arma::vec S_Prime_obs = arma::square(slope_error_cpp(err_full, ga_opt, nu_opt));

  // Determine columns to keep (exclude test_indices)
  arma::uvec all_cols = arma::regspace<arma::uvec>(0, p - 1);

  // Set difference: keep all columns NOT in zero_based
  arma::uvec keep_cols = arma::zeros<arma::uvec>(p - zero_based.n_elem);
  int k = 0;
  for (int i = 0; i < p; ++i) {
    if (!arma::any(zero_based == i)) {
      keep_cols(k++) = i;
    }
  }

  //Rcpp::Rcout << "keep_cols: " << keep_cols << std::endl;

  arma::vec err_reduced = (dat_Y - beta0_opt - dat_X.cols(keep_cols) * beta_opt.elem(keep_cols))/sigma_opt;
  arma::vec S_j_obs = arma::square(slope_error_cpp(err_reduced, ga_opt, nu_opt));
  arma::vec Delta_j_obs = arma::abs(S_j_obs - S_Prime_obs);

  arma::vec C_j_obs = arma::abs(curvature_error_cpp(err_reduced, ga_opt, nu_opt));
  arma::vec CiS_j_obs_i = Delta_j_obs / (C_j_obs + add_correc_CiS);

  return arma::mean(CiS_j_obs_i);
}

// [[Rcpp::export]]
double per_group_fun_cpp(const arma::uvec& j_indices,
                         Rcpp::List dataXY,
                         arma::vec init_beta_per,
                         double init_beta0_per,
                         double init_sigma_per,
                         double init_nu_per,
                         double init_gamma_per,
                         double init_theta_per,
                         double SS_t0_per,
                         double SS_t1_per,
                         double hyper_mu_beta0_per,
                         double hyper_sigma_beta0_per,
                         double hyper_nu_sigma_per,
                         double hyper_A_sigma_per,
                         double hyper_mu_nu_per,
                         double hyper_sigma_nu_per,
                         double hyper_c_gamma_per,
                         double hyper_d_gamma_per,
                         double hyper_a_theta_per,
                         double hyper_b_theta_per,
                         int max_iter_per,
                         double tol_per,
                         double add_correc_CiS) {

  arma::vec dat_Y = dataXY["Y"];
  arma::mat dat_X = dataXY["X"];
  int p = dat_X.n_cols;

  // Validate test indices (1-based)
  if (arma::any(j_indices < 1) || arma::any(j_indices > p)) {
    Rcpp::stop("Invalid index in j_indices. Must be in range 1 to %d.", p);
  }
  // Check if hyper_b_theta was provided
  double hyper_b_theta_val = (hyper_b_theta_per < 0) ? p : hyper_b_theta_per;

  // Convert to 0-based indices
  arma::uvec zero_based = j_indices - 1;

  // Shuffle each selected column independently
  for (arma::uword col_idx : zero_based) {
    dat_X.col(col_idx) = arma::shuffle(dat_X.col(col_idx));
    }

  // Create permuted dataset
  Rcpp::List dat_per = Rcpp::List::create(
    Rcpp::Named("Y") = dat_Y,
    Rcpp::Named("X") = dat_X
  );

   double u_before = R::runif(0.0, 1.0);

  // Run E-M algorithm
  Rcpp::List em_results = TDVS_EM_cpp(dat_per, init_beta_per, init_beta0_per, init_sigma_per, init_nu_per, init_gamma_per, init_theta_per,
                                      SS_t0_per, SS_t1_per, hyper_mu_beta0_per, hyper_sigma_beta0_per, hyper_nu_sigma_per, hyper_A_sigma_per,
                                      hyper_mu_nu_per, hyper_sigma_nu_per, hyper_c_gamma_per, hyper_d_gamma_per,
                                      hyper_a_theta_per, hyper_b_theta_val, max_iter_per, tol_per);
  arma::vec beta_hat = em_results["beta"];
  double beta0_hat = em_results["beta0"];
  double sigma_hat = em_results["sigma"];
  double nu_hat = em_results["nu"];
  double ga_hat = em_results["gamma"];

   double u_after = R::runif(0.0, 1.0);

  // Evaluate the group effect
  double CiS_group_per = CiS_group_fun_cpp(j_indices, beta_hat, beta0_hat, sigma_hat, nu_hat, ga_hat, dat_per, add_correc_CiS);

  return CiS_group_per;
}

