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
Rcpp::List TDVS_EM_cpp(
    Rcpp::List dataXY,
    arma::vec init_beta,
    double init_beta0,
    double init_sigma,
    double init_nu,
    double init_gamma,
    double init_theta,
    double SS_t0,
    double SS_t1,
    double hyper_mu_beta0,
    double hyper_sigma_beta0,
    double hyper_nu_sigma,
    double hyper_A_sigma,
    double hyper_mu_nu,
    double hyper_sigma_nu,
    double hyper_c_gamma,
    double hyper_d_gamma,
    double hyper_a_theta,
    double hyper_b_theta,
    int max_iter,
    double tol,
    bool include_sigma,
    bool update_sigma,
    double fixed_sigma,
    std::string conv_type) {

  // Extract data of Y and X
  arma::vec dat_Y = dataXY["Y"];
  arma::mat dat_X = dataXY["X"];

  // Get n and p
  int n = dat_Y.n_elem;
  int p = dat_X.n_cols;

  if (!include_sigma) {
    update_sigma = false;
    fixed_sigma = 1.0;
    init_sigma = 1.0;
  }

  // initial values for parameters
  arma::vec beta_upd=init_beta;
  double beta0_upd = init_beta0;
  double sigma_upd = init_sigma;
  double nu_upd = init_nu;
  double gamma_upd = init_gamma;
  double theta_upd = init_theta;
  // parameters at the previous iterations
  arma::vec beta_pre = init_beta;
  arma::vec beta_pre_pre = init_beta;
  double beta0_pre = init_beta0;
  double sigma_pre = init_sigma;
  double nu_pre = init_nu;
  double gamma_pre = init_gamma;
  double theta_pre = init_theta;

  // Ensure beta_upd size matches X columns
  if (static_cast<int>(beta_upd.n_elem) != p) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Ensure beta_upd size matches X columns
  if (static_cast<int>(dat_X.n_rows) != n) {
    Rcpp::stop("Size of response does not match predictor dimensions");
  }

  // Check if hyper_b_theta was provided
  double hyper_b_theta_val = (hyper_b_theta < 0) ? p : hyper_b_theta;

  // Output history containers
  arma::mat beta_hist = arma::mat(p, max_iter).fill(arma::datum::nan);
  arma::vec beta0_hist(max_iter);
  arma::vec sigma_hist(max_iter);
  arma::vec nu_hist(max_iter);
  arma::vec gamma_hist(max_iter);
  arma::vec theta_hist(max_iter);

  int iter = 0;
  bool converged = false;
  double diff_val = 0.0;

  Rcpp::Environment pkg_env = Rcpp::Environment::namespace_env("TDVS");
  Rcpp::Function wrapper_beta = pkg_env["wrapper_beta"];
  Rcpp::Function wrapper_beta0 = pkg_env["wrapper_beta0"];
  Rcpp::Function wrapper_sigma = pkg_env["wrapper_sigma"];
  Rcpp::Function wrapper_nu = pkg_env["wrapper_nu"];
  Rcpp::Function wrapper_gamma = pkg_env["wrapper_gamma"];

  for (iter = 0; iter < max_iter; ++iter) {
    beta_pre_pre = beta_pre;
    beta_pre = beta_upd;
    beta0_pre = beta0_upd;
    sigma_pre = sigma_upd;
    nu_pre = nu_upd;
    gamma_pre = gamma_upd;
    theta_pre = theta_upd;

    // M-step
    try {
      // M-step: update parameters via R wrappers
      //Rcpp::Rcout << "Iteration " << iter + 1 << ": beta update\n";
      beta_upd = Rcpp::as<arma::vec>(wrapper_beta(beta_pre, beta0_pre, sigma_pre, nu_pre, gamma_pre, beta_pre_pre, SS_t0, SS_t1, dat_Y, dat_X, theta_pre));
      //Rcpp::Rcout << "Iteration " << iter + 1 << ": beta0 update\n";
      beta0_upd = Rcpp::as<double>(wrapper_beta0(beta0_pre, beta_upd, sigma_pre, nu_pre, gamma_pre, hyper_mu_beta0, hyper_sigma_beta0, dat_Y, dat_X));
      //Rcpp::Rcout << "Iteration " << iter + 1 << ": sigma update\n";
      if (update_sigma) {
        sigma_upd = Rcpp::as<double>(wrapper_sigma(sigma_pre, beta_upd, beta0_upd, nu_pre, gamma_pre, hyper_nu_sigma, hyper_A_sigma, dat_Y, dat_X));
      } else {
        sigma_upd = fixed_sigma;
      }
      arma::vec error_upd = (dat_Y - beta0_upd - dat_X * beta_upd)/sigma_upd;
      //Rcpp::Rcout << "Iteration " << iter + 1 << ": nu update\n";
      nu_upd = Rcpp::as<double>(wrapper_nu(nu_pre, gamma_pre, error_upd, hyper_mu_nu, hyper_sigma_nu));
      //Rcpp::Rcout << "Iteration " << iter + 1 << ": gamma update\n";
      gamma_upd = Rcpp::as<double>(wrapper_gamma(gamma_pre, nu_upd, error_upd, hyper_c_gamma, hyper_d_gamma));
      arma::vec prob_indiv = 1/(1+(SS_t0 * (1-theta_pre) / (SS_t1 * theta_pre)) * exp(-abs(beta_upd) * (SS_t0-SS_t1)));
      double sum_prob_indiv = arma::sum(prob_indiv);
      theta_upd = (sum_prob_indiv + hyper_a_theta - 1)/(hyper_a_theta + p + hyper_b_theta_val -2);
      //   Rcpp::Rcout << "Iter: " << iter+1 << ",  beta0: " << beta0_upd << ",  sigma: " << sigma_upd << ", nu: " << nu_upd << ", gamma: " << gamma_upd << ", theta: " << theta_upd << std::endl;
    } catch (const std::exception& e) {
      Rcpp::stop(
        "Error in optimization during iteration %d: %s",
        iter + 1,
        e.what()
      );
    }
    // Store updates
    beta_hist.col(iter) = beta_upd;
    beta0_hist[iter] = beta0_upd;
    sigma_hist[iter] = sigma_upd;
    nu_hist[iter] = nu_upd;
    gamma_hist[iter] = gamma_upd;
    theta_hist[iter] = theta_upd;

    if (!beta_upd.is_finite() || !R_finite(beta0_upd) || !R_finite(sigma_upd) || !R_finite(nu_upd) || !R_finite(gamma_upd) || !R_finite(theta_upd)) {
      Rcpp::stop("Non-finite parameter encountered.");
    }

    // Check convergence
    if (conv_type == "param") {
      arma::vec prev = arma::join_vert(beta_pre, arma::vec({beta0_pre, sigma_pre, nu_pre, gamma_pre, theta_pre}));
      arma::vec upd = arma::join_vert(beta_upd, arma::vec({beta0_upd, sigma_upd, nu_upd, gamma_upd, theta_upd}));
      diff_val = arma::norm(upd - prev, 1);  // L1 norm
    } else if (conv_type == "loglik") {
      diff_val = abs(beta_neg_lk_cpp(beta_upd, beta0_upd, sigma_upd, nu_upd, gamma_upd, beta_pre, SS_t0, SS_t1, dat_Y, dat_X, theta_upd)-
        beta_neg_lk_cpp(beta_pre, beta0_pre, sigma_pre, nu_pre, gamma_pre, beta_pre_pre, SS_t0, SS_t1, dat_Y, dat_X, theta_pre));
    } else {
      Rcpp::stop("Invalid convergence type. Use 'param' or 'loglik'.");
    }
    if (diff_val < tol) {
      converged = true;
      break;
    }
    //Rcpp::Rcout << "diff_val: " << diff_val << std::endl;
    //Rcpp::Rcout << "beta_pre_pre: " << beta_pre_pre << "beta_pre: " << beta_pre << std::endl;
  }
  //Rcpp::Rcout << "Iter_out: " << iter << std::endl;

  // Trim history lists
  arma::mat beta_hist_trimmed = beta_hist.cols(0, iter-1);
  arma::vec beta0_hist_trimmed = beta0_hist.head(iter);
  arma::vec sigma_hist_trimmed = sigma_hist.head(iter);
  arma::vec nu_hist_trimmed = nu_hist.head(iter);
  arma::vec gamma_hist_trimmed = gamma_hist.head(iter);
  arma::vec theta_hist_trimmed = theta_hist.head(iter);

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta_upd,
    Rcpp::Named("beta0") = beta0_upd,
    Rcpp::Named("sigma") = sigma_upd,
    Rcpp::Named("include_sigma") = include_sigma,
    Rcpp::Named("update_sigma") = update_sigma,
    Rcpp::Named("nu") = nu_upd,
    Rcpp::Named("gamma") = gamma_upd,
    Rcpp::Named("theta") = theta_upd,
    Rcpp::Named("iter") = iter,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("diff_param") = diff_val,
    Rcpp::Named("history") = Rcpp::List::create(
      Rcpp::Named("beta") = beta_hist_trimmed,
      Rcpp::Named("beta0") = beta0_hist_trimmed,
      Rcpp::Named("sigma") = sigma_hist_trimmed,
      Rcpp::Named("nu") = nu_hist_trimmed,
      Rcpp::Named("gamma") = gamma_hist_trimmed,
      Rcpp::Named("theta") = theta_hist_trimmed
    )
  );
}
