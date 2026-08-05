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
double beta_neg_lk_cpp(
    arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk, arma::vec betaPRE, double t0, double t1,
    arma::vec Y_lk, arma::mat X_lk, double theta_lk) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Check if theta_lk was provided
  double theta_val = (theta_lk < 0) ? 1.0 / p : theta_lk;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_lk.n_elem) != p) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec residuals = (Y_lk - beta0_lk - X_lk * beta_lk)/ sigma_lk;
 // Calculate the gamma exponential term
  arma::vec ga_pow(n);
  for (int i = 0; i < n; ++i) {
    double sign_res = (residuals[i] >= 0) ? -1.0 : 1.0;
    ga_pow[i] = std::pow(ga_lk, 2 * sign_res);
  }
  /* arma::vec ratio = pow(residuals, 2) / nu_lk * ga_pow; */
  arma::vec ratio = (pow(residuals, 2) / nu_lk) % ga_pow;
  arma::vec inverseprob = 1 + (t0 * (1 - theta_val) / (t1 * theta_val)) * exp(-abs(betaPRE) * (t0 - t1));
  double bneglog = arma::dot(abs(beta_lk), (t0 - (t0 - t1) / inverseprob)) + (nu_lk/2 + 0.5) * arma::sum(arma::log1p(ratio));
  return  bneglog;
}


// [[Rcpp::export]]
arma::vec beta_neg_gradient_cpp(
    arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Check if theta_lk was provided
  double theta_val = (theta_lk < 0) ? 1.0 / p : theta_lk;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_lk.n_elem) != p) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec residuals = (Y_lk - beta0_lk - X_lk * beta_lk)/ sigma_lk;
 // Calculate the gamma exponential term
  arma::vec ga_pow(n);
  for (int i = 0; i < n; ++i) {
    double sign_res = (residuals[i] >= 0) ? -1.0 : 1.0;
    ga_pow[i] = std::pow(ga_lk, 2 * sign_res);
  }
  arma::vec ratio = (pow(residuals, 2) / nu_lk) % ga_pow;
  arma::vec inverseprob = 1 + (t0 * (1 - theta_val) / (t1 * theta_val)) * exp(-abs(betaPRE) * (t0 - t1));
  arma::vec grad_denom = (2 * residuals % ga_pow / (sigma_lk * nu_lk)) / (1 + ratio);
  arma::vec signbeta_lk = arma::sign(beta_lk);
  arma::vec gradient = signbeta_lk % (t0 - (t0 - t1) / inverseprob) -(nu_lk/2 + 0.5) * X_lk.t() * grad_denom;
  return Rcpp::NumericVector(gradient.begin(), gradient.end());
  //return gradient;
}

// [[Rcpp::export]]
arma::mat beta_neg_hessian_cpp(
    arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk
    ) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_lk.n_elem) != p) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec residuals = (Y_lk - beta0_lk - X_lk * beta_lk)/ sigma_lk;
 // Calculate the gamma exponential term
  arma::vec ga_pow(n);
  for (int i = 0; i < n; ++i) {
    double sign_res = (residuals[i] >= 0) ? -1.0 : 1.0;
    ga_pow[i] = std::pow(ga_lk, 2 * sign_res);
  }
  // weights h_i
  arma::vec h(n);
  for (int i = 0; i < n; ++i) {
    double r = residuals[i];
    double gi = ga_pow[i];
    h[i] = (2.0 * gi / (sigma_lk * sigma_lk * nu_lk) * (1.0 - (gi/nu_lk) * r * r)) /
           std::pow(1.0 + (gi/nu_lk) * r * r, 2.0);
  }
  // Hessian
  arma::mat H = (nu_lk/2.0 + 0.5) * X_lk.t() * arma::diagmat(h) * X_lk;
  //return Hessian;
  return H;
}

// [[Rcpp::export]]
double jbeta_neg_gradient_cpp(
    int j_index, arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Check if theta_lk was provided
  double theta_val = (theta_lk < 0) ? 1.0 / p : theta_lk;
  // a C++ based index
  int j_index_cpp = j_index -1;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_lk.n_elem) != p) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec residuals = (Y_lk - beta0_lk - X_lk * beta_lk)/ sigma_lk;
 // Calculate the gamma exponential term
  arma::vec ga_pow(n);
  for (int i = 0; i < n; ++i) {
    double sign_res = (residuals[i] >= 0) ? -1.0 : 1.0;
    ga_pow[i] = std::pow(ga_lk, 2 * sign_res);
  }
  arma::vec ratio = (pow(residuals, 2) / nu_lk) % ga_pow;
  double inverseprob_j = 1 + (t0 * (1 - theta_val) / (t1 * theta_val)) * exp(-abs(betaPRE[j_index_cpp]) * (t0 - t1));
  arma::vec grad_denom = (2 * residuals % ga_pow / (sigma_lk * nu_lk)) / (1 + ratio);
  double signbeta_j = (beta_lk[j_index_cpp] > 0) ? 1.0 : ((beta_lk[j_index_cpp] < 0) ? -1.0 : 0.0);
  arma::vec X_j = X_lk.col(j_index_cpp);
  double gradient_j = signbeta_j * (t0 - (t0 - t1) / inverseprob_j) - (nu_lk/2.0 + 0.5) * arma::dot(X_j, grad_denom);

  return gradient_j;
}


// [[Rcpp::export]]
double jbeta_neg_hessian_cpp(
    int j_index, arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec Y_lk, arma::mat X_lk
    ) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_lk.n_elem) != p) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec residuals = (Y_lk - beta0_lk - X_lk * beta_lk)/ sigma_lk;
 // Calculate the gamma exponential term
  arma::vec ga_pow(n);
  for (int i = 0; i < n; ++i) {
    double sign_res = (residuals[i] >= 0) ? -1.0 : 1.0;
    ga_pow[i] = std::pow(ga_lk, 2 * sign_res);
  }
  // weights h_i
  arma::vec h(n);
  for (int i = 0; i < n; ++i) {
    double r = residuals[i];
    double gi = ga_pow[i];
    h[i] = (2.0 * gi / (sigma_lk * sigma_lk * nu_lk) * (1.0 - (gi/nu_lk) * r * r)) /
           std::pow(1.0 + (gi/nu_lk) * r * r, 2.0);
  }
  // a C++ based index
  int j_index_cpp = j_index -1;
  arma::vec X_j = X_lk.col(j_index_cpp);
  // Hessian
  double H_j = (nu_lk/2.0 + 0.5) * arma::dot(X_j % X_j, h);
  //return Hessian;
  return H_j;
}


// [[Rcpp::export]]
double jbeta_neg_lk_cpp_maxLik(
    double beta_j, int j_index, arma::vec beta_noj, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Check if theta_lk was provided
  double theta_val = (theta_lk < 0) ? 1.0 / p : theta_lk;
    // a C++ based index
  int j_index_cpp = j_index -1;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_noj.n_elem) != (p-1)) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec X_j = X_lk.col(j_index_cpp);
  arma::uvec idx = arma::regspace<arma::uvec>(0, p - 1);
  idx.shed_row(j_index_cpp);
  arma::mat X_noj = X_lk.cols(idx);
  arma::vec residuals = (Y_lk - beta0_lk - X_j * beta_j- X_noj * beta_noj)/ sigma_lk;
 // Calculate the gamma exponential term
  arma::vec ga_pow(n);
  for (int i = 0; i < n; ++i) {
    double sign_res = (residuals[i] >= 0) ? -1.0 : 1.0;
    ga_pow[i] = std::pow(ga_lk, 2 * sign_res);
  }
  arma::vec ratio = (pow(residuals, 2) / nu_lk) % ga_pow;
  double inverseprob_j = 1 + (t0 * (1 - theta_val) / (t1 * theta_val)) * exp(-abs(betaPRE[j_index_cpp]) * (t0 - t1));
  double bneglog_j = std::abs(beta_j) * (t0 - (t0 - t1) / inverseprob_j) + (nu_lk/2 + 0.5) * arma::sum(arma::log1p(ratio));
  return  bneglog_j;
}


// [[Rcpp::export]]
double jbeta_neg_gradient_cpp_maxLik(
    double beta_j, int j_index, arma::vec beta_noj, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Check if theta_lk was provided
  double theta_val = (theta_lk < 0) ? 1.0 / p : theta_lk;
  // a C++ based index
  int j_index_cpp = j_index -1;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_noj.n_elem) != (p-1)) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec X_j = X_lk.col(j_index_cpp);
  arma::uvec idx = arma::regspace<arma::uvec>(0, p - 1);
  idx.shed_row(j_index_cpp);
  arma::mat X_noj = X_lk.cols(idx);
  arma::vec residuals = (Y_lk - beta0_lk - X_j * beta_j- X_noj * beta_noj)/ sigma_lk;
 // Calculate the gamma exponential term
  arma::vec ga_pow(n);
  for (int i = 0; i < n; ++i) {
    double sign_res = (residuals[i] >= 0) ? -1.0 : 1.0;
    ga_pow[i] = std::pow(ga_lk, 2 * sign_res);
  }
  arma::vec ratio = (pow(residuals, 2) / nu_lk) % ga_pow;
  double inverseprob_j = 1 + (t0 * (1 - theta_val) / (t1 * theta_val)) * exp(-abs(betaPRE[j_index_cpp]) * (t0 - t1));
  arma::vec grad_denom = (2 * residuals % ga_pow / (sigma_lk * nu_lk)) / (1 + ratio);
  double signbeta_j = (beta_j > 0) ? 1.0 : ((beta_j < 0) ? -1.0 : 0.0);
  double gradient_j = signbeta_j * (t0 - (t0 - t1) / inverseprob_j) - (nu_lk/2.0 + 0.5) * arma::dot(X_j, grad_denom);

  return gradient_j;
}


// [[Rcpp::export]]
double jbeta_neg_hessian_cpp_maxLik(
    double beta_j, int j_index, arma::vec beta_noj, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec Y_lk, arma::mat X_lk
    ) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // a C++ based index
  int j_index_cpp = j_index -1;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_noj.n_elem) != (p-1)) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec X_j = X_lk.col(j_index_cpp);
  arma::uvec idx = arma::regspace<arma::uvec>(0, p - 1);
  idx.shed_row(j_index_cpp);
  arma::mat X_noj = X_lk.cols(idx);
  arma::vec residuals = (Y_lk - beta0_lk - X_j * beta_j- X_noj * beta_noj)/ sigma_lk;
 // Calculate the gamma exponential term
  arma::vec ga_pow(n);
  for (int i = 0; i < n; ++i) {
    double sign_res = (residuals[i] >= 0) ? -1.0 : 1.0;
    ga_pow[i] = std::pow(ga_lk, 2 * sign_res);
  }
  // weights h_i
  arma::vec h(n);
  for (int i = 0; i < n; ++i) {
    double r = residuals[i];
    double gi = ga_pow[i];
    h[i] = (2.0 * gi / (sigma_lk * sigma_lk * nu_lk) * (1.0 - (gi/nu_lk) * r * r)) /
           std::pow(1.0 + (gi/nu_lk) * r * r, 2.0);
  }
  // Hessian
  double H_j = (nu_lk/2.0 + 0.5) * arma::dot(X_j % X_j, h);
  //return Hessian;
  return H_j;
}

// [[Rcpp::export]]
Rcpp::NumericVector beta_neg_lk_cpp_nlm(
    arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk, arma::vec betaPRE, double t0, double t1,
    arma::vec Y_lk, arma::mat X_lk, double theta_lk) {

  // Get negtive log likelihood evaluated
  double neg_lk = beta_neg_lk_cpp(beta_lk, beta0_lk, sigma_lk, nu_lk, ga_lk, betaPRE, t0, t1, Y_lk, X_lk, theta_lk);
  // Get gradient of negtive log likelihood evaluated
  arma::vec neg_grad = beta_neg_gradient_cpp(beta_lk, beta0_lk, sigma_lk, nu_lk, ga_lk, betaPRE, t0, t1, Y_lk, X_lk, theta_lk);
  // Get hessian of negtive log likelihood evaluated
  arma::mat neg_hess = beta_neg_hessian_cpp(beta_lk, beta0_lk, sigma_lk, nu_lk, ga_lk, betaPRE, t0, t1, Y_lk, X_lk);

  // Wrap as NumericVector of length 1
  Rcpp::NumericVector bneglog(1);
  bneglog[0] = neg_lk;

  // Attach attributes
  bneglog.attr("gradient") = neg_grad;
  bneglog.attr("hessian")  = neg_hess;

  return bneglog;
}

// [[Rcpp::export]]
Rcpp::NumericVector jbeta_neg_lk_cpp_nlm(
    double beta_j, int j_index, arma::vec beta_noj, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk) {

  // Get negtive log likelihood evaluated
  double jneg_lk = jbeta_neg_lk_cpp_maxLik(beta_j, j_index, beta_noj, beta0_lk, sigma_lk, nu_lk, ga_lk, betaPRE, t0, t1, Y_lk, X_lk, theta_lk);
  // Get gradient of negtive log likelihood evaluated
  double jneg_grad = jbeta_neg_gradient_cpp_maxLik(beta_j, j_index, beta_noj, beta0_lk, sigma_lk, nu_lk, ga_lk, betaPRE, t0, t1, Y_lk, X_lk, theta_lk);
  // Get hessian of negtive log likelihood evaluated
  double jneg_hess = jbeta_neg_hessian_cpp_maxLik(beta_j, j_index, beta_noj, beta0_lk, sigma_lk, nu_lk, ga_lk, Y_lk, X_lk);

  // Wrap as NumericVector of length 1
  Rcpp::NumericVector jbneglog(1);
  jbneglog[0] = jneg_lk;

  // Attach attributes
  jbneglog.attr("gradient") = jneg_grad;
  jbneglog.attr("hessian")  = jneg_hess;

  return jbneglog;
}

// [[Rcpp::export]]
double beta0_neg_lk_cpp(double beta0_lk, arma::vec beta_lk, double sigma_lk, double nu_lk, double gamma_lk,
                        double hyper_mu_beta0, double hyper_sigma_beta0,
                        arma::vec Y_lk, arma::mat X_lk) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_lk.n_elem) != p) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec residuals = (Y_lk - beta0_lk - X_lk * beta_lk)/ sigma_lk;
  // Create index vector
  arma::vec index(n);
  for(int i = 0; i < n; i++) {
    index[i] = (residuals[i] >= 0) ? -1.0 : 1.0;
  }
  // Calculate sum in the formula
  double sum_term = 0.0;
  for(int i = 0; i < n; i++) {
    double ga_term = std::pow(gamma_lk, 2.0 * index[i]);
    sum_term += std::log(1.0 + residuals[i] * residuals[i] / nu_lk * ga_term);
  }
  return  0.5 * std::pow(beta0_lk - hyper_mu_beta0, 2) / hyper_sigma_beta0 +
    (nu_lk / 2.0 + 0.5) * sum_term;
}


// [[Rcpp::export]]
double sigma_neg_lk_cpp(double sigma_lk, arma::vec beta_lk, double beta0_lk, double nu_lk, double gamma_lk,
                        double hyper_nu_sigma, double hyper_A_sigma,
                        arma::vec Y_lk, arma::mat X_lk) {
  // Get n and p
  int n = Y_lk.n_elem;
  int p = X_lk.n_cols;
  // Ensure beta_lk size matches X_lk columns
  if (static_cast<int>(beta_lk.n_elem) != p) {
    Rcpp::stop("Size of beta coefficients does not match predictor dimensions");
  }
  // Compute residuals
  arma::vec residuals = (Y_lk - beta0_lk - X_lk * beta_lk)/sigma_lk;
  // Create index vector
  arma::vec index(n);
  for(int i = 0; i < n; i++) {
    index[i] = (residuals[i] >= 0) ? -1.0 : 1.0;
  }

  // Calculate sum in the formula
  double sum_term = 0.0;
  for(int i = 0; i < n; i++) {
    double ga_term = std::pow(gamma_lk, 2.0 * index[i]);
    sum_term += std::log(1.0 + residuals[i] * residuals[i] / nu_lk * ga_term);
  }
  const double prior_sigma = (hyper_nu_sigma / 2.0 + 0.5) * log1p((sigma_lk * sigma_lk) / (hyper_A_sigma * hyper_A_sigma * hyper_nu_sigma));
  return  prior_sigma + n * std::log(sigma_lk)+ (nu_lk / 2.0 + 0.5) * sum_term;
}


// [[Rcpp::export]]
double nu_neg_lk_cpp(double nu_lk, double ga_lk, Rcpp::NumericVector error_lk,
                     double hyper_mu, double hyper_sigma) {
  // Get n
  int n = error_lk.size();
  Rcpp::NumericVector index(n);
  // Create index vector (ifelse(error_lk>=0,-1,1))
  for(int i = 0; i < n; i++) {
    index[i] = (error_lk[i] >= 0) ? -1.0 : 1.0;
  }
  // Calculate sum in the formula
  double sum_term = 0.0;
  for(int i = 0; i < n; i++) {
    double ga_term = std::pow(ga_lk, 2.0 * index[i]);
    sum_term += std::log(1.0 + error_lk[i] * error_lk[i] / nu_lk * ga_term);
  }
  return (std::pow(std::log(nu_lk) - hyper_mu, 2) / (2.0 * std::pow(hyper_sigma, 2))) +
    (n / 2.0 + 1.0) * std::log(nu_lk) -
    n * (R::lgammafn(nu_lk / 2.0 + 0.5) - R::lgammafn(nu_lk / 2.0)) +
    (nu_lk / 2.0 + 0.5) * sum_term;
}


// [[Rcpp::export]]
double gamma_neg_lk_cpp(double ga_lk, double nu_lk, Rcpp::NumericVector error_lk,
                        double hyper_c, double hyper_d) {
  // Get n
  int n = error_lk.size();
  Rcpp::NumericVector index(n);
  // Create index vector (ifelse(error_lk>=0,-1,1))
  for(int i = 0; i < n; i++) {
    index[i] = (error_lk[i] >= 0) ? -1.0 : 1.0;
  }
  // Calculate sum in the formula
  double sum_term = 0.0;
  for(int i = 0; i < n; i++) {
    double ga_term = std::pow(ga_lk, 2.0 * index[i]);
    sum_term += std::log(1.0 + error_lk[i] * error_lk[i] / nu_lk * ga_term);
  }
  return -(n + hyper_c - 1.0) * std::log(ga_lk) + hyper_d * ga_lk +
    n * std::log(std::pow(ga_lk, 2) + 1.0) + (nu_lk / 2.0 + 0.5) * sum_term;
}


// [[Rcpp::export]]
arma::vec beta_coordinate_descent_cpp(
    arma::vec beta_cd, double beta0_cd, double sigma_cd, double nu_cd, double ga_cd,
    arma::vec betaPRE, double t0, double t1,
    arma::vec Y_cd, arma::mat X_cd, double theta_cd,
    int maX_cd_iter, double tol) {
  // Get p
  int p = X_cd.n_cols;
  // Update beta
  for (int iter = 0; iter < maX_cd_iter; ++iter) {
    Rcpp::Rcout << "iter: " << iter << std::endl;
    arma::vec beta_old = beta_cd;
  // Update beta sequentially
    for (int j = 0; j < p; ++j) {
      double g_j = jbeta_neg_gradient_cpp(j+1, beta_cd, beta0_cd, sigma_cd, nu_cd, ga_cd,
                                   betaPRE, t0, t1, Y_cd, X_cd, theta_cd);
      double h_jj = jbeta_neg_hessian_cpp(j+1, beta_cd, beta0_cd, sigma_cd, nu_cd, ga_cd, Y_cd, X_cd);
    Rcpp::Rcout << "j: " << j << std::endl;
    Rcpp::Rcout << "g_j: " << g_j << std::endl;
    Rcpp::Rcout << "h_jj: " << h_jj << std::endl;
  // avoid division by near-zero, provide a small number in the denominator
   double denom = (std::abs(h_jj) < 1e-12) ? ( (h_jj >= 0) ? 1e-12 : -1e-12 ) : h_jj;
    Rcpp::Rcout << "denom: " << denom << std::endl;
  // Newton update
      beta_cd[j] -= 0.2 * (g_j / denom);
    }
  // Check convergence
    if (arma::norm(beta_cd - beta_old, 2) < tol)
      break;
    Rcpp::Rcout << "diff_val: " << arma::norm(beta_cd - beta_old, 2) << std::endl;
  }
  // Return updated beta
  return beta_cd;
}

// [[Rcpp::export]]
arma::vec beta_coordinate_descent_cpp_maxLik(
    arma::vec beta_cd, double beta0_cd, double sigma_cd, double nu_cd, double ga_cd,
    arma::vec betaPRE, double t0, double t1,
    arma::vec Y_cd, arma::mat X_cd, double theta_cd,
    int maX_cd_iter, double tol) {
  // Get p
  int p = X_cd.n_cols;
  // Update beta
  Rcpp::Environment pkg_env = Rcpp::Environment::namespace_env("TDVS");
  Rcpp::Function wrapper_beta_cd_maxLik = pkg_env["wrapper_beta_cd_maxLik"];
  for (int iter = 0; iter < maX_cd_iter; ++iter) {
    arma::vec beta_old = beta_cd;
  // Update beta sequentially
    for (int j_cpp = 0; j_cpp < p; ++j_cpp) {
  // Newton update using maxLik
    beta_cd[j_cpp] = Rcpp::as<double>(wrapper_beta_cd_maxLik(j_cpp+1, beta_cd, beta0_cd, sigma_cd, nu_cd, ga_cd, betaPRE, t0, t1, Y_cd, X_cd, theta_cd));
    }
  // Check convergence
    if (arma::norm(beta_cd - beta_old, 2) < tol)
      break;
  //  Rcpp::Rcout << "diff_val: " << arma::norm(beta_cd - beta_old, 2) << std::endl;
  }
  // Return updated beta
  return beta_cd;
}

// [[Rcpp::export]]
arma::vec beta_coordinate_descent_cpp_nlm(
    arma::vec beta_cd, double beta0_cd, double sigma_cd, double nu_cd, double ga_cd,
    arma::vec betaPRE, double t0, double t1,
    arma::vec Y_cd, arma::mat X_cd, double theta_cd,
    int maX_cd_iter, double tol) {
  // Get p
  int p = X_cd.n_cols;
  // Update beta
  Rcpp::Environment pkg_env = Rcpp::Environment::namespace_env("TDVS");
  Rcpp::Function wrapper_beta_cd_nlm = pkg_env["wrapper_beta_cd_nlm"];
  for (int iter = 0; iter < maX_cd_iter; ++iter) {
    arma::vec beta_old = beta_cd;
  // Update beta sequentially
    for (int j_cpp = 0; j_cpp < p; ++j_cpp) {
  // Newton update using nlm
    beta_cd[j_cpp] = Rcpp::as<double>(wrapper_beta_cd_nlm(j_cpp+1, beta_cd, beta0_cd, sigma_cd, nu_cd, ga_cd, betaPRE, t0, t1, Y_cd, X_cd, theta_cd));
    }
  // Check convergence
    if (arma::norm(beta_cd - beta_old, 2) < tol)
      break;
  //  Rcpp::Rcout << "diff_val: " << arma::norm(beta_cd - beta_old, 2) << std::endl;
  }
  // Return updated beta
  return beta_cd;
}
