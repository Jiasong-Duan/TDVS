#ifndef TDVS_HELPERS_H
#define TDVS_HELPERS_H

#include <RcppArmadillo.h>

Rcpp::List beta_neg_lk_gradient_cpp(
    arma::vec betaNR, double beta0NR, double gaNR, double nuNR,
    arma::vec betaPRE, double t0, double t1, arma::vec YNR, arma::mat X_noIntNR,
    Rcpp::Nullable<double> thetaNR = R_NilValue);
double beta_neg_lk_cpp(
    arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk, arma::vec betaPRE, double t0, double t1,
    arma::vec Y_lk, arma::mat X_lk,
    double theta_lk = -1.0);
arma::vec beta_neg_gradient_cpp(
    arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk = -1.0);
arma::mat beta_neg_hessian_cpp(
    arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk);
double beta0_neg_lk_cpp(double beta0_lk, arma::vec beta_lk, double sigma_lk, double nu_lk, double gamma_lk,
                        double hyper_mu_beta0, double hyper_sigma_beta0,
                        arma::vec Y_lk, arma::mat X_lk);
double sigma_neg_lk_cpp(double sigma_lk, double beta0_lk, arma::vec beta_lk, double nu_lk, double gamma_lk,
                        double hyper_nu_sigma, double hyper_A_sigma,
                        arma::vec Y_lk, arma::mat X_lk);
double nu_neg_lk_cpp(double nu_lk, double ga_lk, Rcpp::NumericVector error_lk,
                     double hyper_mu=1, double hyper_sigma=1);
double gamma_neg_lk_cpp(double ga_lk, double nu_lk, Rcpp::NumericVector error_lk,
                        double hyper_c=0.0001, double hyper_d=0.0001);
double jbeta_neg_gradient_cpp(
    int j_index, arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk);
double jbeta_neg_hessian_cpp(
    int j_index, arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec Y_lk, arma::mat X_lk
    );
arma::vec beta_coordinate_descent_cpp(
    arma::vec beta_cd, double beta0_cd, double sigma_cd, double nu_cd, double ga_cd,
    arma::vec betaPRE, double t0, double t1,
    arma::vec Y_cd, arma::mat X_cd, double theta_cd,
    int maX_cd_iter = 100, double tol = 1e-6);
double jbeta_neg_lk_cpp_maxLik(
    double beta_j, int j_index, arma::vec beta_noj, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk);
double jbeta_neg_gradient_cpp_maxLik(
    double beta_j, int j_index, arma::vec beta_noj, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk);
double jbeta_neg_hessian_cpp_maxLik(
    double beta_j, int j_index, arma::vec beta_noj, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec Y_lk, arma::mat X_lk);
Rcpp::NumericVector beta_neg_lk_cpp_nlm(
    arma::vec beta_lk, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk, arma::vec betaPRE, double t0, double t1,
    arma::vec Y_lk, arma::mat X_lk, double theta_lk);
Rcpp::NumericVector jbeta_neg_lk_cpp_nlm(
    double beta_j, int j_index, arma::vec beta_noj, double beta0_lk, double sigma_lk, double nu_lk, double ga_lk,
    arma::vec betaPRE, double t0, double t1, arma::vec Y_lk, arma::mat X_lk,
    double theta_lk);
arma::vec beta_coordinate_descent_cpp_maxLik(
    arma::vec beta_cd, double beta0_cd, double sigma_cd, double nu_cd, double ga_cd,
    arma::vec betaPRE, double t0, double t1,
    arma::vec Y_cd, arma::mat X_cd, double theta_cd,
    int maX_cd_iter, double tol);
arma::vec beta_coordinate_descent_cpp_nlm(
    arma::vec beta_cd, double beta0_cd, double sigma_cd, double nu_cd, double ga_cd,
    arma::vec betaPRE, double t0, double t1,
    arma::vec Y_cd, arma::mat X_cd, double theta_cd,
    int maX_cd_iter, double tol);
Rcpp::List TDVS_EM_cpp(
    Rcpp::List dataXY,
    arma::vec init_beta,
    double init_beta0=1,
    double init_sigma=1,
    double init_nu=1,
    double init_gamma=1,
    double init_theta=0.5,
    double SS_t0 = 10.0,
    double SS_t1 = 1.0,
    double hyper_mu_beta0=0,
    double hyper_sigma_beta0=1e6,
    double hyper_nu_sigma=4,
    double hyper_A_sigma=50,
    double hyper_mu_nu=1,
    double hyper_sigma_nu=1,
    double hyper_c_gamma=0.0001,
    double hyper_d_gamma=0.0001,
    double hyper_a_theta=1,
    double hyper_b_theta = -1.0,
    int max_iter = 100,
    double tol = 1e-6,
    std::string conv_type = "param");
arma::vec slope_error_cpp(const arma::vec& err, double nu_slo, double ga_slo);
arma::vec curvature_error_cpp(const arma::vec& err, double nu_cur, double ga_cur);
double CiS_j_fun_cpp(int test_index,
                     const arma::vec& beta_opt,
                     double beta0_opt,
                     double sigma_opt,
                     double nu_opt,
                     double ga_opt,
                     Rcpp::List dataXY,
                     double add_correc_CiS = 0.001);
double per_fun_cpp(int j_index,
                   Rcpp::List dataXY,
                   arma::vec init_beta_per,
                   double init_beta0_per=1,
                   double init_sigma_per=1,
                   double init_nu_per=1,
                   double init_gamma_per=1,
                   double init_theta_per=0.5,
                   double SS_t0_per = 10.0,
                   double SS_t1_per = 1.0,
                   double hyper_mu_beta0_per=0,
                   double hyper_sigma_beta0_per=1e6,
                   double hyper_nu_sigma_per=4,
                   double hyper_A_sigma_per=50,
                   double hyper_mu_nu_per=1,
                   double hyper_sigma_nu_per=1,
                   double hyper_c_gamma_per=0.0001,
                   double hyper_d_gamma_per=0.0001,
                   double hyper_a_theta_per=1,
                   double hyper_b_theta_per = -1.0,
                   int max_iter_per = 100,
                   double tol_per = 1e-6,
                   double add_correc_CiS = 0.001);
double CiS_group_fun_cpp(const arma::uvec& test_indices,
                         const arma::vec& beta_opt,
                         double beta0_opt,
                         double sigma_opt,
                         double nu_opt,
                         double ga_opt,
                         Rcpp::List dataXY,
                         double add_correc_CiS = 0.001);
double per_group_fun_cpp(const arma::uvec& j_indices,
                         Rcpp::List dataXY,
                         arma::vec init_beta_per,
                         double init_beta0_per=1,
                         double init_sigma_per=1,
                         double init_nu_per=1,
                         double init_gamma_per=1,
                         double init_theta_per=0.5,
                         double SS_t0_per = 10.0,
                         double SS_t1_per = 1.0,
                         double hyper_mu_beta0_per=0,
                         double hyper_sigma_beta0_per=1e6,
                         double hyper_nu_sigma_per=4,
                         double hyper_A_sigma_per=50,
                         double hyper_mu_nu_per=1,
                         double hyper_sigma_nu_per=1,
                         double hyper_c_gamma_per=0.0001,
                         double hyper_d_gamma_per=0.0001,
                         double hyper_a_theta_per=1,
                         double hyper_b_theta_per = -1.0,
                         int max_iter_per = 100,
                         double tol_per = 1e-6,
                         double add_correc_CiS = 0.001
);
Rcpp::List TDVS_cpp(
    Rcpp::List dataXY,
    arma::vec init_beta_TDVS,
    int B = 300,
    double sig_cutoff = 0.05,
    double init_beta0_TDVS=1,
    double init_sigma_TDVS=1,
    double init_nu_TDVS=1,
    double init_gamma_TDVS=1,
    double init_theta_TDVS=0.5,
    double SS_t0_TDVS = 10.0,
    double SS_t1_TDVS = 1.0,
    double hyper_mu_beta0_TDVS=0,
    double hyper_sigma_beta0_TDVS=1e6,
    double hyper_nu_sigma_TDVS=4,
    double hyper_A_sigma_TDVS=50,
    double hyper_mu_nu_TDVS=1,
    double hyper_sigma_nu_TDVS=1,
    double hyper_c_gamma_TDVS=0.0001,
    double hyper_d_gamma_TDVS=0.0001,
    double hyper_a_theta_TDVS=1,
    double hyper_b_theta_TDVS = -1.0,
    int max_iter_TDVS = 100,
    double tol_TDVS = 1e-6,
    double add_correc_CiS = 0.001
);
Rcpp::List TDVS_j_cpp(
    int test_index,
    Rcpp::List dataXY,
    arma::vec init_beta_TDVS,
    int B = 300,
    double sig_cutoff = 0.05,
    double init_beta0_TDVS=1,
    double init_sigma_TDVS=1,
    double init_nu_TDVS=1,
    double init_gamma_TDVS=1,
    double init_theta_TDVS=0.5,
    double SS_t0_TDVS = 10.0,
    double SS_t1_TDVS = 1.0,
    double hyper_mu_beta0_TDVS=0,
    double hyper_sigma_beta0_TDVS=1e6,
    double hyper_nu_sigma_TDVS=4,
    double hyper_A_sigma_TDVS=50,
    double hyper_mu_nu_TDVS=1,
    double hyper_sigma_nu_TDVS=1,
    double hyper_c_gamma_TDVS=0.0001,
    double hyper_d_gamma_TDVS=0.0001,
    double hyper_a_theta_TDVS=1,
    double hyper_b_theta_TDVS = -1.0,
    int max_iter_TDVS = 100,
    double tol_TDVS = 1e-6,
    double add_correc_CiS = 0.001
);
Rcpp::List TDVS_group_cpp(
    const arma::uvec& test_indices,
    Rcpp::List dataXY,
    arma::vec init_beta_TDVS,
    int B,
    double sig_cutoff,
    double init_beta0_TDVS,
    double init_sigma_TDVS,
    double init_nu_TDVS,
    double init_gamma_TDVS,
    double init_theta_TDVS,
    double SS_t0_TDVS,
    double SS_t1_TDVS,
    double hyper_mu_beta0_TDVS,
    double hyper_sigma_beta0_TDVS,
    double hyper_nu_sigma_TDVS,
    double hyper_A_sigma_TDVS,
    double hyper_mu_nu_TDVS,
    double hyper_sigma_nu_TDVS,
    double hyper_c_gamma_TDVS,
    double hyper_d_gamma_TDVS,
    double hyper_a_theta_TDVS,
    double hyper_b_theta_TDVS,
    int max_iter_TDVS,
    double tol_TDVS,
    double add_correc_CiS);
Rcpp::List TDVS_multi_stage_cpp(
    Rcpp::List dataXY,
    arma::vec init_beta_TDVS,
    int group_B = 20,
    int indiv_B = 20,
    int B_final = 300,
    double group_cutoff = 3.0 / 20.0,
    double indiv_cutoff = 3.0 / 20.0,
    double sig_cutoff = 0.05,
    int group_size = 4,
    double init_beta0_TDVS=1,
    double init_sigma_TDVS=1,
    double init_nu_TDVS=1,
    double init_gamma_TDVS=1,
    double init_theta_TDVS=0.5,
    double SS_t0_TDVS = 10.0,
    double SS_t1_TDVS = 1.0,
    double hyper_mu_beta0_TDVS=0,
    double hyper_sigma_beta0_TDVS=1e6,
    double hyper_nu_sigma_TDVS=4,
    double hyper_A_sigma_TDVS=50,
    double hyper_mu_nu_TDVS=1,
    double hyper_sigma_nu_TDVS=1,
    double hyper_c_gamma_TDVS=0.0001,
    double hyper_d_gamma_TDVS=0.0001,
    double hyper_a_theta_TDVS=1,
    double hyper_b_theta_TDVS = -1.0,
    int max_iter_TDVS = 100,
    double tol_TDVS = 1e-6,
    double add_correc_CiS = 0.001);
#endif
