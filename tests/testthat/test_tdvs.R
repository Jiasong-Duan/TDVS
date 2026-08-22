test_that("tdvs() works with sample data", {
  data(data_tdvs)  # Your example dataset

  # Basic smoke test
  result <-TDVS(data_tdvs)

  # Test output structure
  expect_true(is.list(result))
  expect_named(result, c("beta_estimates", "beta0_estimate", "sigma_estimate", "include_sigma", "update_sigma", "nu_estimate", "gamma_estimate", "selected_indices", "p_values"))
  expect_true(is.numeric(result$beta_estimates))
  expect_true(is.numeric(result$beta0_estimate))
  expect_true(is.numeric(result$sigma_estimate))
  expect_true(is.logical(result$include_sigma))
  expect_true(is.logical(result$update_sigma))
  expect_true(is.numeric(result$nu_estimate))
  expect_true(is.numeric(result$gamma_estimate))
  expect_true(is.numeric(result$p_values))
  expect_true(is.integer(result$selected_indices) || is.numeric(result$selected_indices))
})
