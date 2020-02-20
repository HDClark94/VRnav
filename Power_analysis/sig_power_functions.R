# Functions for generation of simulated datasets and power analysis
# for nested experiments with binary response variables.
# Example application is virtual reality location estimation experiments.

## General functions to generate synthetic data
## for a single subject taking trials in each condition.

# Make dependent variables for two groups
make_groups <- function(distances, trials, fitter="glm"){
  if (fitter == "aov") {
    x1 <- rep(distances, 2)          # locations 
    x2 <- c(rep(0, length(distances)), rep(1, length(distances)))  # groups
    data.frame(x1=x1, x2=x2)
  } else {
    x1 <- rep(distances, trials * 2)          # locations 
    x2 <- c(rep(0, trials * length(distances)), rep(1, trials * length(distances)))  # groups
    data.frame(x1=x1, x2=x2)
  }
  
}

# Generate data given dependent variables and the coefficients
make_i_data <- function(x1, x2, c1 = 2.1, c2 = -1.14, c3 = 1, c4 = 0.01, fitter="glm", trials=20) {
  z <- c1 + c2*x1 + c3*x2 + c4*x1*x2       # model
  # pass through an inv-logit function
  pr <- 1/(1+exp(-z))
  
  # Or value 1.1 used below in deominator to generate a basal error rate,
  # and prevent model separation when fitting.
  #pr <- 1/(1.1+exp(-z))
  
  
  if (fitter =="aov") {
    y <- rbinom(length(x1), trials ,pr)      # bernoulli response variable
  } else {
    y <- rbinom(length(x1),1 ,pr)      # bernoulli response variable
  }
}


## Generate data for a group of subjects taking multiple trials
## in each of two conditions
generate_i_group <- function(subjects = 20,
                           distances = c(0.6, 0.9, 1.35, 2.025, 3.0375, 4.55625),
                           trials = 20,
                           c1 = 2.1,
                           c2 = -1.14,
                           c3 = 5,
                           c4 = 0.06,
                           fitter = "glm"){
  df_subs <- c()
  for(i in 1:subjects){
    #make data for a single subject
    df <- make_groups(distances = distances, trials = trials, fitter=fitter) %>%
      mutate(y = pmap_dbl(list(x1, x2, c1 = c1, c2 = c2, c3 = c3, c4 = c4, fitter = fitter, trials=trials), make_i_data))
    df$subject <- i
    df_subs <- rbind(df_subs, df)
  }
  df_subs
}


# Make a matrix with values for c3 and c4. We'll stick with default values for pairs of coefficients.
# Write a function to do this.
generate_coefs <- function(vals_1 = seq(-5, 5, 1),
                           vals_2 = seq(-0.05, 0.05, 0.05),
                           repeats = 3,
                           trials = 20,
                           subjects = 20) {
  vals <- matrix(nrow = length(vals_1) * length(vals_2), ncol = 2)
  for (i in 1:length(vals_1)) {
    for(j in 1:length(vals_2)){
      vals[[(i-1)*length(vals_2)+j, 1]] <- vals_1[[i]]
      vals[[(i-1)*length(vals_2)+j, 2]] <- vals_2[[j]]
    }
  }
  # convert vals to a tibble with one value of c3 and c4 per row
  vals_tib <- tibble(
    c3 = rep(vals[,1], repeats),
    c4 = rep(vals[,2], repeats),
    trials = trials,
    subjects = subjects,
    repeats = repeats
  )
  vals_tib
}


# To carry out simulations of a full experiment and return the fitted interaction models.
# Generate data for a given set of coefficients, fits and returns the interaction model
helper_i_glm <- function(subjects = 20,
                       distances = c(0.6, 0.9, 1.35, 2.025, 3.0375, 4.55625),
                       trials = 20,
                       c1 = 2.1,
                       c2 = -1.14,
                       c3 = 5,
                       c4 = 0.06,
                       fitter = "glm") {
  df_group <- generate_i_group(
    subjects = subjects,
    distances = distances,
    trials = trials,
    c1 = c1,
    c2 = c2,
    c3 = c3,
    c4 = c4,
    fitter = fitter
  )
  if (fitter == "glm") {
    glm(y ~ x1 * x2, data = df_group, family = "binomial")
    
  } else if (fitter == "glmer") {
    glmer(y ~ x1 * x2 + (1 | subject), data = df_group, family ="binomial")
  } else
  {
    aov(y ~ x1 * x2, data = df_group)
  }
}

helper_i_glmer_wrap <- function(...) {
  helper_i_glm(fitter = "glmer", ...)
}

helper_i_aov_wrap <- function(...) {
  helper_i_glm(fitter = "aov", ...)
}


# For each set of coefficients simulate an experiment using helper_i_glm
# Then extract coefficients from the model
# Make a function to do this
# ... can include repeats and trials to be passed to generate_coefs 
sim_full_i_exp <- function(vals_1 = seq(-5, 5, 5),
                         vals_2 = seq(-0.05, 0.05, 0.05),
                         ...) {
  models <- generate_coefs(
    vals_1 = vals_1,
    vals_2 = vals_2
  ) %>%
    mutate(
      glm_mod = pmap(list(trials = trials, c3 = c3, c4 = c4), helper_i_glm),
      glm_tidy = map(glm_mod, tidy),
      glm_x2_p = map_dbl(glm_tidy, ~.$p.value[[3]]),
      glm_x3_p = map_dbl(glm_tidy, ~.$p.value[[4]]),
      glmer_mod = pmap(list(trials = trials, c3 = c3, c4 = c4), helper_i_glmer_wrap),
      glmer_tidy = map(glmer_mod, tidy),
      glmer_x2_p = map_dbl(glmer_tidy, ~.$p.value[[3]]),
      glmer_x3_p = map_dbl(glmer_tidy, ~.$p.value[[4]]),
      aov_mod = pmap(list(trials = trials, c3 = c3, c4 = c4), helper_i_aov_wrap),
      aov_tidy = map(aov_mod, tidy),
      aov_x2_p = map_dbl(aov_tidy, ~.$p.value[[2]]),
      aov_x3_p = map_dbl(aov_tidy, ~.$p.value[[3]]),
    )
}


# Summarise the results of a full experiments.
# Generates summary statistics foreach combination of c3 and c4.
# E.g. Counts for p values relative to threshold.
full_i_exp_sum <- function(df){
  df %>% group_by(c3, c4) %>%
    summarise(glm_x2_p_mean = mean(glm_x2_p), 
              glm_x3_p_mean = mean(glm_x3_p),
              glm_x2_thresh_sum = sum(glm_x2_p < 0.05),
              glm_x2_thresh_prop = glm_x2_thresh_sum / mean(repeats),
              glm_x3_thresh_sum = sum(glm_x3_p < 0.05),
              glm_x3_thresh_prop = glm_x3_thresh_sum / mean(repeats),
              glmer_x2_p_mean = mean(glmer_x2_p), 
              glmer_x3_p_mean = mean(glmer_x3_p),
              glmer_x2_thresh_sum = sum(glmer_x2_p < 0.05),
              glmer_x2_thresh_prop = glmer_x2_thresh_sum / mean(repeats),
              glmer_x3_thresh_sum = sum(glmer_x3_p < 0.05),
              glmer_x3_thresh_prop = glmer_x3_thresh_sum / mean(repeats),
              aov_x2_p_mean = mean(aov_x2_p), 
              aov_x3_p_mean = mean(aov_x3_p),
              aov_x2_thresh_sum = sum(aov_x2_p < 0.05),
              aov_x2_thresh_prop = aov_x2_thresh_sum / mean(repeats),
              aov_x3_thresh_sum = sum(aov_x3_p < 0.05),
              aov_x3_thresh_prop = aov_x3_thresh_sum / mean(repeats),
              
    )
}
