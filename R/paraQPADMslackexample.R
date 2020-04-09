#' The Parallel QPADM-slack Algorithm for Nonconvex Penalized Quantile Regression in Distributed Big Data Based on ADMM
#' This is a pseudo parallel implementation.
#'
#' @param y The response vector
#' @param x The design matrix (without intercept)
#' @param K Number of partitions (split the full data into K subsets)
#' @param tau The quantile of interest
#' @param penalty The penalty to use, currently support ("scad" and "mcp")
#' @param a The shape parameter of the SCAD/MCP penalty
#' @param lambda The penalization parameter of the SCAD/MCP penalty
#' @param pho The augmentation parameter for the ADMM
#' @param maxstep Maximum number of iterations allowed, default is 1000
#' @param eps The tolerance parameter for convergence, default is 1e-03
#' @param intercept Whether to include the intercept into the model, default is FALSE
#' @return The coefficient estimation, the number of iterations and the running time for the WQR-ADMM algorithm, and the total time cost for computing the WQR estimator (when the parameter esttype = "WQR")
#' @examples
#' N = 200000
#' p = 200
#' K = 100
#' tau = 0.7
#' rho = 0.5        
#' a = 3.7
#' lambda = 650
#' beta_true = rep(0, p)
#' beta_true[6] = beta_true[12] = beta_true[15] = beta_true[20] = 1
#' gcov = function(p, rho){
#'   cov = matrix(1, p, p);
#'   for(i in 1:p){
#'     for(j in 1:p){
#'       if(i < j) cov[i,j] = rho^{j-i}
#'       else cov[i,j] = cov[j,i]
#'     }
#'   }
#'   cov
#' }
#' cov = gcov(p, rho)
#' X = matrix(rnorm(N*p), nrow=N) 
#' X = X%*%chol(cov)
#' X[,1] = pnorm(X[,1])
#' e = rnorm(N)
#' Y = X[,6]+X[,12]+X[,15]+X[,20]+0.7*X[,1]*e
#' beta_true[1] = 0.7*qnorm(tau)
#' paraqpadmslack = paraQPADMslack(Y, X, K, tau, "scad", a, lambda)
#' beta = paraqpadmslack$Estimation
#' iteration = paraqpadmslack$Iteration
#' Time = paraqpadmslack$Time
#' AE = sum(abs(beta-beta_true))
#' @export
#'
paraQPADMslack <- function(y, x, K, tau, penalty, a, lambda, pho = 15, maxstep = 1000, eps = 1e-03, intercept = FALSE) {
  .Call('_QPADMslack_paraQPADMslackcpp', PACKAGE = 'QPADMslack', y, x, K, tau, penalty, a, lambda, pho, maxstep, eps, intercept)
}

