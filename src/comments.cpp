#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
#include <math.h>
#include <chrono>
using namespace Rcpp;
using namespace arma;
using namespace std;

//function for calculating the SCAD or MCP penalty at a fixed beta value
//input: 
//  beta: a fixed beta value (double)
//  a: shape parameter of the penalty (double) (common choice: a = 3.7 for SCAD and a = 3 for MCP)
//  lambda: penalization parameter (double)
//  penalty: type of penalty (string) (SCAD or MCP)
//output: 
//  pbeta: the calculated SCAD or MCP penalty value at a given beta (double)
double gpenalty(double beta, double a, double lambda, String penalty){
  
  double pbeta;
  //calculate SCAD penalty value when the given penalty type penalty == "scad"
  //SCAD has three different forms according to the absolute value of the given beta
  //  abs(beta): absolute value of the given beta
  if(penalty == "scad"){
    if(abs(beta) > a*lambda)
      pbeta = 0.5*(a+1)*lambda*lambda;
    else if(abs(beta) <= lambda)
      pbeta = lambda*abs(beta);
    else
      pbeta = (a*lambda*abs(beta)-0.5*(beta*beta+lambda*lambda))/(a-1);
  }
  //calculate MCP penalty value when the given penalty type is not "scad"
  //MCP penalty has two different forms according to the absolute value of the given beta
  else{
    if(abs(beta) > a*lambda)
      pbeta = 0.5*a*lambda*lambda;
    else
      pbeta = lambda*abs(beta)-0.5*beta*beta/a;
  }
  return pbeta;
  
}

//function for calculating the accumulated SCAD or MCP penalty value for a given beta vector (i.e., the sum of SCAD or MCP penalty at each component of this beta vector)
//input:
//  beta: a given beta vector (vector)
//  a, lambda and penalty are the same as those in the above "gpenalty" function 
//output:
//  accu(pbeta): the accumulated SCAD or MCP penalty value (the vector pbeta records the SCAD or MCP penalty at each component of the beta vector, and accu(pbeta) calculates the sum of these components in pbeta) (double)   
double penaltysum(arma::vec beta, double a, double lambda, String penalty){
  //compute the length of the given beta vector by using the Rcpp function .size() and put the obtained value into the newly defined integer variable num
  int num = beta.size();
  //define a zero vector pbeta with dimension num 
  arma::vec pbeta(num, fill::zeros);
  //calculate the SCAD or MCP penalty for each component of the beta vector, and put the obtained value to the corresponding position of the pbeta vector
  //the specific implementation is the same as that in the "gpenalty" function
  //beta(j) and pbeta(j) are respectively the jth component of beta and pbeta, and fabs(beta(j)) is absolute value of the component beta(j) 
  if(penalty == "scad"){
    for(int j = 0; j < num; j++){
      if(fabs(beta(j)) > a*lambda)
        pbeta(j) = 0.5*(a+1)*lambda*lambda;
      else if(fabs(beta(j)) <= lambda)
        pbeta(j) = lambda*fabs(beta(j));
      else
        pbeta(j) = (a*lambda*fabs(beta(j))-0.5*(beta(j)*beta(j)+lambda*lambda))/(a-1);
    }
  }
  else{
    for(int j = 0; j < num; j++){
      if(fabs(beta(j)) > a*lambda)
        pbeta(j) = 0.5*a*lambda*lambda;
      else
        pbeta(j) = lambda*fabs(beta(j))-0.5*beta(j)*beta(j)/a;
    }
  }
  return accu(pbeta);
}

//function for calculating the accumulated check loss value at a given vector  
//the definition of check loss function at a fixed value is given below equation (1) in our paper
//input:
//  u: a given vector (vector)
//  tau: the quantile level (double), a fixed value in the open interval (0, 1)
//output:
//  accu(loss): the accumulated check loss value (the vector loss records the check loss for each component of the given vector u) (double)
double checklosssum(arma::vec u, double tau){
  int num = u.size();
  arma::vec loss(num, fill::zeros);
  //calculate the check loss for each component of the given u vector, and put the obtained value to the corresponding position of the loss vector
  //the check loss function has two different forms according to if u(j) > 0
  for(int j = 0; j < num; j++){
    if(u(j) > 0) loss(j) = tau*u(j);
    else loss(j) = (tau-1)*u(j);
  }
  return accu(loss); 
}

//function for implementing the QPADM-slack algorithm
//input:
//  y: a sample data vector (doubles)
//  x: a sample data matrix (doubles)
//  K: the number of data partitions (integer), in reality, it is the number of local machines
//  tau: a given quantile level (double), a fixed value in the open interval (0, 1) 
//  penalty, a and lambda: same as those in the above "gpenalty" function
//  pho: a given augmented parameter (double), here we set its default value as 15  
//  maxstep: allowed maximum number of iterations of the algorithm (integer), here we set its default value as 1000
//  eps: stop criterion of the algorithm (double), if the distance of the objective at two successive iterations is less than eps, we stop the algorithm, here we set its default value as 0.001
//  intercept: logic variable indicating if the model contains an intercept
//output:
//  final: a list type variable containing three parts
//        1. Estimation: the final obtained update value for beta (vector)
//        2. Iteration: number of iterations of the algorithm (integer)
//        3. Time: the computational time of the algorithm (double)
//[[Rcpp::export]]
Rcpp::List paraQPADMslackcpp(arma::vec y, arma::mat x, int K, double tau, String penalty, double a, double lambda, double pho = 5, int maxstep = 500, double eps = 0.001, bool intercept = false){

  //calculate the number of rows and columns of the matrix x by using the Rcpp functions .n_rows and .n_cols, and put the obtained values into the newly defined integer variables n and p, respectively
  //note: if the model contains an intercept, we should first insert a column ones into the matrix x (in the left) and then calculate the number of columns of x
  //the integer nk calculated by n/K is then the number of rows of the kth partition of the matrix x
  int n = x.n_rows, nk = n/K;
  if(intercept){
    x.insert_cols(0, arma::ones(n));
  }
  int p = x.n_cols;
  
  //initialize the variables used in the algorithm by zero vectors or zero matrices
  //zini (matrix), uini (matrix), etaini (vector) and vini (vector) respectively are the updated values of z, u, eta and v in the previous iteration
  //zmean (vector) and umean (vector) respectively are the mean of zini and uini by rows 
  //xmcp (vector), hmcp(vector), xscad (vector) and hscad (vector) are temporary variables for updating beta (for more details, please see the slides)
  //yx (vector) is a temporary variable for updating v (for more details, please see the slides)
  arma::mat zini(p, K, fill::zeros), uini(p, K, fill::zeros), z(p, K, fill::zeros), u(p, K, fill::zeros);
  arma::vec etaini(n, fill::zeros), vini(n, fill::zeros);
  arma::vec beta(p, fill::zeros), xi(n, fill::zeros), eta(n, fill::zeros), v(n, fill::zeros);
  arma::vec zmean(p, fill::zeros), umean(p, fill::zeros);
  arma::vec xmcp(3, fill::zeros), hmcp(3, fill::zeros), xscad(4, fill::zeros), hscad(4, fill::zeros);
  arma::vec yx = y;
  //divide the values of lambda and pho given by users by n to adjust them to the appropriate order 
  lambda = lambda/n, pho = pho/n;
  
  //initialize the variables for recording the computational time of the algorithm
  double max_prep = 0, time_reduce = 0, max_map = 0, time = 0;
  arma::vec map(K, fill::zeros);
  
  //variable distance is the distance of the objective at two successive iterations (double), we initialize it by 1
  //lossini is the objective value in the previous iteration (double)
  //loss is the objective value in the current iteration (double)
  //phi is a temporary variable for updating beta
  double distance = 1, lossini = 0, loss = 0, phi = 0;
  
  //calculate the objective value by using our previously defined functions "checklosssum" and "penaltysum"  
  lossini = checklosssum(y-x*beta, tau)+penaltysum(beta, a, lambda, penalty);
  
  //in updating z, we need to calculate the inversion of K matrices, these matrices are fixed, thus we conduct this process before before implement the iterative algorithm
  //tmp is a 3-dimensional array, it contains K elements, each element is a p*p matrix, which stores a matrix inversion, we initialize it with zeros
  //in reality, these matrix inversions can be conducted in parallel, as each inversion only based on a partition of x, here we implement this by a "for" loop, i.e., it is sequential now. This is the first part we want to be parallelized
  //xk: the kth partition of x (matrix)
  arma::cube tmp = arma::zeros<arma::cube>(p,p,K);
  for(int k = 0; k < K; k++){
    arma::mat tmp2, xk = x.rows(k*nk,k*nk+nk-1);
    //record the starting time of the calculation
    auto start_prep = std::chrono::high_resolution_clock::now();
    if(nk > p) 
      tmp2 = inv(arma::eye(p,p)+xk.t()*xk);
    else tmp2 = arma::eye(p,p)-xk.t()*inv(arma::eye(nk,nk)+xk*xk.t())*xk;
    //record the starting time of the calculation
    auto finish_prep = std::chrono::high_resolution_clock::now();
    //calculate the time cost 
    std::chrono::duration<double> elapsed_prep = finish_prep - start_prep;
    //max_prep records the maximum time of the K matrix inversions
    if(elapsed_prep.count()>max_prep) max_prep = elapsed_prep.count();
    //put the kth matrix inversion result into the kth component of tmp
    tmp.slice(k) = tmp2;
  }
  time = time + max_prep;
  
  int iteration = 0;

  //the specific implementation of QPADM-slack
  while(((distance > eps)|(distance == 0))&&(iteration < maxstep)){

    //update beta (for details see the slides)
    auto start_reduce = std::chrono::high_resolution_clock::now();
    zmean = mean(zini, 1);
    umean = mean(uini, 1);
    if(penalty == "scad"){
      if(intercept){
        beta(0) = zmean(0)+umean(0)/pho;
        for(int j = 1; j < p; j++){
          phi = zmean(j)+umean(j)/pho;
          xscad(0) = sign(phi)*min(lambda, max(0.0, abs(phi)-lambda/(pho*K)));
          xscad(1) = sign(phi)*min(a*lambda, max(lambda, (pho*K*(a-1)*abs(phi)-a*lambda)/(pho*K*(a-1)-1)));
          xscad(2) = sign(phi)*max(a*lambda, abs(phi));
          for(int i = 0; i < 4; i++){
            hscad(i) = 0.5*(xscad(i)-phi)*(xscad(i)-phi)+gpenalty(xscad(i), a, lambda, penalty)/(pho*K); 
          }
          beta(j) = xscad(hscad.index_min());
        }
      }
      else{
        for(int j = 0; j < p; j++){
          phi = zmean(j)+umean(j)/pho;
          xscad(0) = sign(phi)*min(lambda, max(0.0, abs(phi)-lambda/(pho*K)));
          xscad(1) = sign(phi)*min(a*lambda, max(lambda, (pho*K*(a-1)*abs(phi)-a*lambda)/(pho*K*(a-1)-1)));
          xscad(2) = sign(phi)*max(a*lambda, abs(phi));
          arma::vec hscad(4, fill::zeros);
          for(int i = 0; i < 4; i++){
            hscad(i) = 0.5*(xscad(i)-phi)*(xscad(i)-phi)+gpenalty(xscad(i), a, lambda, penalty)/(pho*K); 
          }
          beta(j) = xscad(hscad.index_min());
        }
      }
    }
    else{
      if(intercept){
        beta(0) = zmean(0)+umean(0)/pho;
        for(int j = 1; j < p; j++){
          phi = zmean(j)+umean(j)/pho;
          xmcp(0) = sign(phi)*min(a*lambda, max(0.0, a*(pho*K*abs(phi)-lambda)/(pho*K*a-1)));
          xmcp(1) = sign(phi)*max(a*lambda, abs(phi));
          for(int i = 0; i < 3; i++){
            hmcp(i) = 0.5*(xmcp(i)-phi)*(xmcp(i)-phi)+gpenalty(xmcp(i), a, lambda, penalty)/(pho*K); 
          }
          beta(j) = xmcp(hmcp.index_min());
        }
      }
      else{
        for(int j = 0; j < p; j++){
          phi = zmean(j)+umean(j)/pho;
          xmcp(0) = sign(phi)*min(a*lambda, max(0.0, a*(pho*K*abs(phi)-lambda)/(pho*K*a-1)));
          xmcp(1) = sign(phi)*max(a*lambda, abs(phi));
          for(int i = 0; i < 3; i++){
            hmcp(i) = 0.5*(xmcp(i)-phi)*(xmcp(i)-phi)+gpenalty(xmcp(i), a, lambda, penalty)/(pho*K); 
          }
          beta(j) = xmcp(hmcp.index_min());
        }
      }
    }
    auto finish_reduce = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_reduce = finish_reduce - start_reduce;
    time_reduce = elapsed_reduce.count();
    time = time + time_reduce;
    
    //update xi, eta, z, u and v (see details in the slides), this is the second part we want to be parallelized
    for(int k = 0; k < K; k++){

      arma::mat xk = x.rows(k*nk,k*nk+nk-1);
      arma::vec yk = y.subvec(nk*k, nk*k+nk-1), vinik = vini.subvec(nk*k, nk*k+nk-1), etainik = etaini.subvec(k*nk,k*nk+nk-1);
      auto start_map = std::chrono::high_resolution_clock::now();
      //update xi
      xi.subvec(k*nk,k*nk+nk-1) = yx.subvec(k*nk,k*nk+nk-1)+etainik+vinik/pho-tau*arma::ones(nk)/(n*pho);
      for(int i = k*nk; i<k*nk+nk; i++){
        if(xi(i) < 0){
          xi(i) = 0;
        }
      }
      //update eta
      eta.subvec(k*nk,k*nk+nk-1) = -yx.subvec(k*nk,k*nk+nk-1)+xi.subvec(k*nk,k*nk+nk-1)-vinik/pho-(1-tau)*arma::ones(nk)/(n*pho);
      for(int i = k*nk; i < k*nk+nk; i++){
        if(eta(i) < 0){
          eta(i) = 0;
        }
      }
      //update z
      z.col(k) = tmp.slice(k)*(beta-uini.col(k)/pho+xk.t()*(yk-xi.subvec(k*nk,k*nk+nk-1)+eta.subvec(k*nk,k*nk+nk-1)+vinik/pho));
      //update u
      u.col(k) = uini.col(k)+pho*(z.col(k)-beta);
      //update v
      yx.subvec(k*nk,k*nk+nk-1) = yk-xk*z.col(k);
      v.subvec(k*nk,k*nk+nk-1) = vinik+pho*(yx.subvec(k*nk,k*nk+nk-1)-xi.subvec(k*nk,k*nk+nk-1)+eta.subvec(k*nk,k*nk+nk-1));
      auto finish_map = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_map = finish_map - start_map;
      map(k) = elapsed_map.count();
    }
    max_map = map(map.index_max());
    time = time + max_map;
    
    //calculate the updated objective value "loss" and the distance between the objective at two successive iterations 
    loss = checklosssum(y-x*beta,tau)+penaltysum(beta, a, lambda, penalty);
    distance = sum(abs(loss-lossini));
    
    lossini = loss, zini = z, uini = u, etaini = eta, vini=v;
    iteration = iteration+1;
  }  

  Rcpp::List final;
  final = List::create(Named("Estimation") = beta, Named("Iteration") = iteration, Named("Time") = time);
  return final; 

}
