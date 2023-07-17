#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
#include <math.h>
#include <chrono>
using namespace Rcpp;
using namespace arma;
using namespace std;

double gpenalty(double beta, double a, double lambda, String penalty){
  double pbeta;
  if(penalty == "scad"){
    if(abs(beta) > a*lambda)
      pbeta = 0.5*(a+1)*lambda*lambda;
    else if(abs(beta) <= lambda)
      pbeta = lambda*abs(beta);
    else
      pbeta = (a*lambda*abs(beta)-0.5*(beta*beta+lambda*lambda))/(a-1);
  }
  else{
    if(abs(beta) > a*lambda)
      pbeta = 0.5*a*lambda*lambda;
    else
      pbeta = lambda*abs(beta)-0.5*beta*beta/a;
  }
  return pbeta;
}

double penaltysum(arma::vec beta, double a, double lambda, String penalty){
  int num = beta.size();
  arma::vec pbeta(num, fill::zeros);
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

double checklosssum(arma::vec u, double tau){
  int num = u.size();
  arma::vec loss(num, fill::zeros);
  for(int j = 0; j < num; j++){
    if(u(j) > 0) loss(j) = tau*u(j);
    else loss(j) = (tau-1)*u(j);
  }
  return accu(loss); 
}

//[[Rcpp::export]]
Rcpp::List QPADMslackcpp(arma::vec y, arma::mat x, double tau, String penalty, double a, double lambda, double pho = 5, int maxstep = 1000, double eps = 0.001, bool intercept = false){
  
  int n = x.n_rows;
  if(intercept){
    x.insert_cols(0, arma::ones(n));
  }
  int p = x.n_cols;
  
  arma::vec etaini(n, fill::zeros), zini(p, fill::zeros), uini(p, fill::zeros), vini(n, fill::zeros);
  arma::vec beta(p, fill::zeros), xi(n, fill::zeros), eta(n, fill::zeros), z(p, fill::zeros), u(p, fill::zeros), v(n, fill::zeros);
  arma::vec xmcp(3, fill::zeros), hmcp(3, fill::zeros), xscad(4, fill::zeros), hscad(4, fill::zeros); 
  arma::vec yx = y-x*zini;
  
  double lossini = 0, loss = 0, distance = 1, time = 0, phi = 0;
  lossini = checklosssum(y-x*beta,tau)+penaltysum(beta, a, lambda, penalty);
  
  auto start_pre = std::chrono::high_resolution_clock::now();
  arma::mat tmp(p, p, fill::zeros);
  if(n>p){
    tmp = inv(arma::eye(p,p)+x.t()*x);
  }
  else{
    tmp = arma::eye(p,p)-x.t()*inv(arma::eye(n,n)+x*x.t())*x;
  }
  auto finish_pre = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_pre = finish_pre - start_pre;
  time = time+elapsed_pre.count();
  
  int iteration = 0;
  while(((distance > eps)|(distance == 0))&&(iteration < maxstep)){
    
    auto start = std::chrono::high_resolution_clock::now();
    if(penalty == "scad"){
      if(intercept){
        beta(0) = zini(0)+uini(0)/pho;
        for(int j = 1; j < p; j++){
          phi = zini(j)+uini(j)/pho;
          xscad(0) = sign(phi)*min(lambda, max(0.0, abs(phi)-lambda/pho));
          xscad(1) = sign(phi)*min(a*lambda, max(lambda, (pho*(a-1)*abs(phi)-a*lambda)/(pho*(a-1)-1)));
          xscad(2) = sign(phi)*max(a*lambda, abs(phi));
          for(int i = 0; i < 4; i++){
            hscad(i) = 0.5*(xscad(i)-phi)*(xscad(i)-phi)+gpenalty(xscad(i), a, lambda, penalty)/pho; 
          }
          beta(j) = xscad(hscad.index_min());
        }
      }
      else{
        for(int j = 0; j < p; j++){
          phi = zini(j)+uini(j)/pho;
          xscad(0) = sign(phi)*min(lambda, max(0.0, abs(phi)-lambda/pho));
          xscad(1) = sign(phi)*min(a*lambda, max(lambda, (pho*(a-1)*abs(phi)-a*lambda)/(pho*(a-1)-1)));
          xscad(2) = sign(phi)*max(a*lambda, abs(phi));
          for(int i = 0; i < 4; i++){
            hscad(i) = 0.5*(xscad(i)-phi)*(xscad(i)-phi)+gpenalty(xscad(i), a, lambda, penalty)/pho; 
          }
          beta(j) = xscad(hscad.index_min());
        }
      }
    }
    else{
      if(intercept){
        beta(0) = zini(0)+uini(0)/pho;
        for(int j = 1; j < p; j++){
          phi = zini(j)+uini(j)/pho;
          xmcp(0) = sign(phi)*min(a*lambda, max(0.0, a*(pho*abs(phi)-lambda)/(pho*a-1)));
          xmcp(1) = sign(phi)*max(a*lambda, abs(phi));
          for(int i = 0; i < 3; i++){
            hmcp(i) = 0.5*(xmcp(i)-phi)*(xmcp(i)-phi)+gpenalty(xmcp(i), a, lambda, penalty)/pho; 
          }
          beta(j) = xmcp(hmcp.index_min());
        }
      }
      else{
        for(int j = 0; j < p; j++){
          phi = zini(j)+uini(j)/pho;
          xmcp(0) = sign(phi)*min(a*lambda, max(0.0, a*(pho*abs(phi)-lambda)/(pho*a-1)));
          xmcp(1) = sign(phi)*max(a*lambda, abs(phi));
          for(int i = 0; i < 3; i++){
            hmcp(i) = 0.5*(xmcp(i)-phi)*(xmcp(i)-phi)+gpenalty(xmcp(i), a, lambda, penalty)/pho; 
          }
          beta(j) = xmcp(hmcp.index_min());
        }
      }
    }
    
    xi = yx+etaini+vini/pho-tau*arma::ones(n)/pho;
    for(int i = 0; i < n; i++){
      if(xi(i)<0){
        xi(i) = 0;
      }
    }
    
    eta = -yx+xi-vini/pho-(1-tau)*arma::ones(n)/(pho);
    for(int i = 0; i < n; i++){
      if(eta(i)<0){
        eta(i) = 0;
      }
    }
    
    z = tmp*(beta-uini/pho+x.t()*(y-xi+eta+vini/pho));
    yx = y-x*z;
    u = uini+pho*(z-beta);
    v = vini+pho*(yx-xi+eta);
    
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    time = time + elapsed.count();
    
    loss = checklosssum(y-x*beta,tau)+penaltysum(beta, a, lambda, penalty);
    distance = abs(loss-lossini);
    zini = z, etaini = eta, uini = u, vini = v, lossini = loss;
    iteration = iteration+1;
  } 
  
  Rcpp::List final;
  final = List::create(Named("Estimation") = beta,Named("Iteration") = iteration, Named("Time") = time);
  return final;

}

//[[Rcpp::export]]
Rcpp::List paraQPADMslackcpp(arma::vec y, arma::mat x, int K, double tau, String penalty, double a, double lambda, double pho = 15, int maxstep = 1000, double eps = 0.001, bool intercept = false){

  int n = x.n_rows, nk = n/K;
  if(intercept){
    x.insert_cols(0, arma::ones(n));
  }
  int p = x.n_cols;

  arma::mat zini(p, K, fill::zeros), uini(p, K, fill::zeros), z(p, K, fill::zeros), u(p, K, fill::zeros);
  arma::vec etaini(n, fill::zeros), vini(n, fill::zeros);
  arma::vec beta(p, fill::zeros), xi(n, fill::zeros), eta(n, fill::zeros), v(n, fill::zeros);
  arma::vec zmean(p, fill::zeros), umean(p, fill::zeros);
  arma::vec xmcp(3, fill::zeros), hmcp(3, fill::zeros), xscad(4, fill::zeros), hscad(4, fill::zeros);
  arma::vec yx = y;
  lambda = lambda/n, pho = pho/n;

  double max_prep = 0, time_reduce = 0, max_map = 0, time = 0;
  arma::vec map(K, fill::zeros);
  
  double distance = 1, lossini = 0, loss = 0, phi = 0;
  
  lossini = checklosssum(y-x*beta, tau)+penaltysum(beta, a, lambda, penalty);

  arma::cube tmp = arma::zeros<arma::cube>(p,p,K);
  for(int k = 0; k < K; k++){
    arma::mat tmp2, xk = x.rows(k*nk,k*nk+nk-1);
    auto start_prep = std::chrono::high_resolution_clock::now();
    if(nk > p) 
      tmp2 = inv(arma::eye(p,p)+xk.t()*xk);
    else tmp2 = arma::eye(p,p)-xk.t()*inv(arma::eye(nk,nk)+xk*xk.t())*xk;
    auto finish_prep = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_prep = finish_prep - start_prep;
    if(elapsed_prep.count()>max_prep) max_prep = elapsed_prep.count();
    tmp.slice(k) = tmp2;
  }
  time = time + max_prep;
  
  int iteration = 0;

  while(((distance > eps)|(distance == 0))&&(iteration < maxstep)){
 
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
    
    for(int k = 0; k < K; k++){
      arma::mat xk = x.rows(k*nk,k*nk+nk-1);
      arma::vec yk = y.subvec(nk*k, nk*k+nk-1), vinik = vini.subvec(nk*k, nk*k+nk-1), etainik = etaini.subvec(k*nk,k*nk+nk-1);
      auto start_map = std::chrono::high_resolution_clock::now();
      xi.subvec(k*nk,k*nk+nk-1) = yx.subvec(k*nk,k*nk+nk-1)+etainik+vinik/pho-tau*arma::ones(nk)/(n*pho);
      for(int i = k*nk; i<k*nk+nk; i++){
        if(xi(i) < 0){
          xi(i) = 0;
        }
      }
      eta.subvec(k*nk,k*nk+nk-1) = -yx.subvec(k*nk,k*nk+nk-1)+xi.subvec(k*nk,k*nk+nk-1)-vinik/pho-(1-tau)*arma::ones(nk)/(n*pho);
      for(int i = k*nk; i < k*nk+nk; i++){
        if(eta(i) < 0){
          eta(i) = 0;
        }
      }
      z.col(k) = tmp.slice(k)*(beta-uini.col(k)/pho+xk.t()*(yk-xi.subvec(k*nk,k*nk+nk-1)+eta.subvec(k*nk,k*nk+nk-1)+vinik/pho));
      yx.subvec(k*nk,k*nk+nk-1) = yk-xk*z.col(k);
      u.col(k) = uini.col(k)+pho*(z.col(k)-beta);
      v.subvec(k*nk,k*nk+nk-1) = vinik+pho*(yx.subvec(k*nk,k*nk+nk-1)-xi.subvec(k*nk,k*nk+nk-1)+eta.subvec(k*nk,k*nk+nk-1));
      auto finish_map = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_map = finish_map - start_map;
      map(k) = elapsed_map.count();
    }
    max_map = map(map.index_max());
    time = time + max_map;
  
    loss = checklosssum(y-x*beta,tau)+penaltysum(beta, a, lambda, penalty);
    distance = sum(abs(loss-lossini));
    
    lossini = loss, zini = z, uini = u, etaini = eta, vini=v;
    iteration = iteration+1;
  }  

  Rcpp::List final;
  final = List::create(Named("Estimation") = beta, Named("Iteration") = iteration, Named("Time") = time);
  return final; 

}


