library(parallel)
library(doSNOW)
library(glmnet)

L1logistic_ss= function(x,y,
                        lambda.max,lmr, nlambda = 30, 
                        B = 100){
  
  n = nrow(x); p = ncol(x)
  # compute lambdaSeq
  lambdaSeq = sort(exp(seq(log(lmr*lambda.max), log(lambda.max), length.out=nlambda)),decreasing = T)
  
  # stability selection
  pb = progress::progress_bar$new(total=B)
  ret = list()
  n_sub = round(nrow(x)/2)
  for(k in 1:B){
    pb$tick()
    # n/2 random subsample
    ind1 = sample(1:nrow(x),size = n_sub,replace = F)
    x1 = x[ind1,]; y1= y[ind1]
    # fit LASSO over lambdaSeq
    fit_k = glmnet(x1,y1,family = "binomial",lambda = lambdaSeq)
    # selected features for each lambda
    selected = predict(fit_k, type="nonzero")
    # res = nlambda * p matrix (0=not selected; 1=selected)
    res = replicate(length(lambdaSeq), logical(ncol(x)))
    for(j in 1:length(lambdaSeq)){
      res[selected[[j]],j] = TRUE
    }
    rownames(res) = colnames(x)
    ret[[k]] = res
  }
  
  # pct of selection
  p_select = Reduce("+", ret)/B
  
  return(list(ret = ret, p_select = p_select, lambdaSeq = lambdaSeq))
}

# pi_thr = 0.9
selected_ss_old = function(fit_ss, q0, pi_thr){
  p_select = fit_ss$p_select
  n_select = fit_ss$n_select
  
  # average number of selection at each lambda over B runs
  avg_n_select = apply(n_select,2,mean)
  # smallest lambda index s.t. avg_n_select < q (defines minimum lambda in lambda_grid)
  max_lambda_idx = max(which(avg_n_select < q0))
  
  # selected feats: max(p_select in lambda_grid) > pi_thr
  p_select_tr = p_select[,1:max_lambda_idx]
  p_select_max = apply(p_select_tr,1,max)
  selected_feats_ind = which(p_select_max > pi_thr)
  selected_feats = p_select_tr[selected_feats_ind,ncol(p_select_tr)]
  selected_feats = selected_feats[order(selected_feats,decreasing = T)]
  return(selected_feats)
}

selected_ss = function(fit_ss, q0, pi_thr){
  
  
  # union of selected features up to lambda_j
  # = S^lambda for lambda in [lambda_j, lambda_max]
  n_feats_up_to_j = foreach(i=1:length(fit_ss$ret),.combine="rbind")%do%{
    sapply(1:length(fit_ss$lambdaSeq), function(j){
      v = apply(fit_ss$ret[[i]][,1:j,drop=F],1,sum) # number of selections up to lambda_j
      length(which(v>0)) # number of features which are selected at least once up to lambda_j
    })
  }
  
  # smallest lambda index s.t. n_feats_up_to_j < q (defines minimum lambda in lambda_grid)
  max_lambda_idx = max(which(apply(n_feats_up_to_j,2,mean)<=q0))
  
  # selected feats: max(p_select in lambda_grid) > pi_thr
  p_select_tr = fit_ss$p_select[,1:max_lambda_idx]
  p_select_max = apply(p_select_tr,1,max)
  selected_feats_ind = which(p_select_max > pi_thr)
  selected_feats = p_select_tr[selected_feats_ind,ncol(p_select_tr)]
  selected_feats = selected_feats[order(selected_feats,decreasing = T)]
  return(selected_feats)
}


# selected_ss(fit_ss,q0 = q0,pi_thr = pi_thr)
