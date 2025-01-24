library(ggplot2)
library(ggpubr)

plot_ss = function(fit_ss){
  
  avg_n_select = apply(fit_ss$n_select,2,mean)
  max_lambda_idx = max(which(avg_n_select < fit_ss$q))
  
  rownames(fit_ss$p_select) = gsub(rownames(fit_ss$p_select),pattern="-",replacement="m")
  p_select_tr = fit_ss$p_select[,1:max_lambda_idx]
  p_select_max = apply(p_select_tr,1,max)
  selected_feats_ind = which(p_select_max > fit_ss$pi_thr)
  
  ind_feat_sel1 = which(p_select_max>0) # features that were selected at least once
  lambdaSeq = fit_ss$lambdaSeq
  
  p_select_tr = fit_ss$p_select[,1:max_lambda_idx]
  dat = data.frame(t(p_select_tr))
  dat = dat[,colnames(dat) %in% names(ind_feat_sel1)]
  # ordering based on the last row 
  dat = dat[,order(as.numeric(dat[nrow(dat),]),decreasing = T)] 
  dat = data.frame(lambdaSeq = lambdaSeq[1:max_lambda_idx], dat)
  
  # convert to long format
  dat_l = reshape2::melt(dat, id.vars = 'lambdaSeq',variable.name = 'Variables')
  
  # plot
  group = unique(dat_l$Variables)
  group.colors =ifelse(group %in% names(selected_feats_ind), "red","black")
  pl = 
    ggplot(data=dat_l, aes(x=-log(lambdaSeq), 
                           y = value)) +geom_line(aes(color=Variables))+
    scale_color_manual(values=group.colors)+
    geom_hline(yintercept = fit_ss$pi_thr,linetype = "dashed")
  
  pl
}