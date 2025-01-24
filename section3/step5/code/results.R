setwd("")
remove(list=ls())
load("ss_ecoli.Rdata")
load("ss_yeast.Rdata")
load("ss_human.Rdata")
source("L1logistic_ss.R")

p = 990

f_yeast=selected_ss(ss_yeast, q0 = floor(sqrt(0.5*p)),pi_thr = 0.9)
f_ecoli=selected_ss(ss_ecoli, q0 = floor(sqrt(0.5*p)),pi_thr = 0.9)
f_human=selected_ss(ss_human, q0 = floor(sqrt(0.5*p)),pi_thr = 0.9)


library(tidyr)
f_yeast = data.frame(f_yeast) %>% rownames_to_column(var = "feature")
f_ecoli = data.frame(f_ecoli) %>% rownames_to_column(var = "feature")
f_human = data.frame(f_human) %>% rownames_to_column(var = "feature")

feat_dat = full_join(full_join(f_yeast,f_ecoli,by="feature"), f_human, by = "feature")

feat_dat
# structure table
n_selected = apply(feat_dat[,2:4],1,function(x)sum(!is.na(x)))
avg_pct = apply(feat_dat[,2:4],1,mean,na.rm=T)
feat_dat$feature[feat_dat$feature=="ACH3_pssm"] = "ACH3"

ind_pccm = ifelse(1:nrow(feat_dat) %in%grep(feat_dat$feature,pattern="pssm"),1,0)
feat_dat = feat_dat[order(n_selected,avg_pct,ind_pccm, decreasing = c(T,T,F)), ]
feat_dat = feat_dat[,c("feature","f_ecoli","f_yeast","f_human")]
#feat_dat = feat_dat %>% mutate(common_feat = ifelse(apply(feat_dat[,2:4],1,function(x) sum(!is.na(x)))==3,"$dagger$",""))
colnames(feat_dat) = c("Feature", "E. coli", "Yeast", "Human")
feat_dat[,2:4][!is.na(feat_dat[,2:4])] = "X"
library(xtable)
print.xtable(xtable(feat_dat),include.rownames = F)


