remove(list=ls())
library(data.table)
library(glmnet)
library(doSNOW)
source("L1logistic_ss.R")

args=(commandArgs(TRUE))
if(length(args)==0){
  species = "ecoli"
}else{
  for(i in 1:length(args)){
    eval(parse(text=args[[i]]))
  }
}
# Download data from https://drive.google.com/drive/folders/134Q8mSJzizDzPu1n5Bw3mi2pnmCCylUc?usp=sharing
if(species=="ecoli"){
  species_dat = fread("ecoli_data.csv")
}else if(species=="yeast"){
  species_dat = fread("yeast_data.csv")
}else if(species=="human"){
  species_dat = fread("human_data.csv")
}

species_dat$V1 = NULL

#species_dat = species_dat[1:10000,]
Y_species_dat = species_dat$target_value
X_species_dat = species_dat[,-c("target_value")]
rm(species_dat); gc()

cat("species:", species,"\n")
print(dim(X_species_dat))

q0= round(sqrt(0.5*ncol(X_species_dat)))
pi_thr = 0.9
EV_bound = 1/(2*pi_thr-1) * q0^2/ncol(X_species_dat)

suppressWarnings({fit0 = glmnet(X_species_dat,Y_species_dat,family = "binomial",pmax=q0)})
lambda.max = max(fit0$lambda)

cat("lambda.max", lambda.max, "\n")
cat("lmr", min(fit0$lambda)/max(fit0$lambda), "\n")
cat("EV_bound",EV_bound,"\n")

set.seed(1)
fit_ss = L1logistic_ss(X_species_dat,Y_species_dat, lambda.max = lambda.max,lmr = 0.05, nlambda = 30,B = 100)

assign(paste0("ss_",species),fit_ss)
save(list=paste0("ss_",species), file=paste0("ss_",species,".Rdata"))

print(selected_ss(fit_ss,q0 = q0,pi_thr = pi_thr))
