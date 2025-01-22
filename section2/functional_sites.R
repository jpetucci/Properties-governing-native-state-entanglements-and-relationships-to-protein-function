remove(list=ls())

# load data - you will need to update the placeholder paths below to point to the location of the functional site data
ecoli = read.csv("/path/to/data")
human = read.csv("/path/to/data")
yeast= read.csv("/path/to/data")

###### Fitting #####

# dat<- ecoli; thresh = 5
fit_species <-function(dat, thresh=5, all_species = FALSE){
  FF <- c("free.ligand", "protein", "active_site", 
          "metal_interface_mc", "RNA.binding", "DNA.binding", "zinc_finger_region")
  
  #f<-FF[1]
  out_feat<-c()
  for(f in FF){
    tab1<-table(dat[dat$ent_window==1, f])
    if(sum(tab1 >= thresh) ==2){out_feat<-c(out_feat, f)}
    tab2.1<-table(dat[dat$ent_window==1 &dat$CR_type ==1, f])
    tab2.2<-table(dat[dat$ent_window==1 &dat$CR_type ==0, f])
    
    if(min(sum(tab2.1>=thresh), sum(tab2.2>=thresh)) ==2){ out_feat<-c(out_feat,paste0(f,":CR_type"))}
  }
  out_feat<-c(out_feat,"CR_type")
  if(all_species){out_feat<-c(out_feat,"species")}
  out<- glm(as.formula(paste("ent_window~",paste(out_feat,collapse = "+"))), data=dat,family="binomial")
  return(out)
  }
  

# fit logistic regression model per species  
f_e<- fit_species(ecoli)
f_h<- fit_species(human)
f_y<- fit_species(yeast)

# for all species
all <- rbind(ecoli,human,yeast)
all$species <- c(rep("ecoli",nrow(ecoli)), rep("human",nrow(human)), rep("yeast",nrow(yeast) ))

f_a <-fit_species(all,all_species = T)


###### Confidence Interval and P-values #####
# CI for log(OR)
#f<- "metal_interface_mc"
#fit<-f_h

CI_OR <- function(f,fit){
  
  iloc <- grep(x=names(coefficients(fit)),pattern = f)
  if(length(iloc)==0){return(NULL)}
  
  
  alpha <- 0.05
  
  # beta_f 
  l <- rep(0, length(coefficients(fit)))
  l[iloc[1]] <-1
  pe <- sum(l*coef(fit))
  se <- as.numeric(sqrt(t(l)%*%vcov(fit)%*%l))
  CI<- pe+c(-1,1)*qnorm(1-alpha/2)*se
  
  out1<- data.frame(f=f, pe = exp(pe), lb=exp(CI[1]), ub=exp(CI[2]), pval=2*pnorm(abs(pe/se),lower.tail = F))
  
  # beta_f+beta_f:RT 
  if(length(iloc)==2){
    l <- rep(0, length(coefficients(fit)))
    l[iloc] <-1
    pe <- sum(l*coef(fit))
    se <- as.numeric(sqrt(t(l)%*%vcov(fit)%*%l))
    CI<- pe+c(-1,1)*qnorm(1-alpha/2)*se
    
    out2<- data.frame(f=paste0(f,":CR_type"), pe = exp(pe), lb=exp(CI[1]), ub=exp(CI[2]), pval= 2*pnorm(abs(pe/se),lower.tail = F))
    
  }else{
    out2<- NULL
  }
  out_f <- rbind(out1,out2)
  
 
  return(out_f)
  
}

FF <- c("free.ligand", "protein", "active_site", 
        "metal_interface_mc", "RNA.binding", "DNA.binding", "zinc_finger_region")


# BH correction
res_e<-do.call("rbind", lapply(FF, function(f) CI_OR(f, f_e)))
res_h<-do.call("rbind", lapply(FF, function(f) CI_OR(f, f_h)))
res_y<-do.call("rbind", lapply(FF, function(f) CI_OR(f, f_y)))
res_a<-do.call("rbind", lapply(FF, function(f) CI_OR(f, f_a)))

res_e$pval <- p.adjust(res_e$pval,method = "BH")
res_h$pval <- p.adjust(res_h$pval,method = "BH")
res_y$pval <- p.adjust(res_y$pval,method = "BH")
res_a$pval <- p.adjust(res_a$pval,method = "BH")


library(dplyr)
library(tidyverse)
res_e<- res_e %>% data.frame %>%  mutate(CI=paste(round(pe,3)," [", round(lb,3), ",", round(ub,3), "] ",sep="")) %>% select(f,CI,pval)
res_h<- res_h %>% data.frame %>%  mutate(CI=paste(round(pe,3)," [", round(lb,3), ",", round(ub,3), "] ",sep="")) %>% select(f,CI,pval)
res_y<- res_y %>% data.frame %>%  mutate(CI=paste(round(pe,3)," [", round(lb,3), ",", round(ub,3), "] ",sep="")) %>% select(f,CI,pval)
res<- full_join(full_join(res_e,res_y, by="f"),res_h, by="f")
colnames(res)[-1] <- paste(rep(c("ecoli","yeast","human"),each=2), rep(c("CI","pval"),3),sep="_") # rename res

res_a<-res_a %>% data.frame %>%  mutate(CI=paste(round(pe,3)," [", round(lb,3), ",", round(ub,3), "] ",sep="")) %>% select(f,CI,pval)

knitr::kable(res_e) # ecoli
knitr::kable(res_y) # yeast
knitr::kable(res_h) # human
knitr::kable(res_a) # aggregated

write.csv(res,file = "output.csv",row.names = F)
write.csv(res_a,file = "output_a.csv",row.names = F)

