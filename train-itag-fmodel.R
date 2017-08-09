library(ggplot2)	 # for plotting
library(gridExtra) # for laying out plots
library(caret) 		 # for ML
library(doMC)  		 # for using multiple cores, Linux only
library(dplyr)     # for the %>% operator

#### Set up multicore processing
registerDoMC(cores=20)

#### Get the method (from slurm job array ID)
methods <- c("rpart", "bstTree", "treebag", "ctree", "gbm", "svmRadial", "svmLinear", "RLightGBM", 
	"xgbLinear", "xgbTree", "ranger", "deepboost", "AdaBag", "bagFDAGCV", "bagEarthGCV", "bagEarth", "bagFDA")
idx <- Sys.getenv("SLURM_ARRAY_TASK_ID")
m <- methods[as.integer(idx)]

if(m=="RLightGBM") {
	library(RLightGBM)
	m <- caretModel.LGBM()
}

print(paste("##### Using method ", m, sep=""))

#### Read in data
source("../src/constants.R")
datadir <- "../../data/"

source("experiment-list.R")

dat <- data.frame()
for (expID in exps){
  print(paste("Loading data: ", expID, sep=""))
  exp <- readRDS(paste(datadir, expID, ".Rda", sep=""))
  dat <- rbind(dat, exp)
}
dat$expID <- factor(dat$expID)

#### Divide data

dat <- dat[,!names(dat) %in% c("status")]
# also do some cleaning first
dat <- na.omit(dat)

# only build model for filling-state data
dat <- dat[dat$label=="F" & dat$video!=0,]
dat$video <- factor(dat$video, levels=unique(dat$video))         

print(str(dat))

#### Set formula

labelcol <- "video"                                                             
source("feature-list.R")

dat.formula <- paste(labelcol, paste(featurecols, collapse=" + "), sep=" ~ ")

#### Run a classifier!

folds <- groupKFold(dat$expID, 10)

mod <- train(as.formula(dat.formula), 
                      data=dat, 
                      method = m,
                      preProc = c("center", "scale"),
                      tuneLength = 30, 
                      trControl =
                        trainControl(
                                     index=folds,
                                     sampling="smote"
                        )
)

print(mod)
print(str(mod))
print(mod$resample$Accuracy)
print(mean(mod$resample$Accuracy, na.rm=TRUE))

saveRDS(mod, file=paste("../../output-itag-fmodel/", methods[as.integer(idx)], "-model.Rda", sep=""))

