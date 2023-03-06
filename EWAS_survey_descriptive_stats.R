#######
## Must first run regression_analysis_PS_NHANES_99_06.R with generate_desciptive_statistics = 1
#######

# increase console log limit
options(max.print=1000000)
rm(list = ls())

library(arules)
library(broom)
library(dplyr)
library(foreach)
library(car)
library(Hmisc)
library(survey)
library(mfx)
library(margins)
library(hash)
# library(stargazer)
library(testthat)
library(crayon)
library(readxl)
library(jsonlite)


`%notin%` <- Negate(`%in%`)

current_dir_path = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_dir_path)

######################################################################
########### Settings 1 ############
######################################################################

nut_panel = c('12', '58')[2]


survey_year <- 'all'
debug_run <- FALSE


path_diet_data <- 'input_data/all_diet_data_1999_2006_58_nuts_single_and_ensemble_FPro.csv'

nhanesCCNR <- read.csv(path_diet_data)
cat(bold('Diet Data File Name: ', current_dir_path, '/', path_diet_data, sep=''), '\n')

load('input_data/nh_99-06.Rdata')

MainTable <- merge(
  x = MainTable, 
  y = nhanesCCNR[ , c(
       "SEQN",
       
       'num_unique_dishes',
       'metabolic.syndrome.only.examination', 'metabolic.syndrome.examination.and.drug',
       'LBXACR_lab_detectable', 'LBXGLY_lab_detectable',
       
       # 'framingham_risk_10_years', THIS IS THE PYTHON BUT THE R VERSION IS MORE RELIABLE
       
       'ascvd_10y_accaha_lab', 'ascvd_10y_frs_lab', 'ascvd_10y_frs_simple_lab',
       "Total.calories.consumed.mean.both.days",
       "HEI2015_TOTAL_SCORE",
       
       "FPro.RW.WFDPI.mean.of.both.days.sum",
       "FPro.WFDPI.mean.of.both.days.sum",
       "FPro.WCDPI.mean.of.both.days.sum",
       
       "ens_FPro.WFDPI.mean.of.both.days.sum",
       "ens_FPro.RW.WFDPI.mean.of.both.days.sum",
       "ens_FPro.WCDPI.mean.of.both.days.sum",
       
       "ens_min_FPro.WFDPI.mean.of.both.days.sum",
       "ens_min_FPro.RW.WFDPI.mean.of.both.days.sum",
       "ens_min_FPro.WCDPI.mean.of.both.days.sum",
       
       # "predNOVA4.consumption.kcal.percentage.over.sum.both.days",
       # "predNOVA4.consumption.RW.grams.percentage.over.sum.both.days",
       # "predNOVA4.consumption.grams.percentage.over.sum.both.days"
       
       "manualNOVA4.consumption.kcal.percentage.over.sum.both.days"
     )], 
  by = "SEQN")

# keep age in its current form because it will be normalized
MainTable$age <- MainTable$RIDAGEYR
nrow(MainTable)



####################################################################
# Custom vars module by CCNR 
####################################################################

MainTable$t2d <- I(MainTable$LBXGLU >= 126)
MainTable$metabolic_syndrome_examination <- MainTable$metabolic.syndrome.only.examination
MainTable$metabolic_syndrome <- MainTable$metabolic.syndrome.examination.and.drug

if (survey_year == 'all') {
  ######
  ## Create sample weights for 8 years based on 
  ## https://wwwn.cdc.gov/nchs/nhanes/tutorials/module3.aspx
  ####
  
  MainTable[MainTable$SDDSRVYR == 1, 'WTMEC8YR'] <- MainTable[
    MainTable$SDDSRVYR == 1, 'WTMEC4YR'] * (2 / 4)
  
  MainTable[MainTable$SDDSRVYR == 2, 'WTMEC8YR'] <- MainTable[
    MainTable$SDDSRVYR == 2, 'WTMEC4YR'] * (2 / 4)
  
  MainTable[MainTable$SDDSRVYR == 3, 'WTMEC8YR'] <- MainTable[
    MainTable$SDDSRVYR == 3, 'WTMEC2YR'] * (1 / 4)
  
  MainTable[MainTable$SDDSRVYR == 4, 'WTMEC8YR'] <- MainTable[
    MainTable$SDDSRVYR == 4, 'WTMEC2YR'] * (1 / 4)
  
  #dat <- subset(MainTable2, SDDSRVYR < 5 & age >= 18) 
  
  cat(bold('Number of rows with weight=0 in ALL dataset:'), 
      nrow(MainTable[MainTable$WTMEC8YR == 0, ]), '\n')
  
  nhanesDesign <- svydesign(id      = ~SDMVPSU, 
                            strata  = ~SDMVSTRA, 
                            weights = ~WTMEC8YR, # Use 8 year weights
                            nest    =T,
                            data    = MainTable
  )
  
  nrow(nhanesDesign)
  svymean(~age, nhanesDesign, ci=FALSE)
  #svyby(~age, ~age > 0, design=nhanesDesign, FUN=svymean, ci=TRUE)
  
  #####################
  # CORRECT WAY TO SUBSET survey data is
  # https://static-bcrf.biochem.wisc.edu/courses/Tabular-data-analysis-with-R-and-Tidyverse/book/12-usingNHANESweights.html
  # https://r-survey.r-forge.r-project.org/survey/html/subset.survey.design.html
  #####################
  ageDesign <- subset(nhanesDesign, 
                      age >= 18 & 
                        WTMEC8YR > 0 & 
                        ens_FPro.WFDPI.mean.of.both.days.sum > 0
  )
  
  nrow(ageDesign$variables)
  svymean(~age, ageDesign, ci=TRUE)
  
  #####################################################
  
  MainTable_subset <- subset(MainTable, 
                             age >= 18 & 
                             WTMEC8YR > 0 &
                             ens_FPro.WFDPI.mean.of.both.days.sum > 0
  )
  
  if ((nrow(MainTable_subset) == nrow(ageDesign$variables)) == F){
    cat(bold('Must have equal rows nrow(ageDesign$variables) == nrow(MainTable_subset) otherwise SOMETHING IS WRONG!'), '\n')
    stop()
  }
}


##################
## Settings
##################

d <- ageDesign

# p_names = c("p.50","p.55","p.60","p.65","p.70","P.75","p.80","p.85","p.90","p.95","p.10")
# p_breaks= c(0,0.5,  0.55,  0.6,   0.65,  0.7,   0.75,  0.8,   0.85,  0.9,   0.95,  1.0)
# file_name_postfix <- 'tenP'

p_names = c("p0.2","p0.4","p0.6","p0.8","p1.0")
p_breaks= c(0,0.2,   0.4,   0.6,   0.8,   1.0)
file_name_postfix <- 'fiveP'

#p_names = c("p.50","p.55","p.10")
#p_breaks= c(0,0.5,  0.55,  1.0)

#### SETTING
measure_index <- 3 # Table S8
#measure_index <- 2 # Table S7

col_index = c(
  "ens_min_FPro.WFDPI.mean.of.both.days.sum",
  "ens_min_FPro.WCDPI.mean.of.both.days.sum",
  "ens_min_FPro.RW.WFDPI.mean.of.both.days.sum",
  'HEI2015_TOTAL_SCORE'
  )[measure_index]

col_index_D = c(
  'ens_min_WFDPI_D', 'ens_min_WCDPI_D', 
  'ens_min_RWWFDPI_D', 'HEI2015_D')[measure_index]

if (col_index == 'HEI2015_TOTAL_SCORE'){
  # Becuase HEI is between 0 and 100
  p_breaks = p_breaks * 100
}



##################
## End Settings
##################

##################
## Number of Records
##################

nrow(d)

##################
## End of Records
##################

d$variables[, col_index_D] <- discretize(
  d$variables[, col_index], method="fixed",
  breaks = p_breaks,
  labels = p_names
)


discretize(
  d$variables$HEI2015_TOTAL_SCORE, method="fixed",
  breaks = c(-Inf, 0.2, 0.4, 0.6, 0.8, Inf),
  labels = c("p1", "p2", "p3", "p4", "p5")
)

out_df <- data.frame(matrix(ncol = length(p_names) + 2))
colnames(out_df) <- c('var', 'mean', p_names)

count_subj_partitions = as.data.frame(table(
  d$variables[, col_index_D]
  #d$variables$WCDPI.D
))
colnames(count_subj_partitions) <- c('var', 'f')

i_rows = 1

## Num subjects
out_df[i_rows, 'var'] <- 'counts'

for (p_name in p_names) {
  out_df[i_rows, p_name] <- count_subj_partitions[
    count_subj_partitions['var'] == p_name, 'f']
}


process_svymean_output <- function(out_df, i_rows, var_name,
                                   col_index_D, p_names,design) {
  i_rows <- i_rows + 1
  
  out_df[i_rows, 'var'] <- var_name

  mean_all <- svymean(design$variables[,var_name],design,na.rm=TRUE)
  mean_all <- as.data.frame(mean_all)
  mean_all <- round(mean_all, 2)
  
  out_df[i_rows, 'mean'] <- paste0(mean_all$mean, '±', mean_all$SE)
  
  p_mean = svyby(
    formula = as.formula(paste0('~',var_name)), 
    by = as.formula(paste0('~',col_index_D)), 
    design = design, 
    FUN = svymean, 
    keep.var = TRUE, 
    drop.empty.groups = FALSE, 
    na.rm = TRUE
  )
  # for INDFMPIR we do have NAs
  
  for (p_name in p_names) {
    
    mean_value = round(p_mean[p_mean[col_index_D]==p_name,var_name], 2)
    
    se_value = round(p_mean[p_mean[col_index_D]==p_name,'se'], 2)
    
    out_df[i_rows, p_name] <- paste0(mean_value, '±', se_value)
  }
  
  ret_list <- list("i_rows" = i_rows, "out_df" = out_df
                   # , 'mean_all'=mean_all
                   )
  
  return(ret_list)
}

tmp <- process_svymean_output(out_df,i_rows,'RIDAGEYR',col_index_D,p_names,d)

out_df <- tmp$out_df
i_rows <- tmp$i_rows

tmp <- process_svymean_output(out_df,i_rows,'INDFMPIR',col_index_D,p_names,d)
out_df <- tmp$out_df
i_rows <- tmp$i_rows

tmp <- process_svymean_output(out_df,i_rows,
        'Total.calories.consumed.mean.both.days',col_index_D,p_names,d)
out_df <- tmp$out_df
i_rows <- tmp$i_rows

tmp <- process_svymean_output(out_df,i_rows, 'BMXBMI',col_index_D,p_names,d)
out_df <- tmp$out_df
i_rows <- tmp$i_rows

i_rows <- 5
var_to_count <- 'female'
for (var_to_count in c('female','white','black','mexican','other_hispanic')) {
  i_rows <- i_rows + 1
  out_df[i_rows, 'var'] <- var_to_count
  
  # p_name <- "p.50"
  for (p_name in p_names) {
    p_count_subj <- sum(
      d$variables[d$variables[col_index_D] == p_name, var_to_count]
    )
    
    p_perc_subj = round(p_count_subj / as.integer(out_df[1, p_name]), 2)
    
    out_df[i_rows, p_name] <- paste0(p_count_subj, ' (', p_perc_subj, ')')
  }
}

file_name <- paste0(
  'output_console/EWAS_survery_desc_stats_',
  col_index_D, '_cohort_', survey_year, '_', 
  file_name_postfix, '.csv')

write.csv(out_df, file_name)
# View(out_df)

print(file_name)


