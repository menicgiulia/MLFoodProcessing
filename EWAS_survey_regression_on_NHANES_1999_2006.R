# increase console log limit
options(max.print=1000000)
rm(list = ls())

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


# library("xlsx") No need anymore xls and xlsx have hard limit on max umber of chars in a cell...
# Run R.version and if you see x86_64 you need to install Java 64 bit
# https://java.com/en/download/manual.jsp

`%notin%` <- Negate(`%in%`)

current_dir_path = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_dir_path)

source('EWAS_analysis_base_functions.R')

######################################################################
########### Settings 1 ############
######################################################################

# This will load all independent variables from Patel's dataset
source('EWAS_analysis_Patel_variables.R')

only_work_on_selected_vars <- TRUE

# Select on which nutrient panel the analysis work on
nut_panel = c('12', '58')[2]

dir_reg_analysis <- c(
  paste0('caloric_intake_PSJ1', '_', nut_panel, '_nuts'),
  paste0('caloric_intake_PSJ1', '_', nut_panel, '_nuts_temp'),
  paste0('caloric_intake_PSJ1', '_', nut_panel, '_nuts_y234')
  #
)[1]

cat('Path to reg analysis:', bold(dir_reg_analysis), '\n')

survey_year <- 'all'

#### +-+-+-+- IMPORTAN If set to 1 it WILL NOT RUN regressions
generate_desciptive_statistics <- 0
debug_run <- TRUE

# log <- TRUE
# survey_year_code <- 4
# var <- 'LBXV1A' # Patel marked this is binary. var_desc: Blood 1,1-Dichloroethane (ng/mL)
# var <- 'LBXGLU' # 'PHAFSTHR'
# is_binary_or_categorical_var(var, df, survey_year_code, TRUE)

    ##########################################
    # Select Diet Data Here
    ##########################################

# path_diet_data = paste0('all_diet_data_1999_2006_',nut_panel,'_nuts_Processing index J1.csv')
# path_diet_data = paste0('all_diet_data_1999_2006_',nut_panel,'_nuts_single_and_ensemble_FPro.csv')

# path_diet_data <- 'all_diet_data_1999_2006_58_nuts_ens_FPS.csv'
path_diet_data <- 'input_data/all_diet_data_1999_2006_58_nuts_single_and_ensemble_FPro.csv'
# path_diet_data <- 'all_diet_data_1999_2018_58_nuts_single_and_ensemble_FPro.csv'
# path_diet_data <- 'all_diet_data_1999_2006_58_nuts_single_and_ensemble_FPro__FNDDS9906_C2009.csv'

nhanesCCNR <- read.csv(path_diet_data)
# table(nhanesCCNR$metabolic.syndrome.examination.and.drug, exclude = NULL)

cat(bold('Diet Data File Name: ', current_dir_path, '/', path_diet_data, sep=''), '\n')

load('input_data/nh_99-06.Rdata')
# we added custom vars like t2d so read it from here
VarDescription <- read_excel('input_data/EWAS_VarDescription.xlsx')
VarDescription <- VarDescription %>% mutate_if(is.character, list(~na_if(.,"NA"))) 

if (FALSE){
  # WHYYYYYYYYYYYYY THESE TWO ARE NOT EQUAL!!!!!!!!!!!
  VarDescription[(VarDescription$var == 'LBXV1A') & (VarDescription$series_num == 3), ] 
  VarDescription[(VarDescription$var == 'LBXV1A') && (VarDescription$series_num == 3), ] # RETURN EMPTY!!!!
}

if (only_work_on_selected_vars == TRUE){
  selected_vars_CCNR <- read_excel("input_data/EWAS_exposome_CCNR_selection_modules.xlsx")
  
  selected_vars_CCNR <- selected_vars_CCNR %>%
    dplyr::filter(CCNR_selected == 1)
  
  
  resp_vars_to_work_on <- unique(c(
    response_vars$custom_variables_by_CCNR,
    selected_vars_CCNR$var
  ))
} else{
  # Run regressions on all variable (both custom CCNR and Patel)
  resp_vars_to_work_on <- unique(VarDescription$var)
}


total_independend_vars <- length(resp_vars_to_work_on)

MainTable <- merge(x = MainTable, 
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

nrow(MainTable)

####################################################################
# Custom vars by CCNR 
####################################################################

MainTable$t2d <- I(MainTable$LBXGLU >= 126)
MainTable$metabolic_syndrome_examination <- MainTable$metabolic.syndrome.only.examination
MainTable$metabolic_syndrome <- MainTable$metabolic.syndrome.examination.and.drug

# keep age in its current form because it will be normalized
MainTable$age <- MainTable$RIDAGEYR

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
  
  cat(bold('Number of rows with weight=0 that will be removed:'), 
      nrow(MainTable[MainTable$WTMEC8YR == 0, ]), '\n')
  
  nhanesDesign <- svydesign(id      = ~SDMVPSU, 
                   strata  = ~SDMVSTRA, 
                   weights = ~WTMEC8YR, # Use 8 year weights
                   nest    =T,
                   data    = MainTable
  )
  
  # nrow(nhanesDesign)
  # svymean(~age, nhanesDesign, ci=FALSE)
  #svyby(~age, ~age > 0, design=nhanesDesign, FUN=svymean, ci=TRUE)
  
  sink(paste0("output_console/", dir_reg_analysis, "/R_svydesign_FULL_nhanes.txt")) # Store summary of svydesign
  print(summary(nhanesDesign))
  sink()  # returns output to the console
  
  #### Backup raw ALL DATA
  if (debug_run == TRUE) {
    path_tmp <- paste0('output_console/', dir_reg_analysis,
                       '/nhanesDesign_RAW_ALL_dataset_', dir_reg_analysis, '_cohort_', 
                       survey_year, '.csv')
    write.csv(nhanesDesign$variables, path_tmp)
    cat('Saved RAW ALL Data at: ', bold(path_tmp), '\n')
  }
  ####
  
  
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
  
  sink(paste0("output_console/", dir_reg_analysis, "/R_svydesign_ageDesign_nhanes.txt")) # Store summary of svydesign
  print(summary(ageDesign))
  sink()  # returns output to the console

}

######################################################################
######### End Settings 1 ##########
######################################################################

#DEL EM
if (FALSE){
  svyhist(~manualNOVA4.consumption.kcal.percentage.over.sum.both.days, nhanesDesign)
  svymean(~manualNOVA4.consumption.kcal.percentage.over.sum.both.days, nhanesDesign,
          na.rm=TRUE)
  
  svyhist(~manualNOVA4.consumption.kcal.percentage.over.sum.both.days, 
          nhanesDesign)
  svyhist(~logit_trans(manualNOVA4.consumption.kcal.percentage.over.sum.both.days), 
          nhanesDesign)
  
  svyhist(~ens_FPro.WCDPI.mean.of.both.days.sum, 
          nhanesDesign)
  svyhist(~logit_trans(ens_FPro.WCDPI.mean.of.both.days.sum), 
          nhanesDesign)
  
  
  
  
  box_cox_out = boxcox_trans_return_lambda(
    ageDesign$variables, 'ens_FPro.RW.WFDPI.mean.of.both.days.sum'
  )
  ageDesign$variables$ens_FPro.RW.WFDPI.mean.of.both.days.sum.boxcox = box_cox_out$out
  
  print(paste('lambda for ens_FPro.RW.WFDPI.mean.of.both.days.sum', box_cox_out$lambda))
  
  svyhist(~ens_FPro.RW.WFDPI.mean.of.both.days.sum, 
          ageDesign)
  svyhist(~ens_FPro.RW.WFDPI.mean.of.both.days.sum.boxcox, 
          ageDesign)
  svyhist(~logit_trans(ens_FPro.RW.WFDPI.mean.of.both.days.sum), 
          ageDesign)
  
  svyhist(~manualNOVA4.consumption.kcal.percentage.over.sum.both.days, ageDesign)
  svymean(~manualNOVA4.consumption.kcal.percentage.over.sum.both.days, ageDesign, 
          na.rm=TRUE)
}

######################################################################
########### Settings 2 ############
######################################################################

covar <- c(
  'FPro.WFDPI.mean.of.both.days.sum',    # Diet Processing Score Gram Weighted
  'FPro.RW.WFDPI.mean.of.both.days.sum', # Removed Water - Diet Processing Score Gram Weighted
  'FPro.WCDPI.mean.of.both.days.sum',    # Diet Processing Score Calorie Weighted
  
  "ens_FPro.WFDPI.mean.of.both.days.sum",
  "ens_FPro.RW.WFDPI.mean.of.both.days.sum",
  "ens_FPro.WCDPI.mean.of.both.days.sum",
  
  "ens_min_FPro.WFDPI.mean.of.both.days.sum",
  "ens_min_FPro.RW.WFDPI.mean.of.both.days.sum",
  "ens_min_FPro.WCDPI.mean.of.both.days.sum",
  
  'HEI2015_TOTAL_SCORE',
  #'predNOVA4.consumption.kcal.percentage.over.sum.both.days',
  #'NOVA4.consumption.grams.percentage.over.sum.both.days',
  #'NOVA4.consumption.RW.grams.percentage.over.sum.both.days'
  
  'manualNOVA4.consumption.kcal.percentage.over.sum.both.days'
)

logit_transform_vars <- c(
  # 'framingham_risk_10_years', 
  'ascvd_10y_accaha_lab', 'ascvd_10y_frs_lab', 'ascvd_10y_frs_simple_lab'
)

# Adjusting vars
# 'male', 'other_eth' are not added because of singularities
adj <- c('BMXBMI', 'RIDAGEYR', 'female',
         'INDFMPIR', #poverty income ratio
         'white', 'black', 'mexican', 'other_hispanic'
         ,'Total.calories.consumed.mean.both.days',
         'current_past_smoking' # 0 means never smoked, 1 is past smoker, 2 is currently smoker, none cant identify
         )

# Make sure adjusting vars wont be used as respone variable, 
# it can happen for BMXBMI. Also, use this to ignore a response var if needed!
ignore_resp_vars <- c(adj)

resp_vars_to_work_on <- resp_vars_to_work_on[resp_vars_to_work_on %notin% ignore_resp_vars]

# These variables will be transformed AT MODEL LEVEL.
boxcox_vars <- c(
  covar, 'BMXBMI', 'RIDAGEYR', 
  'INDFMPIR' #  'INDFMPIR' is poverty ratio
)

for (patel_tab in keys(response_vars)){
  for(patel_var in response_vars[[patel_tab]]){
    
    if (patel_var %in% logit_transform_vars){
      next
    }
    
    if(is_binary_or_categorical_var(patel_var, ageDesign$variables, 'all', TRUE) == 0){
      
      # Only work on selected variables!
      if (patel_var %in% resp_vars_to_work_on){
        boxcox_vars <- c(boxcox_vars, patel_var)
      }

    } else{
      cat(blue("Is Binary: ", patel_var), "\n")
    }
  }
}

boxcox_vars <- unique(boxcox_vars)


# If you want to avoid running all tabs in keys(response_vars), 
# you can use this variable to run a selected few, otherwise set it empty.
only_work_on_tabs <- c('Heavy_Metals', 'Any_Disease', 'custom_variables_by_CCNR',
                       'Pharmaceutical Use', 'Blood_Pressure') 

only_work_on_tabs <- c('C_Reactive_Protein', 'Environmental_phenols', 
                       'Total_Cholesterol', 'Urinary_Albumin_and_Creatinine')
only_work_on_tabs <- c('Vitamin_A_E_and_Carotenoids', 'Melamine_Surplus_Urine')
if (TRUE) {only_work_on_tabs <- c()}

######################################################################
######### End Settings 2 ##########
######################################################################

print(paste(
  "Number of non-binary vars to be tranformed by BoxCox (at model level): ", 
  length(boxcox_vars)))

# Apply z transformation on these vars
scale_vars <- unique(c(boxcox_vars, logit_transform_vars))

print(paste(
  "Number of non-binary vars to be centered by Z-transformation",
  "(at model level after BoxCox or logit transformation): ", 
  length(scale_vars)))

##################################

# Backup ageDesign data
if (TRUE && debug_run == TRUE) {
  path_tmp <- paste0('output_console/', dir_reg_analysis,
                     '/ageDesign_dataset_', dir_reg_analysis, '_cohort_', 
                     survey_year, '.csv')
  write.csv(ageDesign$variables, path_tmp)
  cat('Saved ageDesign dataset at: ', bold(path_tmp), '\n')
}

####################################################################
################### Analyze (Run Regressions)#######################
####################################################################


# Check you dont get empty subset
cat(bold('----------------- Year: '), survey_year, 
    bold(' Subjects {nrow(ageDesign)}: '), nrow(ageDesign), '\n') 


table(ageDesign$variables$current_past_smoking)
sum(is.na(ageDesign$variables$current_past_smoking))

resp_var_done_regression <- c()

boxcox_lambda_df <- data.frame(matrix(ncol = 3))
colnames(boxcox_lambda_df) <- c(
  'resp_var', 'var', 'lambda')
boxcox_lambda_i <- 1

j = 0

time_start_regs <- Sys.time()

#module_file_name <- keys(response_vars)[1]
#module_file_name <- 'custom_variables_by_CCNR'
#module_file_name <- 'Blood_Pressure'
#module_file_name <- 'Total_Cholesterol'
#module_file_name <- only_work_on_tabs[2]
for (module_file_name in keys(response_vars)) {
  skip = FALSE
  if(length(only_work_on_tabs) > 0){
    skip = TRUE
    if (module_file_name %in% only_work_on_tabs){
      skip = FALSE
    }
  }
  
  if (skip == TRUE) { next }
  
  file_name <- module_file_name
  
  cat(bold("\n\n**********WORKING ON TAB:", file_name, ' & year: ', 
           survey_year, ' **********'), '\n')
  
  response_vars_tab <- response_vars[[module_file_name]]
  
  #########
  
  out_df <- data.frame(matrix(ncol = 16))
  colnames(out_df) <- c(
    'resp_var', 'resp_var_type', 'N', 'NA_count', 
    'covariate', 'reg_family', 'num_covars', 
    'unique_val_counts', 'value_counts',
    'coef','std_error', 't_value', 'p_val',
    'dispersion', 'coefficients', 'summary')
  i <- 1
  
  # resp_var <- c('LBXTHG', 'prostate_cancer_self_report')[2] #DELME !!
  # resp_var <- response_vars_tab[3]
  for (resp_var in response_vars_tab){
    # Only work on the selected variables
    if (resp_var %notin% resp_vars_to_work_on){ next; }
  
    ###############
    #Do not repeat regressions for a variable
    ###############
    
    if(TRUE){
      if (resp_var %in% resp_var_done_regression){
        cat(bold(blue('Already done regressions for respone variable')), 
            bold(resp_var), '\n')
        next;
      }
      
      resp_var_done_regression <- c(resp_var_done_regression, resp_var)
    }

    ##########################################
    
    phenotypeDesign <- subset(ageDesign, 
                              is.na(ageDesign$variables[[resp_var]]) == FALSE & 
                                is.na(INDFMPIR) == FALSE
    )
    
    # nrow(phenotypeDesign)
    
    resp_var_subset = data.table::copy(phenotypeDesign$variables)
    

    cat(bold(
      '\n+++++++++[STATS] Response Var:', resp_var, '| Num Subjects:' ,
       nrow(phenotypeDesign)
      ), blue(
       '\nAFTER REMOVING  subject with NA socio-economic status (NDFMPIR):',
       red(
         nrow(ageDesign$variables %>%
                filter(!is.na(ageDesign$variables[[resp_var]]) & is.na(INDFMPIR)))
       )
     ), '+++++++++\n\n')
    
    ################################################
    ## Transformations for this model
    ################################################
    
    reg_all_vars = c(resp_var, covar, adj)
    
    #var_tmp <- reg_all_vars[1]
    for (var_tmp in reg_all_vars) {
      if (var_tmp %in% boxcox_vars){
        tryCatch(
          {
          boxcox_trans_out <- boxcox_trans_return_lambda(
            phenotypeDesign$variables, var_tmp
          )
          
          phenotypeDesign$variables[[var_tmp]] <- boxcox_trans_out$out[,1]
          
          boxcox_lambda_df[boxcox_lambda_i, 'resp_var'] <- resp_var
          boxcox_lambda_df[boxcox_lambda_i, 'var'] <- var_tmp
          boxcox_lambda_df[boxcox_lambda_i, 'lambda'] <- boxcox_trans_out$lambda
          
          boxcox_lambda_i <- boxcox_lambda_i + 1
          
          cat(bold('[Tranform BoxCox] '), 'on var:', blue(var_tmp),
              'lambda', boxcox_trans_out$lambda, '\n')
          },
          error=function(error_message) {
            # message(error_message)
            cat(red(bold(
              "!!! BoxCox Failed !!! VarName:", var_tmp
            ))
              # , 'error_message:', error_message
            , '\n'
            )
            
            cat(red("This variable might be empty; length(unique(", var_tmp, "))=", 
                    length(unique(phenotypeDesign$variables[[var_tmp]]))
                    ), ';\n')
            return(NA)
          }
        )
      }
    }
    
    for (var_tmp in reg_all_vars) {
      if (var_tmp %in% logit_transform_vars){
        tryCatch(
          {
            phenotypeDesign$variables[[var_tmp]] <- logit_trans(
              phenotypeDesign$variables[[var_tmp]]
            )
            
            cat(bold('[Tranform Logit] '), 'on var:', blue(var_tmp), '\n')
          },
          error=function(error_message) {
            message(paste("!!! logit_trans Failed !!! VarName: ", var_tmp))
            cat(red("This variable might be empty: unique(", var_tmp, ")=", 
                    unique(phenotypeDesign$variables[[var_tmp]])), '\n')
            message(error_message)
            return(NA)
          }
        )
      }
    }
    
    for (var_tmp in reg_all_vars) {
      if (var_tmp %in% scale_vars){
        tryCatch(
          {
            phenotypeDesign$variables[[var_tmp]] <- scale(
              phenotypeDesign$variables[[var_tmp]], center = TRUE, scale = TRUE
            )
            
            cat(bold('[Tranform Scale] '), 'on var:', blue(var_tmp), '\n')
          },
          error=function(error_message){
            message(paste("!!! Z-Transformation Failed !!! VarName: ", var_tmp))
            cat(red("This variable might be empty: unique(", var_tmp, ")=", unique(
              MainTable_subset[[var_tmp]])), '\n')
            message(error_message)
            return(NA)
          }
        )
      }
    }
    
    ################################################
    ################################################
    ################################################
    
    # cov_ <- covar[1]
    for (cov_ in covar){
      
      out_df[i, 'resp_var'] <- resp_var
      out_df[i, 'N'] <- nrow(phenotypeDesign)
      out_df[i, 'NA_count'] <- nrow(
        ageDesign$variables[is.na(ageDesign$variables[[resp_var]]), ]
      )
      
      out_df[i, 'covariate'] <- cov_
      
      out_df[i, 'unique_val_counts'] <- length(unique(phenotypeDesign$variables[[resp_var]]))
      
      # Check if an adjusting variable is binary convert it to factor
      adj_vars_prepped = c()
      
      # adj_var <- adj[1]
      for(adj_var in adj) {
        
        adj_var_type <- is_binary_or_categorical_var(adj_var, resp_var_subset, survey_year, FALSE)
        # print(paste(adj_var, adj_var_type))

        if (adj_var_type > 0){
          ##########################################
          # TODO MAYBE filter a covar if it has not enough levels.
          # adj_var_length <- length(unique(phenotypeDesign$variables[[adj_var]]))
          # in other words, put condition on 'adj_var_length'
          ##########################################
          
          if(length(unique(phenotypeDesign$variables[[adj_var]])) > 1 ){
            adj_vars_prepped <- c(adj_vars_prepped, paste0('factor(', adj_var, ')'))
          } else {
            cat(bold('!!! Adjusting var "', adj_var, 
                     '" removed because not enough levels to be factored.'), '\n')
          }
        }else{
          adj_vars_prepped <- c(adj_vars_prepped, adj_var)
        }
      }
      
      ######
      # Check if independent variable is binary, convert it to factor. 
      # Use MainTable_subset to assess in the whole dataset not a subset
      ######
      resp_var_type <- is_binary_or_categorical_var(resp_var, resp_var_subset, survey_year, TRUE)
      out_df[i, 'resp_var_type'] <- resp_var_type
      
      if (resp_var_type > 0){
         doForm <- as.formula(paste0(
           'factor(', resp_var, ')', '~', paste(c(cov_, adj_vars_prepped), collapse = '+')
         ))
         
         ##############
         value_counts <- as.data.frame(table(phenotypeDesign$variables[[resp_var]]))
         names(value_counts) <- substring(names(value_counts), first = 1, last = 1)
         value_counts <- value_counts[order(-value_counts$F),]
         out_df[i, 'value_counts'] <- capture_output(toJSON(value_counts), width=800, print=TRUE)
         
      } else {
        doForm <- as.formula(paste(resp_var, '~', paste(c(cov_, adj_vars_prepped), collapse = '+')))
        
        ############## Store value count for numerical variables as well
        value_counts <- as.data.frame(table(phenotypeDesign$variables[[resp_var]]))
        names(value_counts) <- substring(names(value_counts), first = 1, last = 1)
        value_counts <- value_counts[order(-value_counts$F),]
        out_df[i, 'value_counts'] <- capture_output(toJSON(value_counts), width=800, print=TRUE)
      }
      
      out_df[i, 'num_covars'] <- length(adj_vars_prepped) + 1
      print(doForm)
      
      reg_family = gaussian()
      
      if(resp_var_type > 0){
        reg_family = quasibinomial(link = logit)
      }
      
      out_df[i, 'reg_family'] <- trimws(capture_output(reg_family, width=800, print=TRUE))
      
      tryCatch(
        {
          reg <- svyglm(formula = doForm , design=phenotypeDesign, family=reg_family)
          
          reg_sum <- summary(reg)
          
          out_df[i, 'coef'] <- reg_sum$coefficients[2,][1]
          out_df[i, 'std_error'] <- reg_sum$coefficients[2,][2]
          out_df[i, 't_value'] <- reg_sum$coefficients[2,][3]
          out_df[i, 'p_val'] <- reg_sum$coefficients[2,][4]
          
          last_reg_output <- paste(
            capture_output(doForm, width=800, print=TRUE),
            capture_output(reg_sum,  width = 800, print=TRUE),
            sep = "\n"
          )
          
          # Save all output of regression
          out_df[i, 'summary'] <- last_reg_output
          
          ############# Save Coef ############

          out_df[i, 'coefficients'] <- toJSON(
            as.data.frame(reg_sum$coefficients), 
            digits=10
          )

          out_df[i, 'dispersion'] <- reg_sum$dispersion
        },
        error=function(error_message) {
          message(paste("!!! ERROR !!!!"))
          cat(red(bold(error_message)))
          
          out_df[i, 'summary'] <- paste(error_message, sep = "\n")
          return(NA)
        }
      )

      i <- i + 1
      j <- j + 1
      
      if (j %% 10 == 0){
        cat(bold(blue(
          #round(j/(total_independend_vars * length(covar)), 3) * 100, 
          round(j/(1577 * length(covar)), 3) * 100, ## see below comments why I used 1577!
          '% of regressions (',
          (total_independend_vars * length(covar)),
          'total) completed from survey year ', survey_year , '...\n'
        )))
      }
    }
  }
  
  out_df$sig <- out_df$p_val <= 0.05
  round_df(out_df, 3)
  
  write.csv(out_df, paste0('output_console/', dir_reg_analysis, '/',
                           survey_year ,'/reg_analysis_boxcox_', file_name , '.csv'))
  
  print(paste0('output_console/', dir_reg_analysis, '/',
               survey_year ,'/reg_analysis_boxcox_', file_name , '.csv'))
}



cat('########## DONE REGRESSIONS ##############\n')

path_lambda_boxcox <- paste0('output_console/', dir_reg_analysis,
                             '/ageDesign_lambda_boxcox_cohort_', survey_year, '.csv')
cat(bold('EXPORT Lambda Box Cox --> ', path_lambda_boxcox), '\n')
write.csv(boxcox_lambda_df, path_lambda_boxcox)


cat('Regs started:', format(time_start_regs), 'and ended:',
    format(Sys.time())
)

