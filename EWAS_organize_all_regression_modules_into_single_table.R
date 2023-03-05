# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# install.packages("hexbin")
# BiocManager::install("IHW")
# BiocManager::install("airway")

options(max.print=1000000)
rm(list = ls())

library(crayon)
library("DESeq2")
library("dplyr")
library(hash)
library("IHW")
library("ggplot2")
library(readxl)

`%notin%` <- Negate(`%in%`)

current_dir_path = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_dir_path)

metrics <- hash()

######################################################################
########### Settings ############
######################################################################

path_CCNR_var_module_selection <- paste0(current_dir_path, "/input_data/EWAS_exposome_CCNR_selection_modules.xlsx")
CCNR_var_module_selection <- readxl::read_excel(path_CCNR_var_module_selection)


vars_not_have_min_samples <- c()

if (FALSE){
  min_sample_needed <- read.csv('binary_cat_vars_min_sample_needed.csv')
  
  min_sample_needed$has_min_samples <- min_sample_needed$N > min_sample_needed$Nest
  
  vars_not_have_min_samples <- min_sample_needed[min_sample_needed$has_min_samples == FALSE, 'var']
}

cohorts <- c(234) # c(1,2,3,4,234,0)
cohorts <- c(0) # 0 means ALL cohorts (1999-2006)

nbins_IHW <- 2

export_folder <- "IHW_metric_then_cohort/"

file_name <- "_rejected_caloric_intake_"

path_base_regs <- 'output_console/'
path_reg_experiment <- 'caloric_intake_PSJ1_58_nuts'

path_reg_analysis <- paste0(path_base_regs, path_reg_experiment, '/')


# path_sample_stats <- paste0(path_reg_analysis, 'sample_stats_for_', path_reg_experiment, '.csv')
path_sample_stats <- paste0('output_console/caloric_intake_PSJ1_58_nuts/sample_stats_for_caloric_intake_PSJ1_58_nuts.csv')
sample_stats <- read.csv(path_sample_stats)

metrics[["HEI15"]] <- read.csv(paste0(
  path_reg_analysis,
  "vars_regressions_table_HEI-15 caloric_intake_PSJ1_58_nuts.csv"))

if (TRUE){
  metrics[["FPro.WFDPI"]] <- read.csv(paste0(path_reg_analysis,
                                             "vars_regressions_table_FPro.WFDPI caloric_intake_PSJ1_58_nuts.csv"))
  metrics[["FPro.WFDPI.RW"]] <- read.csv(paste0(path_reg_analysis,
                                                "vars_regressions_table_FPro.RW.WFDPI caloric_intake_PSJ1_58_nuts.csv"))
  metrics[["FPro.WCDPI"]] <- read.csv(paste0(path_reg_analysis,
                                             "vars_regressions_table_FPro.WCDPI caloric_intake_PSJ1_58_nuts.csv"))
}

metrics[["ens_FPro.WFDPI"]] <- read.csv(paste0(path_reg_analysis,
                                               "vars_regressions_table_ens_FPro.WFDPI caloric_intake_PSJ1_58_nuts.csv"))
metrics[["ens_FPro.WFDPI.RW"]] <- read.csv(paste0(path_reg_analysis,
                                                  "vars_regressions_table_ens_FPro.RW.WFDPI caloric_intake_PSJ1_58_nuts.csv"))
metrics[["ens_FPro.WCDPI"]] <- read.csv(paste0(path_reg_analysis,
                                               "vars_regressions_table_ens_FPro.WCDPI caloric_intake_PSJ1_58_nuts.csv"))

if (TRUE){
  metrics[["ens_min_FPro.WFDPI"]] <- read.csv(paste0(path_reg_analysis,
                                                     "vars_regressions_table_ens_min_FPro.WFDPI caloric_intake_PSJ1_58_nuts.csv"))
  metrics[["ens_min_FPro.WFDPI.RW"]] <- read.csv(paste0(path_reg_analysis,
                                                        "vars_regressions_table_ens_min_FPro.RW.WFDPI caloric_intake_PSJ1_58_nuts.csv"))
  metrics[["ens_min_FPro.WCDPI"]] <- read.csv(paste0(path_reg_analysis,
                                                     "vars_regressions_table_ens_min_FPro.WCDPI caloric_intake_PSJ1_58_nuts.csv"))
}

metrics[["manualNOVA4.kcal"]] <- read.csv(paste0(path_reg_analysis,
                                                 "vars_regressions_table_manualNOVA4.Kcal caloric_intake_PSJ1_58_nuts.csv"))

# head(metrics[["NOVA4.Kcal"]])

######################################################################
########### End Settings ############
######################################################################


if (FALSE){
  i <- 1
  for (module in unique(df_m$module.Patel)){
    cat(paste0('modules_num[["', module, '"]] <-',i), '\n')
    
    i <- i + 1
  }
}

source('EWAS_modules_num_dict.R')

# metric_name <- 'FPro.WFDPI'
for (metric_name in keys(metrics)) {
  
  df_m <- metrics[[metric_name]]
  
  df_m.cohort <- data.frame(matrix(ncol = 12))
  colnames(df_m.cohort) <- c("var","cohort", "module", 
                             'is.categorical', "pvalue", "N", "coef",
                             'adj_pvalue_IHW', 'weights_IHW', 'nbins_IHW',
                             'adj_pvalue_BH', 'weights_BH')
  i=1
  
  # cohort <- 0
  for (cohort in cohorts){
    for (row in 1:nrow(df_m)){
      var_name <- df_m[row, 'var']
      
      if (FALSE) {
        if (var_name %in% vars_not_have_min_samples) {
          cat('Var deleted not having enough samples: ', var_name, '\n')
          next
        }
      }
      
      df_m.cohort[i, 'var'] <- var_name
      df_m.cohort[i, 'cohort'] <- cohort
      
      module_name <- df_m[row, 'module.Patel']
      if (module_name %notin% keys(modules_num)) {
        cat(red('Error: Module "', module_name, '" not in dictionary modules_num.'), '\n')
      }
      df_m.cohort[i, 'module'] <- modules_num[[module_name]]
      
      df_m.cohort[i, 'is.categorical'] <- as.integer(df_m[row, 'is.categorical'] > 0)
      
      df_m.cohort[i, 'pvalue'] <- df_m[row, paste0('P', cohort)]
      df_m.cohort[i, 'N'] <- df_m[row, paste0('N', cohort)]
      
      df_m.cohort[i, 'coef'] <- df_m[row, paste0('B', cohort)]
      
      i <- i+ 1
    }
    
    num_vars_all = nrow(df_m.cohort)
    df_m.cohort = df_m.cohort[!is.na(df_m.cohort['pvalue']), ]
    # View(df_m.cohort)
    
    ###
    # Filter variables from un-related modules such as 'Housing Characteristics' and
    # filer categorical variables that do not have the minimum number of sample.
    ###
    
    ###### NOT selected by CCNR selection
    nrow(df_m.cohort)
    mask.only_CCNR_selected_vars <- df_m.cohort$var %in% CCNR_var_module_selection[
      CCNR_var_module_selection$CCNR_selected == 1,
    ]$var
    
    cat(green(italic('[',metric_name,'Cohort:',cohort, 'NBins:', nbins_IHW,
                     '] Number of vars removed because ', bold('NOT selected'),
                     ' in CCNR selection: ',
                     sum(!(mask.only_CCNR_selected_vars)), ' from ',
                     nrow(df_m.cohort)
    )), '\n')
    
    df_m.cohort_f <- df_m.cohort[mask.only_CCNR_selected_vars, ]
    
    ###### NOT passed min sample size condition
    # DONT DO THIS PART Giulia wrote the code to take care sample size conditions
    if (FALSE) {
      
      mask.filer_sample_size_condition <- df_m.cohort_f$var %notin% sample_stats[
        (is.na(sample_stats$passed_condition_for_min_sample_size) == FALSE) &
          (sample_stats$passed_condition_for_min_sample_size == FALSE),
      ]$var
      
      ##
      # TODO THIS NEEDS TO ADDRESS MIN SAMPLE SIZE FOR CONTINUES VARS AS WELL
      #      CURRNTLY PYTHON HANDLES THIS YOU NEED TO IMPLIMENT AND DOUBLE CHECK
      ##
      cat(green(italic('[',metric_name,'Cohort:',cohort, 'NBins:', nbins_IHW,
                       '] Number of vars removed because ', 
                       bold('NOT passed min sample size condition'),
                       ' in CCNR selection: ',
                       sum(!(mask.filer_sample_size_condition)), ' from ',
                       nrow(df_m.cohort_f)
      )), '\n')
      
      df_m.cohort_f <- df_m.cohort_f[mask.filer_sample_size_condition, ]
    }
    
    ###### NOT present in at least two cohorts
    
    mask.filter_present_min_two_cohorts <- df_m.cohort_f$var %in% sample_stats[
      sample_stats$num_series_present > 1,
    ]$var
    
    cat(green(italic('[',metric_name,'Cohort:',cohort, 'NBins:', nbins_IHW,
                     '] Number of vars removed because ', bold('NOT present in min two cohorts'),
                     ' in passed samples size condition and CCNR selection: ',
                     sum(!(mask.filter_present_min_two_cohorts)), ' from ',
                     nrow(df_m.cohort_f)
    )), '\n')
    
    df_m.cohort_f <- df_m.cohort_f[mask.filter_present_min_two_cohorts, ]
    
    ######
    
    df_m.cohort <- df_m.cohort_f
    
    cat(green(italic('[',metric_name,'Cohort:',cohort, 'NBins:', nbins_IHW,
                     '] Number of variables:', nrow(df_m.cohort), 
                     '. Removed ', num_vars_all - nrow(df_m.cohort),
                     'from', num_vars_all)), '\n')
    
    ##########################################################################################
    ## Trying to consider that some p-values dont exists because not all regressions worked
    ## (https://bioconductor.org/packages/release/bioc/vignettes/IHW/inst/doc/introduction_to_ihw.html)
    ##########################################################################################
    
    nbins <- nbins_IHW
    
    sim <- mutate(df_m.cohort, 
                  group = groups_by_filter(N + is.categorical + module, nbins)
    )
    
    # nrow(sim[sim$group == 1,])
    
    m_groups  <- table(sim$group)
    nrow(sim)
    
    if (nrow(sim) == 0) {
      # Complete code to not do IHW. This is in case no var in module passed the condition of being present in at least two cohorts! Its not the case in the selected modules
    }
      
    
    cat('     IHW - run IHW with defined bins:\n')
    ihw_res <- ihw(pvalue ~ group, alpha = 0.1, data = sim, 
                   m_groups = m_groups, covariate_type='nominal')
    
    df_m.cohort['adj_pvalue_IHW'] <- adj_pvalues(ihw_res)
    df_m.cohort['weights_IHW'] <- weights(ihw_res)
    df_m.cohort['nbins_IHW'] <- nbins
    
    if (F){
      df_m.cohort$weights_IHW <- weights(ihw_res)
      
      
      modules_num_df <- data.frame(matrix(ncol = 2))
      colnames(modules_num_df) <- c("number","name")
      j <- 1
      
      num_to_module <- hash()
      for (k in keys(modules_num)){
        modules_num_df[j, 'number'] <- modules_num[[k]]
        modules_num_df[j, 'name'] <- k
        
        j <- j + 1
        
        num_to_module[ modules_num[[k]] ] <- k
      }
      
      write.csv(modules_num_df, "IHW_model_patel_to_CCNR_number.csv")
      
      f = function(x){
        tmp <- num_to_module[[paste0(x[3])]]
        #cat(type(tmp))
        return (paste0(tmp, ' '))
      }
      
      df_m.cohort$module_name <- apply(df_m.cohort, 1, f)
      
      df_m.cohort$weights_IHW
      df_m.cohort$module_name
      
      gg <- ggplot(df_m.cohort,
                   aes(x = module_name, y=weights_IHW)) + 
        geom_bar(stat='identity')  + coord_flip()
    }
    
    #########################################################################################
    ## BH
    #########################################################################################
    cat('     BH - run IHW with no predefined bins:\n')
    
    BH <- ihw(pvalue ~ N + module + is.categorical, alpha = 0.1, data = df_m.cohort)
    
    cat(blue(bold("Compare number of rejections for BH:", rejections(BH), 
                  "IHW: ", rejections(ihw_res))), '\n'
    )
    
    df_m.cohort['adj_pvalue_BH'] <- adj_pvalues(BH)
    df_m.cohort['weights_BH'] <- weights(BH)
    
    ##############################
    ###############Export BH
    ##############################
    ihw_method_to_export <- "BH"
    ihwRes <- BH
    
    cat(bold("Method:", ihw_method_to_export, "Rejections:", rejections(ihwRes), 
             "/", nrow(df_m.cohort[df_m.cohort$pvalue < 0.05, ]), 'sig',
             '(all-data',nrow(df_m.cohort),')',
             "Metric:", metric_name, 
             "Experiment:", path_reg_analysis), '\n'
    )
    
    export_file_name <- paste0(path_reg_analysis, '/', export_folder,
                               'nbins_', nbins, '_', file_name , metric_name , '.csv')
    
    write.csv(df_m.cohort, export_file_name)
    cat(magenta(export_file_name, '\n'))
    
    ##############################
    ####### Charts
    ##############################
    
    if (FALSE){
      plot(ihw_res)
      ggsave(paste0(export_file_name,'_plot_1.png'),width=20,height=12,dpi=150)
      
      
      qplot(adj_pvalues(BH), adj_pvalues(ihw_res), cex = I(0.2), 
            xlim = c(0, 0.1), ylim = c(0, 0.1)) + coord_fixed()
      ggsave(paste0(export_file_name,'_plot_2.png'),width=10,height=8,dpi=150)
      
      
      plot(BH, what = "decisionboundary")
      ggsave(paste0(export_file_name,'_plot_3_BH_decision_boundary.png'),width=10,height=8,dpi=150)
      
      
      gg <- ggplot(as.data.frame(ihw_res),
                   aes(x = pvalue, y = adj_pvalue, col = group)) + 
        geom_point(size = 1.0) + 
        scale_colour_hue(l = 70, c = 150, drop = FALSE)
      gg
      ggsave(paste0(export_file_name,'_plot_4.png'),width=10,height=8,dpi=150)
      
      
      gg <- ggplot(subset(as.data.frame(ihw_res), adj_pvalue <= 0.2), 
                   aes(x = pvalue, y = adj_pvalue, col = group)) + 
        geom_point(size = 1.0) + 
        scale_colour_hue(l = 70, c = 150, drop = FALSE)
      gg
      ggsave(paste0(export_file_name,'_plot_5.png'),width=10,height=8,dpi=150)
    }
    
    if (FALSE){
      export_file_name <- paste0(path_reg_analysis, '/', export_folder,
                                 'nbins_', nbins, '_',
                                 ihw_method_to_export, file_name , metric_name , '.csv')
      ## Export rejected p values
      write.csv(df_m.cohort[rejected_hypotheses(ihwRes), ], export_file_name)
      cat(magenta(export_file_name, '\n'))
      
      ##############################
      ###############Export IHW
      ##############################
      ihw_method_to_export <- "IHW"  
      ihwRes <- ihw_res
      
      cat(bold("Method:", ihw_method_to_export, "Rejections:", rejections(ihwRes), 
               "/", nrow(df_m.cohort[df_m.cohort$pvalue < 0.05, ]), 'sig',
               '(all-data',nrow(df_m.cohort),')',
               "Metric:", metric_name, 
               "Experiment:", path_reg_analysis), '\n'
      )
      
      # cat(bold("Method:", ihw_method_to_export, "Rejections:", rejections(ihwRes), 
      #             "/", nrow(df_m.cohort), "Metric:", metric_name, 
      #             "Experiment:", path_reg_analysis), '\n'
      # )
      export_file_name <- paste0(path_reg_analysis, '/', export_folder,
                                 'nbins_', nbins, '_',
                                 ihw_method_to_export, file_name , metric_name , '.csv')
      ## Export rejected p values
      write.csv(df_m.cohort[rejected_hypotheses(ihwRes), ], export_file_name)
      cat(magenta(export_file_name, '\n'))
    }
  }
}



