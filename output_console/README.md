This folder is used for conducting the EWAS study.

For the sake of an example on how te reproduce the EWAS study, we uploaded the output of the EWAS pipline on the following modules: 'Heavy_Metals', 'Any_Disease', 'custom_variables_by_CCNR',
                       'Pharmaceutical Use', 'Blood_Pressure'.

In order to run the EWAS study, please start from running "EWAS_survey_regression_on_NHANES_1999_2006.R" and following the comments in it for various option the survey design and selection of independent variables and modules.

Then, please run "EWAS_merge_regression_analysis_from_R.ipynb" notebook, only running section 1, 2, and 3.

Then, please run the "EWAS_organize_all_regression_modules_into_single_table.R" for organizing all regression and also it runs the Independent Hypothesis Weighting (https://bioconductor.org/packages/release/bioc/html/IHW.html) for adjusting p-values. BUT, in the end we decided to not use IHW in final analaysis and used Benjamini–Hochberg method since we did not have enough tests (need to have  over 1000) to perform IHW, hence IHW resulted in versy similar adjusment as BH.

Lastly, please only run sections 4 and 5 of the  "EWAS_merge_regression_analysis_from_R.ipynb" to get the final daa files that are currently provided in https://github.com/menicgiulia/MLFoodProcessing/tree/main/input_data
