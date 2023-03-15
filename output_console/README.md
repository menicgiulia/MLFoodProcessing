# This folder is used for conducting the EWAS study

For the sake of an example on how te reproduce the EWAS study, we uploaded the output of the EWAS pipline on the following modules: 'Heavy_Metals', 'Any_Disease', 'custom_variables_by_CCNR',
                       'Pharmaceutical Use', 'Blood_Pressure'.

In order to run the EWAS study, please start from running "EWAS_survey_regression_on_NHANES_1999_2006.R" and following the comments in it for various option the survey design and selection of independent variables and modules.

Then, please run "EWAS_merge_regression_analysis_from_R.ipynb" notebook, only running section 1, 2, and 3.

Then, please run the "EWAS_organize_all_regression_modules_into_single_table.R" for organizing all regression and also it runs the Independent Hypothesis Weighting (https://bioconductor.org/packages/release/bioc/html/IHW.html) for adjusting p-values. BUT, in the end we decided to not use IHW in final analaysis and used Benjamini–Hochberg method since we did not have enough tests (need to have  over 1000) to perform IHW, hence IHW resulted in versy similar adjusment as BH.

Lastly, please only run sections 4 and 5 of the  "EWAS_merge_regression_analysis_from_R.ipynb" to get the final daa files that are currently provided in https://github.com/menicgiulia/MLFoodProcessing/tree/main/input_data

# The R Environment Info


```
> sessionInfo()

R version 4.0.3 (2020-10-10)
Platform: x86_64-w64-mingw32/x64 (64-bit)
Running under: Windows 10 x64 (build 22621)

Matrix products: default

locale:
[1] LC_COLLATE=English_United States.1252 
[2] LC_CTYPE=English_United States.1252   
[3] LC_MONETARY=English_United States.1252
[4] LC_NUMERIC=C                          
[5] LC_TIME=English_United States.1252    

attached base packages:
 [1] parallel  stats4    grid      stats     graphics  grDevices utils    
 [8] datasets  methods   base     

other attached packages:
 [1] DESeq2_1.28.1               SummarizedExperiment_1.18.2
 [3] DelayedArray_0.14.1         matrixStats_0.57.0         
 [5] Biobase_2.48.0              GenomicRanges_1.40.0       
 [7] GenomeInfoDb_1.24.2         IRanges_2.22.2             
 [9] S4Vectors_0.26.1            BiocGenerics_0.34.0        
[11] jsonlite_1.7.1              readxl_1.3.1               
[13] crayon_1.3.4                testthat_3.0.0             
[15] margins_0.3.23              mfx_1.2-2                  
[17] betareg_3.1-4               MASS_7.3-53                
[19] lmtest_0.9-38               zoo_1.8-8                  
[21] sandwich_3.0-0              survey_4.0                 
[23] Matrix_1.2-18               Hmisc_4.4-1                
[25] ggplot2_3.3.2               Formula_1.2-4              
[27] survival_3.2-7              lattice_0.20-41            
[29] car_3.0-10                  carData_3.0-4              
[31] foreach_1.5.1               dplyr_1.0.2                
[33] broom_0.7.2                 hash_2.2.6.1               
[35] IHW_1.16.0                 

loaded via a namespace (and not attached):
 [1] colorspace_2.0-0       ellipsis_0.3.1         modeltools_0.2-23     
 [4] rio_0.5.16             htmlTable_2.1.0        XVector_0.28.0        
 [7] base64enc_0.1-3        rstudioapi_0.13        bit64_4.0.5           
[10] flexmix_2.3-17         AnnotationDbi_1.50.3   codetools_0.2-16      
[13] splines_4.0.3          geneplotter_1.66.0     knitr_1.30            
[16] annotate_1.66.0        cluster_2.1.0          png_0.1-7             
[19] compiler_4.0.3         backports_1.2.0        htmltools_0.5.0       
[22] tools_4.0.3            gtable_0.3.0           glue_1.4.2            
[25] GenomeInfoDbData_1.2.3 Rcpp_1.0.5             slam_0.1-47           
[28] cellranger_1.1.0       vctrs_0.3.5            iterators_1.0.13      
[31] xfun_0.19              stringr_1.4.0          openxlsx_4.2.3        
[34] lifecycle_0.2.0        XML_3.99-0.5           zlibbioc_1.34.0       
[37] scales_1.1.1           hms_0.5.3              RColorBrewer_1.1-2    
[40] curl_4.3               memoise_1.1.0          gridExtra_2.3         
[43] rpart_4.1-15           latticeExtra_0.6-29    stringi_1.5.3         
[46] RSQLite_2.2.1          genefilter_1.70.0      checkmate_2.0.0       
[49] zip_2.1.1              BiocParallel_1.22.0    rlang_0.4.8           
[52] pkgconfig_2.0.3        bitops_1.0-6           lpsymphony_1.16.0     
[55] purrr_0.3.4            prediction_0.3.14      htmlwidgets_1.5.2     
[58] bit_4.0.4              tidyselect_1.1.0       magrittr_2.0.1        
[61] R6_2.5.0               generics_0.1.0         DBI_1.1.0             
[64] pillar_1.4.6           haven_2.3.1            foreign_0.8-80        
[67] withr_2.3.0            abind_1.4-5            RCurl_1.98-1.2        
[70] nnet_7.3-14            tibble_3.0.4           fdrtool_1.2.15        
[73] jpeg_0.1-8.1           locfit_1.5-9.4         data.table_1.13.2     
[76] blob_1.2.1             forcats_0.5.0          digest_0.6.27         
[79] xtable_1.8-4           tidyr_1.1.2            munsell_0.5.0         
[82] mitools_2.4     
```    