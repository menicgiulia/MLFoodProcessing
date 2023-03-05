cache_binary_or_categorical_vars <- hash()

is_binary_or_categorical_var <- function(var, df, survey_year_code, log) {
  
  if (var %notin% names(df)){
    cat(red('[Var not exists] Var: ', var), '\n') 
    stop('')
  }
  
  cache_key <- paste(survey_year_code, var)
  
  if (cache_key %in% keys(cache_binary_or_categorical_vars)){
    
    if (log) {
      cat(bold('[CACHE Var Type] Cache Key: "', cache_key ,'" Type: ', 
               cache_binary_or_categorical_vars[[cache_key]], sep=''), '\n')
    }
    
    return(cache_binary_or_categorical_vars[[cache_key]])
  }
  
  # df <- MainTable
  ret = 0
  df <- df[]
  
  ############################
  ######## ALTER IF OTHER SURVEY YEAR COMBINATIONS ARE USED
  ############################
  if (survey_year_code == 'all'){
  } else if (survey_year_code == '234'){
    df <- df[df$SDDSRVYR > 1,]
  } else if (survey_year_code >= 1 && survey_year_code <= 4){
    df <- df[df$SDDSRVYR == survey_year_code,]
  } else{
    stop('giveb surver year code is not valid')
  }
  
  unique_val <- unique(df[[var]])[!is.na(unique(df[[var]]))]
  
  unique_val <- sort(unique_val)
  
  if(length(unique_val) == 2 && all(unique_val == c(0, 1))){
    if (log) { 
      cat(bold('[Var is Binary] (tested) Var: ', var ,
               ' Unique Vals: ', toString(unique_val), sep=''), '\n') 
    }
    ret <- 1
  }
  
  if(length(unique_val) == 1){
    # CAN NOT IDENTIFY TYPE OF THIS VAR BECAUSE IT ONLY HAS ONE UNIQUE VALUE REPORTED
    ret <- 3
  }
  else {
    if (survey_year_code == 'all'){
      var_desc_patel <- VarDescription[VarDescription$var == var, ]
    } else {
      var_desc_patel <- VarDescription[
        (VarDescription$var == var) & (VarDescription$series_num == survey_year_code), ]
      
      if (nrow(var_desc_patel) > 1){
        cat(red(bold("[WARNING] Variable ", var, " has more than one record in varDescription for series: ", survey_year_code, sep='')))
      }
    }
    
    cat_levels <- var_desc_patel$categorical_levels
    
    if (nrow(var_desc_patel) > 0){
      if (1 %in% var_desc_patel$is_binary){
        if (log) { 
          cat(bold('[Var is Binary] (Patel marked) Var: ', var ,
                   ' Unique Vals [#', length(unique_val) ,']: ',
                   toStringVector(unique_val), sep=''), '\n')
        }
        ret <- 2
      } else if (1 %in% var_desc_patel$is_ordinal){
        if (log) { 
          cat(bold('[Var is Ordinal] (Patel marked) Var: ', var,
                   ' Unique Vals [#', length(unique_val) ,']: ',
                   toStringVector(unique_val), sep=''), '\n') 
        }
        ret <- 2
      } else if (is.na(cat_levels) == FALSE && is.character(cat_levels) && nchar(cat_levels) > 0){
        if (log) {
          cat(bold('[Var is Categorical] (Patel marked) Var: ', var,
                   ' Unique Vals [#', length(unique_val) ,']: ',
                   toStringVector(unique_val), sep=''), '\n') 
        }
        ret <- 2
      }
    }
  }
  
  if (ret == 0 & log == TRUE){
    cat(bold('[Var is Continues] Var: ', var ,
             ' Unique Vals [#', length(unique_val) ,']: ',
             toStringVector(unique_val), sep=''), '... \n')
  }
  
  cache_binary_or_categorical_vars[[cache_key]] = ret
  
  return(ret)
}

round_df <- function(x, digits) {
  numeric_columns <- sapply(x, mode) == 'numeric'
  x[numeric_columns] <-  round(x[numeric_columns], digits)
  x
}

get_searies_years <- function(searies_number) {
  ret <- NULL
  if (searies_number == 1){
    ret <- '1999-2000'
  } else if(searies_number == 2){
    ret <- '2001-2002'
  } else if (searies_number == 3){
    ret <- '2003-2004'
  } else if (searies_number == 4){
    ret <- '2005-2006'
  }
  
  return(ret)
}

toStringVector <- function(v) {
  slice_size = 20
  
  if (length(v) < slice_size){ 
    return (toString(v))
  } else {
    return (paste(toString(v[1:slice_size]), "..."))
  }
}

boxcox_lambda <- hash()

boxcox_trans <- function(df, var_name){
  
  lambda <- boxCox(df[[var_name]]~1, family="yjPower", plotit = FALSE)
  lam_df <- data.frame(lambda$x, lambda$y)
  lambda <- lam_df[with(lam_df, order(-lam_df$lambda.y)),][1,1]
  
  print(paste("BoxCox Lambda for ", var_name , ": ", lambda))
  
  boxcox_lambda[[var_name]] <- lambda
  
  out <- yjPower(df[[var_name]], lambda, jacobian.adjusted=TRUE)
  
  out <- data.frame(out)#, ncol=1)
  colnames(out) <- var_name
  return(out)
}


boxcox_trans_return_lambda <- function(df, var_name){
  
  lambda <- boxCox(df[[var_name]]~1, family="yjPower", plotit = FALSE)
  lam_df <- data.frame(lambda$x, lambda$y)
  lambda <- lam_df[with(lam_df, order(-lam_df$lambda.y)),][1,1]
  
  # print(paste("BoxCox Lambda for ", var_name , ": ", lambda))
  
  boxcox_lambda[[var_name]] <- lambda
  
  out <- yjPower(df[[var_name]], lambda, jacobian.adjusted=TRUE)
  
  out <- data.frame(out)#, ncol=1)
  colnames(out) <- var_name
  return(list("out"=out, "lambda"=lambda))
}

#boxcox_trans(MainTable_subset, 'age')
#boxcox_lambda


logit_trans <- function(x){
  
  return (log(x / (1-x)))
}
