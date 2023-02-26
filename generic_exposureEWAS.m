function exposureratio=generic_exposureEWAS(responsevar, xold, xnew, beta,l, lresponse)
    invboxcox= @(y,ll) (y.*ll + 1).^(1/ll);
    
    %boxcox
    transdatold = boxcox(l,xold);
    transdatnew = boxcox(l,xnew);
    
    %zscore
    meant=mean(transdatold);
    stdt=std(transdatold);
    oldvar=(transdatold-meant)/stdt;
    newvar=(transdatnew-meant)/stdt;
    
    % response transformation
    transresponsevar=boxcox(lresponse, responsevar);
    stdtresponsevar=std(transresponsevar(~isnan(transresponsevar)));

    estimatedresponsevar=invboxcox(transresponsevar+stdtresponsevar*beta*(newvar-oldvar), lresponse);
    exposureratio=estimatedresponsevar./responsevar;