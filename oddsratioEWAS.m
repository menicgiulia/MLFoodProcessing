function oddsratio=oddsratioEWAS(xold, xnew, beta, l)

    %boxcox
    transdatold = boxcox(l,xold);
    transdatnew = boxcox(l,xnew);
    
    %zscore
    meant=mean(transdatold);
    stdt=std(transdatold);
    oldvar=(transdatold-meant)/stdt;
    newvar=(transdatnew-meant)/stdt;

    oddsratio=exp(beta*(newvar-oldvar));