%weighted histogram (e.g., survey weights)
function [newdata]=whist(data, weights)

      minw=min(weights);
      minw=minw(1);
      
      normalized_w=round(weights/minw); 
      
      % not optimized
      newdata=[];
      for ind=1:length(data)
          newdata=[newdata;data(ind)*ones(normalized_w(ind),1)];
      end
      
      
      
      