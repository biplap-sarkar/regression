function x_n = mapNonLinear(x,d)
% Inputs:
% x - a single column vector (N x 1)
% d - integer (>= 0)
% Outputs:
% x_n - (N x (d+1))

N = size(x,1);
if(d==0)
    x_n = ones(N,1);
else
    x_n = [ones(N,1) x];
    if(d>1)
        %x_n = [ones(N,1) x_n];
        for i=2:d
            x_n = [x_n x.^i];
        end
    end
end

end
