function [result] = costEquation(predictions,y)
    m = length(y); % number of training examples
    left = -y .* log(predictions); # -(mx1) .* log(mx1)
    right = (1 .- y) .* log(1 .- predictions); # still mx1 , sum(mx1) = 1x1
    result = left.-right;
end