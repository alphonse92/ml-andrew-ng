function p = predict(Theta1, Theta2, X)
    [ a1,a2,a3,z2,z3,predictions] = predict_no_indices(Theta1, Theta2, X);
    [dummy, p] = max(predictions, [], 2);
end
