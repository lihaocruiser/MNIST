% probsAverage = mean(w1probs);
% probsVariance = var(w1probs);
% w1av = [probsAverage; probsVariance];

average = zeros(286, 1);
variances = zeros(286, 1);
for ii = 1:286
    average(ii) = mean(phone(ii,:));
    variances(ii) = var(phone(ii,:));
end
