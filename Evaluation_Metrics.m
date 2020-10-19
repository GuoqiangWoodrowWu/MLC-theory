function [ HammingLoss, SubsetAccuracy, Ranking_Loss ] = Evaluation_Metrics( pre_Label, pre_F, Y )
% Evaluate the model for many metrics

    cd('./measures');
    HammingLoss = Hamming_loss(pre_Label, Y);
    SubsetAccuracy = Subset_accuracy(pre_Label, Y);
    Ranking_Loss = Ranking_loss(pre_F, Y);
    cd('../');
end
