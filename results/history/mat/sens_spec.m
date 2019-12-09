dat = load('2019-11-27_23:20_26_best_model_30epochs_300size.mat')

figure, 
hold on;
title("Sensitivity & Specificity over a 30 epoch trial");
xlabel("Epochs");
ylabel("Accuracy");
xlim([0 30]);
% plot(dat.accuracy,'lineWidth',1.5);
% plot(dat.val_accuracy,'lineWidth',1.5);
plot(dat.val_sensitivity,'lineWidth',1.5);
plot(dat.val_specificity,'lineWidth',1.5);
legend({'Validation sensitivity','Validation specificity'},'Location','southEast');
set(findall(gcf,'-property','FontSize'),'FontSize',14);
hold off;