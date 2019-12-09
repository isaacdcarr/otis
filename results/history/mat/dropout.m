norm_dat = load('2019-11-19_23:37_22_newmodel_alldata_split_100epochs_300size.mat');
drop_dat = load('2019-11-21_09:54_24_newnew_alldata_split_adddropout_100epochs_300size.mat');

figure, 
hold on;
title("Effect of drop out on accuracy");
xlabel("Epochs");
ylabel("Accuracy");
plot(norm_dat.accuracy,'lineWidth',1.5);
plot(norm_dat.val_accuracy,'lineWidth',1.5);
plot(drop_dat.accuracy,'lineWidth',1.5);
plot(drop_dat.val_accuracy,'lineWidth',1.5);
legend({'Train normal','Test normal', 'Train with drop out','Test with drop out'},'Location','southEast');
set(findall(gcf,'-property','FontSize'),'FontSize',14)
hold off;