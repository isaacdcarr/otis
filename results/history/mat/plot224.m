dat = load('2019-11-18_00:01_13_1stdataset_250epochs_224size.mat');
dat2 = load('2019-11-19_11:49_21_newmodel_alldata_100epochs_300size.mat');
dat3 = load('2019-11-19_11:30_20_newmodel_100epochs_400size.mat');
figure, 
hold on;
title("224x224 versus 300x300 versus 400x400 Image Input");
xlabel("Epochs");
ylabel("Accuracy");
xlim([0 60]);
ylim([0.2 1.0]);
plot(dat.accuracy,'lineWidth',1.5);
plot(dat.val_accuracy,'lineWidth',1.5);
plot(dat2.accuracy,'lineWidth',1.5);
plot(dat2.val_accuracy,'lineWidth',1.5);
plot(dat3.accuracy,'lineWidth',1.5);
plot(dat3.val_accuracy,'lineWidth',1.5);
legend({'224 Train','224 Test', '300 Train','300 Test','400 Train','400 Test'},'Location','southEast');
set(findall(gcf,'-property','FontSize'),'FontSize',14);
hold off;