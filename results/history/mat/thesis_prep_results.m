
dat1 = load('/Users/isaac/r/otis/results/history/mat/2019-11-19_23:37_22_newmodel_alldata_split_100epochs_300size.mat')
dat2 = load('/Users/isaac/r/otis/results/history/mat/2019-11-19_11:49_21_newmodel_alldata_100epochs_300size.mat')


figure,
hold on;
title("Otis model training and validation accuracy")
xlabel("Epoch");
ylabel("Accuracy");
xlim([0 100]);
ylim([0.5 1]);
plot(dat1.accuracy, 'linewidth',2);
plot(dat1.val_accuracy, 'linewidth',2);
plot(dat2.accuracy, 'linewidth',2);
plot(dat2.val_accuracy, 'linewidth',2);
legend({'0.2 split training','0.2 split validation', 'Small sample training', 'Small sample validation'},'Location','southeast')
hold off;