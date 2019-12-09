dat = load('2019-11-18_00:01_13_1stdataset_250epochs_224size.mat');

figure, 
hold on;
title("Instability in CNN");
xlabel("Epochs");
ylabel("Accuracy");
xlim([0 100]);
plot(dat.accuracy,'lineWidth',1.5);
plot(dat.val_accuracy,'lineWidth',1.5);
legend({'Train','Test'},'Location','northEast');
set(findall(gcf,'-property','FontSize'),'FontSize',14);
hold off;