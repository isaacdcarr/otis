dat = load('2019-11-19_11:30_20_newmodel_100epochs_400size.mat');

figure, 
hold on;
title("Large variance");
xlabel("Epochs");
ylabel("Accuracy");
xlim([0 100]);
plot(dat.accuracy,'lineWidth',1.5);
plot(dat.val_accuracy,'lineWidth',1.5);
legend({'Train','Test'},'Location','northEast');
set(findall(gcf,'-property','FontSize'),'FontSize',14);
hold off;