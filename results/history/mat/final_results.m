norm_dat = load('2019-11-19_11:49_21_newmodel_alldata_100epochs_300size.mat');
drop_dat = load('2019-11-21_09:54_24_newnew_alldata_split_adddropout_100epochs_300size.mat');

figure, 
hold on;
title("Final results");
xlabel("Epochs");
ylabel("Accuracy");
xlim([0 100]);
ylim([0.5 1.0]);
plot(norm_dat.accuracy,'lineWidth',1.5);
plot(norm_dat.val_accuracy,'lineWidth',1.5);
plot(drop_dat.accuracy,'lineWidth',1.5);
plot(drop_dat.val_accuracy,'lineWidth',1.5);
legend({'Train specific test-set','Test specific test-set', 'Train 20% split','Test 20% split'},'Location','southEast');
set(findall(gcf,'-property','FontSize'),'FontSize',14)
hold off;

final_val_small = sprintf('Final val small %.4f', norm_dat.val_accuracy(100))
final_val_split= sprintf('Final val split %.4f', drop_dat.val_accuracy(100))
mean_val_small = sprintf('Mean val small %.4f', mean(norm_dat.val_accuracy))
mean_val_split= sprintf('Mean val split %.4f', mean(drop_dat.val_accuracy))
sd_val_small = sprintf('Mean val small %.4f', std(norm_dat.val_accuracy))
sd_val_split= sprintf('Mean val split %.4f', std(drop_dat.val_accuracy))