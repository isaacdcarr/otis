dr
dat1= load('2019-10-30_10:07_3_newdata_100epochs_224size.mat');
dat2 = load('2019-10-31_00:23_4_data_split0.2_100epochs_224size.mat') ;
dat3 = load('2019-10-31_05:29_5_nosplit_100epochs_224size.mat');
dat4  = load('2019-10-31_11:28_6_0.1split_100epochs_224size.mat');
dat5 = load('2019-11-02_04:04_8_5000training_100epochs_600size.mat');
dat6 = load('2019-11-01_12:20_8_5000training_100epochs_400size.mat');
dat7 = load('2019-11-02_19:40_9_all_250epochs_500size.mat');
dat8 = load('2019-11-17_03:46_10_final_250epochs_224size.mat');
dat9 = load('2019-11-17_03:51_11_extralayer512_250epochs_300size.mat');
dat10 = load('2019-11-17_11:44_12_fin_250epochs_300size.mat');
dat11 = load('2019-11-18_00:01_13_1stdataset_250epochs_224size.mat');
dat12 = load('2019-11-18_08:37_16_alldatasingleout_singledataset_250epochs_300size.mat');
dat13 = load('2019-11-18_08:41_16_alldatasingleout_singledataset_250epochs_300size.mat');
dat14 = load('2019-11-18_12:39_17_alldatasingleout_split_100epochs_300size.mat');
dat15 = load('2019-11-18_22:39_18_singledata_quickresults_30epochs_250size.mat');
dat16 = load('2019-11-19_11:30_20_newmodel_100epochs_400size.mat');

figure,
sgtitle("16/37 iterations");

subplot(4,4,1);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 1');
plot(dat1.accuracy);
plot(dat1.val_accuracy);
hold off;

subplot(4,4,2);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 2');
plot(dat2.accuracy);
plot(dat2.val_accuracy);
hold off;

subplot(4,4,3);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 3');
plot(dat3.accuracy);
plot(dat3.val_accuracy);
hold off;

subplot(4,4,4);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 4');
plot(dat4.accuracy);
plot(dat4.val_accuracy);
hold off;

subplot(4,4,5);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 5');
plot(dat5.accuracy);
plot(dat5.val_accuracy);
hold off;

subplot(4,4,6);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 6');
plot(dat6.accuracy);
plot(dat6.val_accuracy);
hold off;

subplot(4,4,7);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 7');
plot(dat7.accuracy);
plot(dat7.val_accuracy);
hold off;

subplot(4,4,8);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 8');
plot(dat8.accuracy);
plot(dat8.val_accuracy);
hold off;

subplot(4,4,9);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 9');
plot(dat9.accuracy);
plot(dat9.val_accuracy);
hold off;

subplot(4,4,10);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 10');
plot(dat10.accuracy);
plot(dat10.val_accuracy);
hold off;

subplot(4,4,11);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 11');
plot(dat11.accuracy);
plot(dat11.val_accuracy);
hold off;

subplot(4,4,12);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 12');
plot(dat12.accuracy);
plot(dat12.val_accuracy);
hold off;

subplot(4,4,13);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 13');
plot(dat13.accuracy);
plot(dat13.val_accuracy);
hold off;

subplot(4,4,14);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 14');
plot(dat14.accuracy);
plot(dat14.val_accuracy);
hold off;

subplot(4,4,15);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 15');
plot(dat15.accuracy);
plot(dat15.val_accuracy);
hold off;

subplot(4,4,16);
hold on;
xlabel("Epoch");
ylabel("Accuracy");
title('Iteration 16');
plot(dat16.accuracy);
plot(dat16.val_accuracy);
hold off;
