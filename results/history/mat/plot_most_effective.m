dat = load('2019-10-31_05:29_5_nosplit_100epochs_224size.mat');


figure,
t = title({'{\fontsize{18} Otis Accuracy}'},'FontWeight','Normal');
ylabel('Accuracy');
xlabel('Epochs');
ylim([0 1])
hold on;
plot((1:100),dat.accuracy, (1:100), dat.val_accuracy,'lineWidth',1.5);
lgd = legend({'Train','Test'},'Location','southEast');
lgd.FontSize = 14;
hold off;
s = annotation('textbox',[0.15,.15,.25,.15],'String',{'size: 224pixels','Train imgs: 31916', 'Test imgs: 624'});
s.FontSize = 14;
savefig('most_effective.fig')