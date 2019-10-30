% NOT NEEDED FOUND KERAS EQUIV
% ---
% Isaac Carr (i.carr@unsw.edu.au)
% Developed for MMAN4020, 19T3
% Health Group 4
% ---
% This file aims to create new grayscale images from the train and test 
% set 

%% Set up

path        = '../../data/chest_xray/';
ntr_path    = strcat(path, 'train/NORMAL/');      % normal train   
ptr_path    = strcat(path, 'train/PNEUMONIA/');   % pneumonia train   
nte_path    = strcat(path, 'test/NORMAL/');       % normal test
pte_path    = strcat(path, 'test/PNEUMONIA/');    % pneumonia test   

%% Make train normal images gray
files = dir(strcat(ntr_path, '*.jpeg')); 
len = length(files);

for i=1:len
    files(i)
    info        = imfinfo(strcat(ntr_path, files(i).name))
    img         = imread(strcat(ntr_path, files(i).name)); 
    imshow(img);
    pause;
    grey_img    = rgb2gray(img); 
    imshow(grey_img);
    pause; 
end 