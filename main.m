
srcFiles = dir('Data/Pedestrian/*.ppm')  % the folder in which ur images exists
numFiles = length(srcFiles)
numRows = ceil(sqrt(numFiles));
for k = 1 : numFiles
    thisFileName = fullfile(srcFiles(k).folder, srcFiles(k).name);
    thisImage = imread(thisFileName);
    thatImage = imresize(thisImage, [300,300]);
    imwrite(thatImage, thisFileName);
end