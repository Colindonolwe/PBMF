% function [m,n] = readnoise()

I = imread('lena.jpg');
Agr1 = rgb2gray(I);
% Agr1 = I;
[m, n] = size(Agr1);

J2 = impulsenoise(Agr1,0.2,1);
% J3 = impulsenoise(Agr1,0.3,1);
% J4 = imnoise(Agr1,'gaussian',0,400/255^2);
% J5 = imnoise(Agr1,'gaussian',0,900/255^2);
% J6 = imnoise(Agr1,'salt & pepper',0.2);
% J7 = imnoise(J2,'gaussian',0,400/255^2);
% J8 = impulsenoise(Agr1,0.5,1);
% J = imnoise(Agr1,'salt & pepper',0.2);
% J(m/2+1:m,:) = I(m/2+1:m,:);
% J = imnoise(Agr1,'gaussian',0,900/255^2);
% J = imnoise(J,'gaussian',0,400/255^2);
% J = awgn(reshape(Agr,[m*n,1]),30);
% J = reshape(J,[m,n]);
dtr = im2double(Agr1);
% A = im2double(J);
% figure;
imshow(J2)
% imwrite(A,'noisyImageSP20.jpg','jpg','Comment','My JPEG file')
