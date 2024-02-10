%% Slicing 2D image out of 3D image 
% Define 3D data

%Load the .mat image matrix (takes quite some time)
tic;
 load('3DMPRAGErecon.mat');%consider revising this to load('3DMPRAGErecon.mat');
toc;
%comment for saving time

%% Isolate real part and imagenary part to two matrices with same size, combine all channels - NOT required

dim=size(data);
dim2=dim;
dim2(4)=1;
allch1=zeros(dim2);
allch2=zeros(dim2);

for i=1:dim(4) % iterate through channels
    allch1=allch1+real(data(:,:,:,i)); %all the real 
    allch2=allch2+imag(data(:,:,:,i)); %all the imagenary
    a=i;
    disp(a);
end

allch=complex(allch1,allch2);

%% Crop the image to eliminate the oversampling voxels

res=256;
cutnumb=round((dim(1)-res)/2);
imagecropped_with_channel=data(cutnumb+1:(cutnumb+res)-1,:,:,:);%check for typos
imagecropped=allch(cutnumb+1:(cutnumb+res)-1,:,:);


%% Locate the center of the image - for initializing the vector if required

centerpoint=round(size(imagecropped)/2);
centerpoint=centerpoint(1:3);


%% Having a visualization of slicing

realpart=real(imagecropped(:,:,:));
imagpart=imag(imagecropped(:,:,:));
Slicer3Dvisual(abs(complex(realpart,imagpart)),centerpoint);


%% Make slice on each complex

chl=6; %choose a arbitrary channel
realpart=real(imagecropped_with_channel(:,:,:,chl));
imagpart=imag(imagecropped_with_channel(:,:,:,chl));


centerpoint=round(size(imagecropped_with_channel)/2); % Indicating the initial point of the vector, can be arbitrary
vec = [0 0 1];                          % Indicating the direction of the vector. Defined the normal vector of the plane we choose
%[realslice, sliceInd,subX,subY,subZ] = extractSlice(realpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
%[imagslice, sliceInd,subX,subY,subZ] = extractSlice(imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
% slice real part and imagenary part seperately, and combine them

[realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
comb_slice=complex(realslice,imagslice);
AB=abs(comb_slice);
AB(isnan(AB))=0; % zero filling the possible nan

% filename=[num2str(i),'Axial_middle_pic.png'];
% imwrite(AB,filename);
figure();
imshow(AB)




%% Make some slices


realpart=1*real(imagecropped(:,:,:));
imagpart=1*imag(imagecropped(:,:,:));

centerpoint=round(size(imagecropped)/2); % Indicating the initial point of the vector, can be arbitrary
vec = [0 0 1];                          % Indicating the direction of the vector. Defined the normal vector of the plane we choose
[realslice, sliceInd,subX,subY,subZ] = extractSlice(realpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
[imagslice, sliceInd,subX,subY,subZ] = extractSlice(imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
% slice real part and imagenary part seperately, and combine them

comb_slice=complex(realslice,imagslice);
AB=abs(comb_slice);
AB(isnan(AB))=0; % zero filling the possible nan

% filename=[num2str(i),'Axial_middle_pic.png'];
% imwrite(AB,filename);
figure();
imshow(AB)

%% given number of slides you want to have, generate them for you
chl=7;  % choose the channel
m=1; % define number of slices you want here

realpart=real(imagecropped_with_channel(:,:,:,chl));
imagpart=imag(imagecropped_with_channel(:,:,:,chl));
dim=size(realpart);
x=rand(m,1);
y=rand(m,1);
z=rand(m,1);

x=round(dim(1)*x);
y=round(dim(2)*y);
z=round(dim(3)*z);

alpha=rand(m,1);
gamma=rand(m,1);

alpha=cos((alpha-0.5)*2*pi);
beta=sin((alpha-0.5)*2*pi);
gamma=cos((gamma)*pi);

k=1;
for i=1:m
    centerpoint=[x(i) y(i) z(i)];
    vec=[alpha(i) beta(i) gamma(i)];
    [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
   comb_slice=complex(realslice,imagslice);
   comb_slice(isnan(comb_slice))=0;
   k_space=fftshift(fft2(comb_slice));
   AB=abs(comb_slice);
   AB(isnan(AB))=0;
   k=k+1;
%    filename=strcat('image',num2str(i),'.mat');
%    save(filename,'AB','-v7.3')
    figure();
    imagesc(AB);
    pause(0.1);
    
    %caxis([0 0.5]);
    figure(101);
    imagesc(abs(k_space));pause(0.5);
    
    
    
end

% show the corresponding k-space. and see if it make sence
    


figure(102);
imagesc(abs((ifft2((k_space)))));pause(0.5);

%% Axial slices

chl=7; %choose a channel

realpart=real(imagecropped_with_channel(:,:,:,chl));
imagpart=imag(imagecropped_with_channel(:,:,:,chl));

m=dim(1);

x=1:1:m;
y=ones(1,m)*round(dim(2)/2);
z=ones(1,m)*round(dim(3)/2);

alpha=ones(1,m);
beta=zeros(1,m);
gamma=zeros(1,m);

k=1;
%for i=1:m
figure();

%for i=120:140
for i=125

    centerpoint=[x(i) y(i) z(i)];
    vec=[alpha(i) beta(i) gamma(i)];
    [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
    comb_slice=complex(realslice,imagslice);
    comb_slice(isnan(comb_slice))=0;
    %k_space=fftshift(fft2(comb_slice));
    k_space=(fft2(comb_slice));
    AB=abs(comb_slice);
    AB(isnan(AB))=0;
    k=k+1;
%    filename=strcat('set1image',num2str(i),'.mat');
%    save(filename,'AB','-v7.3')
%    filename=strcat('set1_kspace_image',num2str(i),'.mat');
%    save(filename,'k_space','-v7.3')
%    figure();
    imagesc(AB);
    pause(0.1);
    
    %caxis([0 0.5]);
    %figure(101);
    %imagesc(abs(k_space));pause(0.1);
end

%% show image with phase, without phase and with random phase
% %with phase
% figure();
% imagesc(abs(comb_slice));
% 

figure();
imagesc(angle(comb_slice));
%with random phase
% pha=ones(size(AB));
k_from_image=fft2(AB);
k_from_image2=fftshift(k_from_image); %k-space from image
s=rand(3,3);
phasemap=(imresize(s,[257,257]).*2.*pi)-pi;
k_space_radphase=fftshift(k_from_image.*exp(-1i.*phasemap)); %random phase
image_rand_phase=(fft2(k_space_radphase));
figure();
imagesc(abs(image_rand_phase));
figure();
imagesc(angle(image_rand_phase));


%%
% 
% figure();
% for i =1:9
%     s=rand(2,2);
% %     phasemap=sin(imresize(s,[256,256]).*2.*pi)*pi;
%     phasemap=(imresize(s,[256,256]).*2.*pi)-pi;
%     subplot(3,3,i);
%     imagesc(phasemap);
%     axis off;
% % figure();
% % imagesc(abs(ranpha));
% 
% 
% end
%% Saggital

chl=8; %choose a channel

realpart=real(imagecropped_with_channel(:,:,:,chl));
imagpart=imag(imagecropped_with_channel(:,:,:,chl));

m=dim(2);

x=ones(1,m)*round(dim(1)/2);
y=1:1:m;
z=ones(1,m)*round(dim(3)/2);

alpha=zeros(1,m);
beta=ones(1,m);
gamma=zeros(1,m);

k=1;
%for i=1:m
for i=100:120

    centerpoint=[x(i) y(i) z(i)];
    vec=[alpha(i) beta(i) gamma(i)];
    [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
    comb_slice=complex(realslice,imagslice);
    comb_slice(isnan(comb_slice))=0;
    %k_space=fftshift(fft2(comb_slice));
    k_space=(fft2(comb_slice));
    AB=abs(comb_slice);
    AB(isnan(AB))=0;
    k=k+1;
%    filename=strcat('set2image',num2str(i),'.mat');
%    save(filename,'AB','-v7.3')
%    filename=strcat('set2_kspace_image',num2str(i),'.mat');
%    save(filename,'k_space','-v7.3')
    figure(101);
    imagesc(AB);
    pause(0.05);
    
    %caxis([0 0.5]);
    %figure(101);
    %imagesc(abs(k_space));pause(0.1);
end

%% Coronal

chl=8; %choose a channel

realpart=real(imagecropped_with_channel(:,:,:,chl));
imagpart=imag(imagecropped_with_channel(:,:,:,chl));

m=dim(3);

x=ones(1,m)*round(dim(1)/2);
y=ones(1,m)*round(dim(2)/2);
z=1:1:m;

alpha=zeros(1,m);
beta=zeros(1,m);
gamma=ones(1,m);

k=1;
%for i=1:m
for i=101

    centerpoint=[x(i) y(i) z(i)];
    vec=[alpha(i) beta(i) gamma(i)];
    [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
    comb_slice=complex(realslice,imagslice);
    comb_slice(isnan(comb_slice))=0;
    %k_space=fftshift(fft2(comb_slice));
    k_space=(fft2(comb_slice));
    AB=abs(comb_slice);
    AB(isnan(AB))=0;
    k=k+1;
%    filename=strcat('set3image',num2str(i),'.mat');
%    save(filename,'AB','-v7.3')
%    filename=strcat('set3_kspace_image',num2str(i),'.mat');
%    save(filename,'k_space','-v7.3')
    figure(101);
    imagesc(AB);
    pause(0.05);
    
    %caxis([0 0.5]);
    %figure(101);
    %imagesc(abs(k_space));pause(0.1);
end

%% Generate all the possible slides. Don't run if not necessary 
% i=0;
% for chl=1:dim(4)
%     realpart=real(imagecropped_with_channel(:,:,:,chl));
%     imagpart=imag(imagecropped_with_channel(:,:,:,chl));
%     centerpoint=round(size(imagecropped)/2);
%    for initial1 = 1:dim(1)
%        for initial2=1:dim(2)
%            for initial3=1:dim(3)
%                for z=-1:0.01:1
%                    for x=0:0.01:2*pi
%                        vec=[sin(x) cos(x) z];
%                        centerpoint=[initial1 initial2 initial3];
% 
%                        [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
%                        comb_slice=complex(realslice,imagslice);
%                        AB=abs(comb_slice);
%                        AB(isnan(AB))=0;
%                        i=i+1;
%                        filename=strcat('image',num2str(i),'.mat');
%                        save(filename,'AB','-v7.3')
%                    end
%                end
%            end
%        end
%    end
% end








