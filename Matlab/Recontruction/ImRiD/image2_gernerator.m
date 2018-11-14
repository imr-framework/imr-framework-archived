%% Slicing 2D image out of 3D image 
% This code slice 2D image first, then it will do nufft to get radial
% k-space. The radial k-space is used to reconstruct image2 using
% nufft_adj. After that, the new cartisian k-space will be generated.


%addpath(genpath('.'));
% system = mr.opts('MaxGrad',32,'GradUnit','mT/m',...
%     'MaxSlew',130,'SlewUnit','T/m/s');


%Load the .mat image matrix (takes quite some time)
% tic;
%  %load('/Users/pdhe/Documents/??/Columbia/Research/irt/Code/MP-RAGE Recon/8-5-2018/ACR3DMPRAGErecon-8-3.mat');%consider revising this to load('3DMPRAGErecon.mat');
%  %load('/Users/pdhe/Documents/??/Columbia/Research/irt/Code/MP-RAGE Recon/7-19-2018/3DMPRAGErecon.mat');
%  %load('/Users/pdhe/Documents/??/Columbia/Research/irt/Code/MP-RAGE Recon/ktraj.mat');
%  %load('3DMPRAGErecon.mat');
%  load('ACR3DMPRAGErecon-8-3.mat');
%  load('ktraj.mat');
%  
% toc;

%comment for saving time

%% Crop the image to eliminate the oversampling voxels

dim=size(Im_ch);

res=256;
cutnumb=round((dim(1)-res)/2);
imagecropped_with_channel=Im_ch(cutnumb+1:(cutnumb+res)-1,:,:,:);%check for typos

%% Make slice on each complex
Q=1; %global location of one data point 

num_per_set=3;
sets=3;

x_train_i2=zeros(256,256,num_per_set);
x_train_k2=zeros(256,256,num_per_set);
y_train=zeros(256,256,num_per_set);


ktraj2=ktraj';
[numpoints,numspokes]=size(ktraj2);
ktraj2=ktraj2(:);

kx=real(ktraj2);
ky=imag(ktraj2);
rows = 256;
cols = 256;%size(comb_slice);
N=[rows,cols];
ktraj_use= [kx ky];
om=2*pi*ktraj_use;

J = [5 5];	% interpolation neighborhood
K = N*2;	% two-times oversampling

st = nufft_init(om, N, J, K, N/2, 'minmax:kb');

for k=1:sets
    i=1;
    Q=1;
    while i<=num_per_set
        disp((i));
        chl=randi(dim(4));
        centerpoint=[randi(dim(1)) randi(dim(2)) randi(dim(3))];
        z=randi([-1000 1000])/1000;
        x=randi([0 1000])/1000*2*pi;
        realpart=real(imagecropped_with_channel(:,:,:,chl));
        imagpart=imag(imagecropped_with_channel(:,:,:,chl));
        vec=[sin(x) cos(x) z];
        [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
        
        comb_slice=complex(realslice,imagslice);
        comb_slice=(imresize(comb_slice,[256,256]));
        if sum(sum(abs(comb_slice)>0.1))<10000
            continue
        end
        %%add random noise to the region that does not have signal
        AB=abs(comb_slice);
        noisemat=[];
        parfor l=1:length(AB(:))
             if AB(l)>0&&AB(l)<0.025
                 noisemat=[noisemat,comb_slice(l)];
             end
        end
        parfor m=1:length((comb_slice(:)))
            if isnan(comb_slice(m))
                comb_slice(m)=noisemat(randi(length(noisemat)));
                %comb_slice(m)=0+0i;
            end
        end

        kspace=fft2(comb_slice);
        %st = nufft_init(om, N, J, K, N/2, 'minmax:kb');
        kundersample=nufft(comb_slice,st);
        kundersample_use = reshape(kundersample,[numpoints,numspokes]);
    %     fname = ['ImRiD_coronal', num2str(i),'_channel',num2str(chl),'_','kspace'];
    %     save(fname, 'kundersample_use');
    %     fname = ['ImRiD_coronal', num2str(i),'_channel',num2str(chl),'_','GTimage'];
    %     save(fname, 'comb_slice');    
         % image 2 from undersample k-space
        image2=nufft_adj(kundersample,st);
        kspace2=fft2(image2);
    %     fname = ['Image2_ImRiD_sagital', num2str(i),'_channel',num2str(chl),'_','image'];
    %     save(fname, 'image2');
    %     fname = ['kspace2_ImRiD_sagital', num2str(i),'_channel',num2str(chl),'_','kspace'];
    %     save(fname, 'kspace2');  
        % check the image;

    %     figure();
    %     imagesc(abs(image2));
    %     colormap(gray);
    %     title('image 2 from radial kspace');
    %     axis off;   
    %     figure();
    %     imagesc(abs(comb_slice));
    %     colormap(gray);
    %     title('original image with absolute value');
    %     axis off;
    %     figure();
    %     
    %     imagesc(abs(kspace2));
    %     title('kspace from image2');
    %     axis off;
    %     figure();
    %     imagesc(abs(kspace));
    %     title('kspace ');
    %     axis off;

        x_train_i2(:,:,Q)=image2;
        x_train_k2(:,:,Q)=kspace2;
        y_train(:,:,Q)=comb_slice;

        Q=Q+1;
        i=i+1;
    end
    fname = ['ImRiD_ACR_image2_set', num2str(k)];
    save(fname, 'x_train_i2');
    fname = ['ImRiD_ACR_kspace2_set', num2str(k)];
    save(fname, 'x_train_k2');
    fname = ['ImRiD_ACR_compleximage_set', num2str(k)];
    save(fname, 'y_train');
    
    disp('image set generated');

end

        
        
        
        
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

%%

% figure();
% zz=abs(y_train(100,100,1:2000));
% zz=reshape(zz,[1,2000]);
% plot(zz);
% 
% %%
% 
% x_train_i2=x_train_i2(:,:,1:251);
% x_train_k2=x_train_k2(:,:,1:251);
% y_train=y_train(:,:,1:251);


