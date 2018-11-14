% reconstruct the MP-RAGE
%datfile=importdata('meas_MID00563_FID06072_pulseq.dat');
%matfile=dat2mat(datfile);

addpath(genpath('.'));
%[kspace_data,image_data]= dat2mat();

%% Obtain the file
[Filename,Pathname] = uigetfile('*.dat','Pick the raw data file');


%% Read data using mapVBVD
% image_obj = mapVBVD(fullfile(Pathname,Filename));
twix_obj = mapVBVDVE(fullfile(Pathname,Filename));
image_obj = twix_obj{2}.image;
sizeData = image_obj.sqzSize; %kx, nch, ky, slices, partitions
dimsData = image_obj.sqzDims;
dimsallData = image_obj.dataDims;

%% Confirmed that this is the noise char
% image_obj1 = twix_obj{1}.image;
% sizeData1 = image_obj1.sqzSize; %kx, nch, ky, slices, partitions
% dimsData1 = image_obj1.sqzDims;
% dimsallData = image_obj1.dataDims;


%% Directly perform inverse FT

kspace = squeeze(image_obj(:,:,:,:));
kspace_store = zeros(size(kspace));
Im_ch = zeros(size(kspace));
%Im_ch = zeros([500 20 500 500]);

%% Store complex images for each channel
%for p =1: size(kspace,4)
    parfor nch =1:size(kspace,2)
        %temp = squeeze(kspace(:,nch,:,p));
        %kspace_store(:,nch,:,p) = fftshift(temp);
        
        Im_ch(:,nch,:,:) = fftshift(fftn(kspace(:,nch,:,:)));
        %Im_ch(:,nch,:,:) = fftshift(fftn(((kspace(:,nch,:,:))),[500 1 500 500]));
        %figure(101); imagesc(abs(fftshift(fft2(temp)))); drawnow;pause(0.1);

    end
%end
Im_ch = permute(Im_ch, [1 3 4 2]);

%% enlarge the matrix size and do FFT










%% Save the reconstructed image
% figure();
% % pltimage=abs(Im_ch(:,120,:,8));
% % imshow(squeeze(pltimage));
% 
% hold on;
% for i=1:200
%    pltimage=abs(Im_ch(:,:,i,8));
%    imshow(squeeze(pltimage));
%     
%    
%    
% end

%%

figure();
pltimage=abs(Im_ch(:,:,100,8));
imagesc((pltimage));



%%
save('3DMPRAGErecon.mat','Im_ch','-v7.3');% lower version does not support saving files larger than 2GB

% %% Alternate size_data
% 
% sizeData(5)=1;
% 
% 
% 
% 
% %% Concatenate slices and partitions together and has a N x N acquisition set
% kspace_data = zeros([sizeData(1) sizeData(3) sizeData(4)*sizeData(5) sizeData(2)]);
% % kspace_data = permute(kspace_data,[1 3  4  5  2]); %kx, ky, slices, partitions, nch
% image_data = zeros(size(kspace_data));
% sl = 0;
% partitions=1;
% 
% %% Determine k-space shift to the center
% 
% temp = squeeze(image_obj(:,1,:,round(sizeData(4)/2),round(sizeData(5)/2)));
% [~,idx] = max(abs(temp(:)));
% [idx,idy] = ind2sub(size(temp),idx);
% 
% %%
% 
% tic;
% for nch=1:size(kspace_data,4)
% 
%        for nslices=1:size(kspace_data,3)
%            
%        sl = sl +1;
%            
%         temp = squeeze(image_obj(:,nch,:,sl,partitions));
%         temp = circshift(temp, [round(size(temp,1)/2) - idx,round(size(temp,2)/2) - idy ]);
%         trunc_part =round(0.5 .*(size(temp,1) - size(temp,2)));
%         temp = squeeze(temp(trunc_part+1:end-trunc_part,:));
%         
%         
%         kspace_data(:,:,nslices,nch) = temp;
%         image_data(:,:,nslices,nch) = fftshift(fft2(temp));
%         
%         if(sl == sizeData(4))
%             sl=0;
%             if(partitions == sizeData(5))
%                 partitions =0;
%             end
%             partitions = partitions+1;
%         end
%         
% %         imagesc(abs(fftshift(fft2(temp))));drawnow;
% %      imagesc(abs(fftshift(fft2(temp))));drawnow;
% %         pause;
% %         
%         
%         
%        end
% 
% end
% toc;
% 
% image_data = squeeze(image_data);
% kspace_data = squeeze(kspace_data);
% 
% 
