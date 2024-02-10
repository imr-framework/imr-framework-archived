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

%% Sequence events
% % Some sequence parameters are defined using standard MATLAB variables
% fov=256e-3;   %One parameter to test
% Nx=256; Ny=256;
% sliceThickness=5e-3;
% TR = 20e-3;
% TE = 5e-3;%minimum 
% dx = fov/Nx;
% dy  =dx;
% 
% numslices=1;
% 
% %zstart=sliceThickness*(-numslices/2);
% %zend=sliceThickness*(numslices/2);
% 
% Np =256; %Cartesian phase encodes - play around with this number later for evals
% Ns = ceil(pi*Np); % Number of spokes required for 360 based on Cartesian Phase encodes
% % theta = linspace(0,360,radp.Ns); %This will be replaced with golden angle
% dtheta = 360/Ns;
% theta = 0:dtheta: (Ns-1)*dtheta; % For debugging
% % theta = get_goldenangle(Ns); % For better temporally resolved data acquisition - not meaningful for FID acq
% 
% % Np = length(theta); %equivalent to Ns
% ktraj = zeros(Ns, 256);
% scantime=TR*Ns*numslices;
% deltak=1/fov;
% kWidth = Nx*deltak;
% readoutTime = 6.4e-3;
% 
% 
% gx = mr.makeTrapezoid('x',system,'FlatArea',kWidth,'FlatTime',readoutTime);
% adc = mr.makeAdc(Nx,'Duration',gx.flatTime,'Delay',gx.riseTime);
% 
% 
% for np=1:Ns % generate the k-space trajectory
%     
%         %seq.addBlock(rf,gz);
%         %seq.addBlock(gzReph);
%         kWidth_projx = kWidth.*cosd(theta(np));
%         kWidth_projy = kWidth.*sind(theta(np));
% 
%         gx = mr.makeTrapezoid('x',system,'FlatArea',kWidth_projx,'FlatTime',readoutTime);
%         gy = mr.makeTrapezoid('y',system,'FlatArea',kWidth_projy,'FlatTime',readoutTime);
%         ktraj(np,:) = get_ktraj(gx,gy,adc,1);
%         disp(atan2d(gy.flatArea, gx.flatArea));
%         %seq.addBlock(delay1);
%         %seq.addBlock(gx,gy,adc);
%         %seq.addBlock(gzSpoil);
%         %seq.addBlock(delay2);
% 
% end

% %% Isolate real part and imagenary part to two matrices with same size, combine all channels - NOT required
% 
% data=Im_ch;
% dim=size(data);
% dim2=dim;
% dim2(4)=1;
% allch1=zeros(dim2);
% allch2=zeros(dim2);
% 
% for i=1:dim(4) % iterate through channels
%     allch1=allch1+real(data(:,:,:,i)); %all the real 
%     allch2=allch2+imag(data(:,:,:,i)); %all the imagenary
%     a=i;
%     disp(a);
% end
% 
% allch=complex(allch1,allch2);

%% Crop the image to eliminate the oversampling voxels

dim=size(Im_ch);

res=256;
cutnumb=round((dim(1)-res)/2);
imagecropped_with_channel=Im_ch(cutnumb+1:(cutnumb+res)-1,:,:,:);%check for typos

%% Make slice on each complex
Q=1; %global location of one data point 

x_trai_i2=zeros(256,256,2000);
x_train_k2=zeros(256,256,2000);
y_train=zeros(256,256,2000);

for chl=1:dim(4)


    
 %choose a arbitrary channel
realpart=real(imagecropped_with_channel(:,:,:,chl));
imagpart=imag(imagecropped_with_channel(:,:,:,chl));



centerpoint=round(size(imagecropped_with_channel)/2); % Indicating the initial point of the vector, can be arbitrary
startpoint=[centerpoint(1) centerpoint(2) centerpoint(3)];
vec = [0 0 1];
[realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,startpoint(1),startpoint(2),startpoint(3),vec(1),vec(2),vec(3),128);
comb_slice=complex(realslice,imagslice);
comb_slice=(imresize(comb_slice,[256,256]));
comb_slice(isnan(comb_slice))=0;

ktraj2=ktraj';
[numpoints,numspokes]=size(ktraj2);

ktraj2=ktraj2(:);
kx=real(ktraj2);
ky=imag(ktraj2);

[rows,cols] = size(comb_slice);
N=[rows,cols];
ktraj_use= [kx ky];
om=2*pi*ktraj_use;

J = [5 5];	% interpolation neighborhood
K = N*2;	% two-times oversampling

% generate the k space trajectory in the format of IRT
st = nufft_init(om, N, J, K, N/2, 'minmax:kb');



for i=120:130%192
    
    startpoint=[centerpoint(1) centerpoint(2) i];
    vec = [0 0 1]; % Indicating the direction of the vector. Defined the normal vector of the plane we choose
    [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,startpoint(1),startpoint(2),startpoint(3),vec(1),vec(2),vec(3),128);
    comb_slice=complex(realslice,imagslice);
    
    comb_slice=(imresize(comb_slice,[256,256]));
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
        end
    end
    
    kspace=fft2(comb_slice);
    %st = nufft_init(om, N, J, K, N/2, 'minmax:kb');
    kundersample=nufft(comb_slice,st);
    kundersample_use = reshape(kundersample,[numpoints,numspokes]);
%     fname = ['ImRiD_axial', num2str(i),'_channel',num2str(chl),'_','kspace'];
%     save(fname, 'kundersample_use');
%     fname = ['ImRiD_axial', num2str(i),'_channel',num2str(chl),'_','GTimage'];
%     save(fname, 'comb_slice');    
    % image 2 from undersample k-space
    image2=nufft_adj(kundersample,st);
    kspace2=fft2(image2);
%     fname = ['Image2_ImRiD_axial', num2str(i),'_channel',num2str(chl),'_','image'];
%     save(fname, 'image2');
%     fname = ['kspace2_ImRiD_axial', num2str(i),'_channel',num2str(chl),'_','kspace'];
%     save(fname, 'kspace2');  
    
    
    x_trai_i2(:,:,Q)=image2;
    x_train_k2(:,:,Q)=kspace2;
    y_train(:,:,Q)=comb_slice;
    
    Q=Q+1;
    
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
%     figure();
%     imagesc(abs(nufft_adj(kundersample,st)));
%     colormap(gray);
%     axis off;

%    pause(0.002)
    

end

%% saggital

%centerpoint=round(size(imagecropped_with_channel)/2); % Indicating the initial point of the vector, can be arbitrary
%startpoint=[centerpoint(1) centerpoint(2) centerpoint(3)];
vec = [0 1 0];
[realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,startpoint(1),startpoint(2),startpoint(3),vec(1),vec(2),vec(3),128);
comb_slice=complex(realslice,imagslice);
comb_slice=(imresize(comb_slice,[256,256]));
comb_slice(isnan(comb_slice))=0;
    
% ktraj2=ktraj';
% [numpoints,numspokes]=size(ktraj2);
% 
% ktraj2=ktraj2(:);
% kx=real(ktraj2);
% ky=imag(ktraj2);
% 
% [rows,cols] = size(comb_slice);
% N=[rows,cols];
% ktraj_use= [kx ky];
% om=2*pi*ktraj_use;
% 
% J = [5 5];	% interpolation neighborhood
% K = N*2;	% two-times oversampling
% 
% st = nufft_init(om, N, J, K, N/2, 'minmax:kb');

for i=102%1:256
    
    startpoint=[centerpoint(1) i centerpoint(3)];
    vec = [0 1 0]; % Indicating the direction of the vector. Defined the normal vector of the plane we choose
    [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,startpoint(1),startpoint(2),startpoint(3),vec(1),vec(2),vec(3),128);
    comb_slice=complex(realslice,imagslice);
    
    comb_slice=(imresize(comb_slice,[256,256]));
    
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
        end
    end
    kspace=fft2(comb_slice);
    %st = nufft_init(om, N, J, K, N/2, 'minmax:kb');
    kundersample=nufft(comb_slice,st);
    kundersample_use = reshape(kundersample,[numpoints,numspokes]);
%     fname = ['ImRiD_sagital', num2str(i),'_channel',num2str(chl),'_','kspace'];
%     save(fname, 'kundersample_use');
%     fname = ['ImRiD_sagital', num2str(i),'_channel',num2str(chl),'_','GTimage'];
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
    
    x_trai_i2(:,:,Q)=image2;
    x_train_k2(:,:,Q)=kspace2;
    y_train(:,:,Q)=comb_slice;
    
    Q=Q+1;

end


%% coronal

%startpoint=[centerpoint(1) centerpoint(2) centerpoint(3)];
vec = [1 0 0];
[realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,startpoint(1),startpoint(2),startpoint(3),vec(1),vec(2),vec(3),128);
comb_slice=complex(realslice,imagslice);
comb_slice=(imresize(comb_slice,[256,256]));
comb_slice(isnan(comb_slice))=0;
    
% ktraj2=ktraj';
% [numpoints,numspokes]=size(ktraj2);
% 
% ktraj2=ktraj2(:);
% kx=real(ktraj2);
% ky=imag(ktraj2);
% 
% [rows,cols] = size(comb_slice);
% N=[rows,cols];
% ktraj_use= [kx ky];
% om=2*pi*ktraj_use;
% 
% J = [5 5];	% interpolation neighborhood
% K = N*2;	% two-times oversampling
% 
% st = nufft_init(om, N, J, K, N/2, 'minmax:kb');

for i=125%1:255
    
    startpoint=[i centerpoint(2) centerpoint(3)];
    vec = [1 0 0]; % Indicating the direction of the vector. Defined the normal vector of the plane we choose
    [realslice, imagslice, ~, ~, ~, ~] = extractSlice_complex(realpart,imagpart,startpoint(1),startpoint(2),startpoint(3),vec(1),vec(2),vec(3),128);
    comb_slice=complex(realslice,imagslice);
    
    comb_slice=(imresize(comb_slice,[256,256]));
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
    
    x_trai_i2(:,:,Q)=image2;
    x_train_k2(:,:,Q)=kspace2;
    y_train(:,:,Q)=comb_slice;
    
    Q=Q+1;

end

end


%%

figure();
zz=abs(y_train(100,100,1:2000));
zz=reshape(zz,[1,2000]);
plot(zz);

%%

x_trai_i2=x_trai_i2(:,:,1:251);
x_train_k2=x_train_k2(:,:,1:251);
y_train=y_train(:,:,1:251);


