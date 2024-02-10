function [realslice, imagslice, sliceInd, subX, subY, subZ] = extractSlice_complex(realpart,imagpart,centerX,centerY,centerZ,normX,normY,normZ,radius)

[realslice, sliceInd,subX,subY,subZ] = extractSlice(realpart,centerX,centerY,centerZ,normX,normY,normZ,radius);
[imagslice, ~,~,~,~] = extractSlice(imagpart,centerX,centerY,centerZ,normX,normY,normZ,radius);


end