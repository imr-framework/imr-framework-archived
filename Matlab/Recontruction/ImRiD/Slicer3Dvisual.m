function Slicer3Dvisual(realpart,centerpoint)

% A small function that uses extratSlice to visualize the slicing of the
% image in 3D

for i = 0:30
%pt = [202 127 112];
vec = [0 30-i i];
[slice, sliceInd,subX,subY,subZ] = extractSlice(realpart,centerpoint(1),centerpoint(2),centerpoint(3),vec(1),vec(2),vec(3),128);
surf(subX,subY,subZ,slice,'FaceColor','interp','EdgeColor','none');
%axis([1 130 1 130 1 40]);
axis([1 404 1 254 1 224]);axis off;
drawnow;colormap(gray);


end