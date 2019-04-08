clear all;

% Scale factor
m = 10000;

% Open file for post processing, fid is file pointer
fid = fopen('test_pos.txt','r');
n = fscanf(fid,'%i',1);

% Setting the xy ratio of the figure, 1:1
set(gca,'DataAspectRatio',[1 1 1])

% Set up a color bar next to the figure
hc=colorbar;

% Use specific color map
colormap(jet(10));

% This holds the figure instead of creating new ones
% Useful to create overlapping 
hold on;

% Plotting deformed shape
for j=1:n
   for i=1:4
      x(i) = fscanf(fid,'%g',1);
      y(i) = fscanf(fid,'%g',1);
      z(i) = 0;
      u(i) = fscanf(fid,'%g',1);
      v(i) = fscanf(fid,'%g',1); 
      w(i) = 0;
      x(i) = x(i) + m*u(i);
      y(i) = y(i) + m*v(i);
      z(i) = z(i) + m*w(i);
      sxx(i) = fscanf(fid,'%g',1);
      syy(i) = fscanf(fid,'%g',1);
      sxy(i) = fscanf(fid,'%g',1);      
   end

   xx = [x(1) x(2) x(3) x(4)]';
   yy = [y(1) y(2) y(3) y(4)]';
   zz = [z(1) z(2) z(3) z(4)]';
   %vv = [sxy(1) sxy(2) sxy(3) sxy(4)]';
   %vv = [syy(1) syy(2) syy(3) syy(4)]';
   vv = [sxx(1) sxx(2) sxx(3) sxx(4)]';

   patch(xx,yy,vv,'Marker','o','EdgeColor','g','LineWidth',3);
end

% Plotting undeformed shape
fid = fopen('test_pos.txt','r');
n = fscanf(fid,'%i',1);

for j=1:n
   for i=1:4
      x(i) = fscanf(fid,'%g',1);
      y(i) = fscanf(fid,'%g',1);
      z(i) = 0;
      u(i) = fscanf(fid,'%g',1);
      v(i) = fscanf(fid,'%g',1); 
      w(i) = 0;
      sxx(i) = fscanf(fid,'%g',1);
      syy(i) = fscanf(fid,'%g',1);
      sxy(i) = fscanf(fid,'%g',1); 
   end

   xx = [x(1) x(2) x(3) x(4)]';
   yy = [y(1) y(2) y(3) y(4)]';
   zz = [z(1) z(2) z(3) z(4)]';

   patch(xx,yy,zeros(size(xx)),'Marker','o','EdgeColor','black','LineWidth',3,'FaceColor','none');
end
hold off


   