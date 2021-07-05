close all;
clear all;

basename0='I:\seg-augmentation\dataset3\';

for n = 1:5
    basename = [basename0 int2str(n)];
    subdir=dir(basename);
    for s = 1 : length(subdir)
        
        if(isequal(subdir(s).name,'.')||...
            isequal(subdir(s).name,'..')||...
            ~subdir( s ).isdir)               % 如果不是目录则跳过
            continue;
        end

        basename = [basename0 int2str(n) '\' subdir(s).name '\'];

        % inputfile=[basename 'image\'];
        inputfile2=[basename 'mask\'];
        % outputfile=[basename2 'image\'];
        % outputfile2=[basename2 'mask\'];
        outputfile3=[basename 'contour\'];
        outputfile4=[basename 'dis_mask\'];
        outputfile5=[basename 'dis_fore\'];
        outputfile6=[basename 'dis_contour\'];
        outputfile7=[basename 'dis_signed01\'];
        outputfile8=[basename 'dis_signed11\'];

        % Files=dir([inputfile '*.png']);
        Files=dir([inputfile2 '*.png']);
        number=length(Files);
        get_size=imread([inputfile2 Files(1).name]);
        [h,w]=size(get_size);

        re = h;
        sigma=1.6;%影响半径
        x=1:1:re;
        y=1:1:re;
        [X,Y]=meshgrid(x,y); 
        se=[0 1 0
            1 1 1
            0 1 0];

        for i=1:number
            mask=imread([inputfile2 Files(i).name]);
            mask = imbinarize(mask,126/255).*255;

        %     G = fspecial('gaussian', [5 5], 2);
        %     G = rescale(G,0,1);
        %     contour = bwperim(mask_out,8);
            contour00 = bwperim(mask,4);
            contour00 = double(contour00);
        %     contour01 = imfilter(contour00,G,'same');
        %     contour = rescale(contour01,0,1);
            [y,x]=find(contour00~=0);
            Z=zeros(size(X));
            Zsum=Z;
            for j=1:length(y)      
                Z=Gauss2D(X,Y,sigma,x(j),y(j));
        %         Z(Z<0.0001)=0;
        %         Z=rescale(Z,0,1);
                Zsum=Heatsum(Zsum,Z);
        %         Zsum=Zsum+Z;
            end

            contour=rescale(Zsum,0,1);
            contour(contour<0.01)=0;

            dis = bwdist(mask,'euclidean'); 
            dis = rescale(dis);
%             dis_show = imagesc(dis);
%             saveas(dis_show,[outputfile4 Files(i).name(1:end-4) '.png']);

            f_dis = bwdist(255-mask,'euclidean'); 
            f_dis = rescale(f_dis);
        %     f_dis_show = imagesc(f_dis);
        %     saveas(f_dis_show,[outputfile5 Files(i).name(1:end-4) '.png']);

            c_dis = bwdist(contour00,'euclidean'); 
            c_dis = rescale(c_dis);
        %     c_dis_show = imagesc(c_dis);
        %     saveas(c_dis_show,[outputfile6 Files(i).name(1:end-4) '.png']);


        %     s_dis = dis - (c_dis - dis);
        %     s_dis = rescale(s_dis);
        %     s_dis_show = imagesc(s_dis);
        %     saveas(s_dis_show,[outputfile6 Files(i).name(1:end-4) '.jpg']);

        %     figure()
        %     surf(X,Y,s_dis)
        %     shading interp
            s_dis_0 = - (bwdist(contour00,'euclidean') - bwdist(mask,'euclidean'));
            s_dis = rescale(s_dis_0,0,0.5);
            s_dis01 = rescale(dis,0,0.5) + s_dis;
        %     s_dis_show01 = imagesc(s_dis01);
        %     saveas(s_dis_show01,[outputfile7 Files(i).name(1:end-4) '.png']);

            s_dis_1 = - (bwdist(contour00,'euclidean') - bwdist(mask,'euclidean'));
            s_dis = rescale(s_dis_1,-1,0);
            s_dis11 = dis + s_dis;
        %     s_dis_show11 = imagesc(s_dis11);
        %     saveas(s_dis_show11,[outputfile8 Files(i).name(1:end-4) '.png']);

        %     imwrite(img_out,[outputfile Files(i).name(1:end-4) '.png']);
            imwrite(mask,[inputfile2 Files(i).name(1:end-4) '.png']);
            imwrite(contour00*255.,[outputfile3 Files(i).name(1:end-4) '.png']);
            save([outputfile3 Files(i).name(1:end-4) '.mat'],'contour');
        %     imwrite(dis,[outputfile4 Files(i).name(1:end-4) '.png']);
        %     imwrite(f_dis,[outputfile5 Files(i).name(1:end-4) '.png']);
        %     imwrite(c_dis,[outputfile6 Files(i).name(1:end-4) '.png']);
        %     imwrite(s_dis_show01,[outputfile7 Files(i).name(1:end-4) '.png']);
        %     imwrite(s_dis_show11,[outputfile8 Files(i).name(1:end-4) '.png']);

            save([outputfile4 Files(i).name(1:end-4) '.mat'],'dis');
            save([outputfile5 Files(i).name(1:end-4) '.mat'],'f_dis');
        %     save([outputfile6 Files(i).name(1:end-4) '.mat'],'s_dis');
            save([outputfile6 Files(i).name(1:end-4) '.mat'],'c_dis');
            save([outputfile7 Files(i).name(1:end-4) '.mat'],'s_dis01');
            save([outputfile8 Files(i).name(1:end-4) '.mat'],'s_dis11');
        end
    end
end
function Z=Gauss2D(X,Y,sigma,a,b)
%XY是网格坐标,sigma是高斯分布的宽度,ab是中心点坐标
    Z=0.5*exp(-((X-a).^2+(Y-b).^2)./sigma.^2);
end

function B=Heatsum(A1,A2)
%两个点之间叠加
B=1-(1-A1).*(1-A2);
end