clc;
clear all;

folder_info_c1=dir('*_c1.tif');
folder_info_c2=dir('*_c2.tif');

for l=1:numel(folder_info_c1)%since c1 and c2 are always paired, we can only use c1 to scan file
    %opening the cropped image
    %read data
    filename_open=erase(folder_info_c1(l).name,"_c1.tif");%change file name to change file
    image_c1=imread(filename_open+"_c1.tif"); %red channel (cy5)
    image_c2=imread(filename_open+"_c2.tif"); %yelow channel (ATTO488)

    %setting up for data saving (creating a new folder)
    filename_hdf5=sprintf(filename_open+".hdf5");%filename for hdf5 saving 
    folder_name_saving=filename_open+"_data";
    folder_name_saving_path="\"+filename_open+"_data\";
    mkdir (folder_name_saving);


    %removing objects connected to border. Then
    %saving the result
    gray_image_c1= image_c1(:,:,1);
    binarize_filt_c1=imbinarize((imclearborder(image_c1(:,:,1))),0.0018);
    gray_image_noborder_object_c1=imclearborder(gray_image_c1);
    h5create(filename_hdf5,'/gray_image_c1',size(gray_image_c1));
    h5write(filename_hdf5,'/gray_image_c1',gray_image_c1);
    h5create(filename_hdf5,'/gray_image_noborder_object_c1',size(gray_image_noborder_object_c1));
    h5write(filename_hdf5,'/gray_image_noborder_object_c1',gray_image_noborder_object_c1);
    imwrite(gray_image_c1, fullfile(folder_name_saving, 'gray_image_c1.png'));
    imwrite(gray_image_noborder_object_c1, fullfile(folder_name_saving, 'gray_image_noborder_object_c1.png'));
    
    gray_image_c2= image_c2(:,:,1);
    binarize_filt_c2=imclearborder(~(imbinarize(imgaussfilt(image_c2(:,:,1),4))));
    gray_image_noborder_object_c2=imclearborder(gray_image_c2);
    h5create(filename_hdf5,'/gray_image_c2',size(gray_image_c2));
    h5write(filename_hdf5,'/gray_image_c2',gray_image_c2);
    h5create(filename_hdf5,'/gray_image_noborder_object_c2',size(gray_image_noborder_object_c2));
    h5write(filename_hdf5,'/gray_image_noborder_object_c2',gray_image_noborder_object_c2);
    imwrite(gray_image_c2, fullfile(folder_name_saving, 'gray_image_c2.png'));
    imwrite(gray_image_noborder_object_c2, fullfile(folder_name_saving, 'gray_image_noborder_object_c2.png'));
    
    %detecting circle uing c1 data
    %for data before blunt end, sensitivity 0.92 and edge threshold 0.075 work
    %best
    %for data after blunt end, sensitivity 0.95 and edge threshold 0.025 work
    %best
    [centers_c1,radii_c1] = imfindcircles(binarize_filt_c1,[16 40],"ObjectPolarity","bright", "Sensitivity",0.91,  "EdgeThreshold", 0.12); 
    h5create(filename_hdf5,'/centers_c1',size(centers_c1));
    h5write(filename_hdf5,'/centers_c1',centers_c1);
    h5create(filename_hdf5,'/radii_c1',size(radii_c1));
    h5write(filename_hdf5,'/radii_c1',radii_c1);
    writematrix(centers_c1,  filename_open+"_center.csv");
    writematrix(radii_c1,  filename_open+"_radii.csv");

    % %showing the detected circles
    fig1=figure
    GUVs_c1 = insertText(gray_image_c1,centers_c1-20,string(linspace(1,length(centers_c1),length(centers_c1))),FontSize=25,TextColor="white", BoxOpacity=0);
    imshow(GUVs_c1);
    h_c1 = viscircles(centers_c1,1.1*radii_c1, 'LineWidth',0.5);
    exportgraphics(fig1,fullfile(folder_name_saving, 'detected_GUVs_c1.png'),'Resolution',1000);
    close(fig1)

    fig1a=figure
    imshow(gray_image_c1,[min(gray_image_c1(:)) max(gray_image_c1(:))]);
    h_c1 = viscircles(centers_c1,1.1*radii_c1, 'LineWidth',0.5);
    exportgraphics(fig1a,fullfile(folder_name_saving, 'detected_GUVs_c1a.png'),'Resolution',1000);
    close(fig1a)

    fig2=figure
    GUVs_c2 = insertText(gray_image_c2,centers_c1-20,string(linspace(1,length(centers_c1),length(centers_c1))),FontSize=25,TextColor="white", BoxOpacity=0);
    imshow(GUVs_c2);
    h_c2 = viscircles(centers_c1,1.1*radii_c1, 'LineWidth',0.5);
    exportgraphics(fig2,fullfile(folder_name_saving, 'detected_GUVs_c2.png'),'Resolution',1000);
    close(fig2)

    fig2a=figure
    GUVs_c2 = insertText(gray_image_c2,centers_c1-20,string(linspace(1,length(centers_c1),length(centers_c1))),FontSize=25,TextColor="white", BoxOpacity=0);
    imshow(gray_image_c2,[min(gray_image_c2(:)) max(gray_image_c2(:))]);
    h_c2 = viscircles(centers_c1,1.1*radii_c1, 'LineWidth',0.5);
    exportgraphics(fig2a,fullfile(folder_name_saving, 'detected_GUVs_c2a.png'),'Resolution',1000);
    close(fig2a)

    %adding 10% increase to radius for cropping purpose of c1 and c2
    radii_crop=1.25*radii_c1;
    % imshow(gray_image_noborder_object)
    % h = viscircles(centers,radii_crop);

    %cropping c1 and c2 followed by masking for each GUVs, rotation average
    for i=1:length(radii_crop)
        filename_c1=sprintf('/data/image_cropped_c1_%d', i);
        filename_c2=sprintf('/data/image_cropped_c2_%d', i);

        X_A(i,1)=centers_c1(i,1)-radii_crop(i);
        Y_A(i,1)=centers_c1(i,2)-radii_crop(i);

        image_cropped_c1 = imcrop(gray_image_c1,[X_A(i,1) Y_A(i,1)  2*radii_crop(i) 2*radii_crop(i)]);
        image_cropped_center_c1=size(image_cropped_c1)/2;

        image_cropped_c2 = imcrop(gray_image_c2,[X_A(i,1) Y_A(i,1)  2*radii_crop(i) 2*radii_crop(i)]);
        image_cropped_center_c2=size(image_cropped_c2)/2;

        fig1=figure;
        cropping_c1 = insertText(image_cropped_c1,image_cropped_center_c1-20,num2str(i),FontSize=25,TextColor="white", BoxOpacity=0);
        imshow(cropping_c1);
        a = viscircles(image_cropped_center_c1,1.1*radii_c1(i), 'LineWidth',0.5);
        exportgraphics(fig1,fullfile(folder_name_saving, sprintf('cropped_image_c1_%d.png',i)),'Resolution',300);
        close(fig1)

        fig2=figure;
        cropping_c2 = insertText(image_cropped_c2,image_cropped_center_c1-20,num2str(i),FontSize=25,TextColor="white", BoxOpacity=0);
        imshow(cropping_c2);
        b = viscircles(image_cropped_center_c2,1.1*radii_c1(i), 'LineWidth',0.5);
        exportgraphics(fig2,fullfile(folder_name_saving, sprintf('cropped_image_c2_%d.png',i)),'Resolution',300);
        close(fig2)

        %ploting and saving the cropping results of c1 and c2 using imshowpair for comparison
        fig3=figure;
        imshowpair(image_cropped_c1,image_cropped_c2, "montage");
        exportgraphics(fig3,fullfile(folder_name_saving, sprintf('cropped_image_c1_c2_%d.png',i)),'Resolution',300);
        close(fig3)

        %imwrite(image_cropped, fullfile(folder_name_saving, sprintf('cropped_image%d.png',i)));
        h5create(filename_hdf5,filename_c1,size(image_cropped_c1));
        h5write(filename_hdf5,filename_c1,image_cropped_c1);

        h5create(filename_hdf5,filename_c2,size(image_cropped_c2));
        h5write(filename_hdf5,filename_c2,image_cropped_c2);

        %finding the centerpoint of cropped image from c1 (also can be applied
        %to c2)
        image_cropped_centerpoint_c1=round(size(image_cropped_c1)/2);
        x_A(i,1)=image_cropped_centerpoint_c1(1,1)-radii_crop(i);
        y_A(i,1)=image_cropped_centerpoint_c1(1,2)-radii_crop(i);

        %masking cropped image of c and c2 using masked created using c2
        masked_image_c1=uint32(image_cropped_c1);
        masked_image_c2=uint32(image_cropped_c2);

        %ploting and saving the masking results of c1 and c2 using imshowpair for comparison
        fig4=figure;
        imshowpair(masked_image_c1,masked_image_c2, "montage");
        exportgraphics(fig4,fullfile(folder_name_saving, sprintf('masked_image_c1_c2_%d.png',i)),'Resolution',300);
        close(fig4)

        %saving the data of masked image
        filename_masked_c1=sprintf('/data/image_masked_c1_%d', i);
        filename_masked_c2=sprintf('/data/image_masked_c2_%d', i);

        h5create(filename_hdf5,filename_masked_c1,size(masked_image_c1));
        h5write(filename_hdf5,filename_masked_c1,masked_image_c1);

        h5create(filename_hdf5,filename_masked_c2,size(masked_image_c2));
        h5write(filename_hdf5,filename_masked_c2,masked_image_c2);

        %rotational averaging
        image_rotated_c1=masked_image_c1;
        image_rotated_c2=masked_image_c2;

        for angle=1:1:360

            image_rotated_c1=image_rotated_c1+imrotate(masked_image_c1,angle,'crop');
            image_rotated_c2=image_rotated_c2+imrotate(masked_image_c2,angle,'crop');

        end

        threshold=0.5;%interior threshold for ring detection after normalizing the intensity
        image_rotated_average_c1=round(image_rotated_c1/max(image_rotated_c1(:))-threshold+0.5);
        image_rotated_average_c2=round(image_rotated_c2/max(image_rotated_c2(:))-threshold+0.5);


        %ploting and saving the masking results of c1 and c2 using imshowpair for comparison
        fig5=figure;
        imshowpair(image_rotated_c1,image_rotated_c2, "montage");
        exportgraphics(fig5,fullfile(folder_name_saving, sprintf('rotated_averaged_image_c1_c2_%d.png',i)),'Resolution',300);
        close(fig5)

        %ploting and saving the masking results of c1 and c2 using imshowpair for comparison
        fig6=figure;
        imshowpair(image_rotated_average_c1,image_rotated_average_c2, "montage");
        exportgraphics(fig6,fullfile(folder_name_saving, sprintf('rotated_averaged_thresholded_image_c1_c2_%d.png',i)),'Resolution',300);
        close(fig6)

        %saving the data of rotation-averaged image
        filename_rotated_c1=sprintf('/data/image_rotated_averaged_c1_%d', i);
        filename_rotated_c2=sprintf('/data/image_rotated_averaged_c2_%d', i);

        h5create(filename_hdf5,filename_rotated_c1,size(image_rotated_c1));
        h5write(filename_hdf5,filename_rotated_c1,image_rotated_c1);

        h5create(filename_hdf5,filename_rotated_c2,size(image_rotated_c2));
        h5write(filename_hdf5,filename_rotated_c2,image_rotated_c2);

        %saving the data of rotation-averaged thresholded image
        filename_rotated_averaged_c1=sprintf('/data/image_rotated_averaged_thresholded_c1_%d', i);
        filename_rotated_averaged_c2=sprintf('/data/image_rotated_averaged_thresholded_c2_%d', i);

        h5create(filename_hdf5,filename_rotated_averaged_c1,size(image_rotated_average_c1));
        h5write(filename_hdf5,filename_rotated_averaged_c1,image_rotated_average_c1);

        h5create(filename_hdf5,filename_rotated_averaged_c2,size(image_rotated_average_c2));
        h5write(filename_hdf5,filename_rotated_averaged_c2,image_rotated_average_c2);

        %plot along x diameter rotated averaged crossection
        image_size_rotated_c1=size(image_rotated_c1);
        image_size_rotated_c2=size(image_rotated_c2);

        y_plot_rotated_c1=image_rotated_c1(round(image_size_rotated_c1(1,1)/2),:);
        x_plot_rotated_c1=linspace(1, image_size_rotated_c1(1,2),image_size_rotated_c1(1,2));

        y_plot_rotated_c2=image_rotated_c2(round(image_size_rotated_c2(1,1)/2),:);
        x_plot_rotated_c2=linspace(1, image_size_rotated_c2(1,2),image_size_rotated_c2(1,2));

        %plot along x diameter rotated averaged smoothed crossection
        image_size_rotated_smoothed_c1=size(image_rotated_c1);
        image_size_rotated_smoothed_c2=size(image_rotated_c2);

        y_plot_rotated_smoothed_c1=smoothdata(image_rotated_c1(round(image_size_rotated_c1(1,1)/2),:),"gaussian",3);
        x_plot_rotated_smoothed_c1=linspace(1, image_size_rotated_c1(1,2),image_size_rotated_c1(1,2));

        y_plot_rotated_smoothed_c2=smoothdata(image_rotated_c2(round(image_size_rotated_c2(1,1)/2),:),"gaussian",3);
        x_plot_rotated_smoothed_c2=linspace(1, image_size_rotated_c2(1,2),image_size_rotated_c2(1,2));

        %saving smoothdata crossection
        filename_rotated_averaged_smoothed_c1=sprintf('/data/y_plot_rotated_smoothed_c1_%d', i);
        filename_rotated_averaged_smoothed_c2=sprintf('/data/y_plot_rotated_smoothed_c2_%d', i);

        h5create(filename_hdf5,filename_rotated_averaged_smoothed_c1,size(y_plot_rotated_smoothed_c1));
        h5write(filename_hdf5,filename_rotated_averaged_smoothed_c1,y_plot_rotated_smoothed_c1);

        h5create(filename_hdf5,filename_rotated_averaged_smoothed_c2,size(y_plot_rotated_smoothed_c2));
        h5write(filename_hdf5,filename_rotated_averaged_smoothed_c2,y_plot_rotated_smoothed_c2);

        %threshold the smoothed data
        threshold=0.92;%interior threshold for ring detection after normalizing the intensity
        y_plot_rotated_smoothed_thresholded_c1=round(y_plot_rotated_smoothed_c1/max(y_plot_rotated_smoothed_c1)-threshold+0.5);
        y_plot_rotated_smoothed_thresholded_c2=round(y_plot_rotated_smoothed_c2/max(y_plot_rotated_smoothed_c2)-threshold+0.5);

        %saving smoothed and thresholded data
        filename_rotated_averaged_smoothed_thresholded_c1=sprintf('/data/y_plot_rotated_smoothed_thresholded_c1_%d', i);
        filename_rotated_averaged_smoothed_thresholded_c2=sprintf('/data/y_plot_rotated_smoothed_thresholded_c2_%d', i);

        h5create(filename_hdf5,filename_rotated_averaged_smoothed_thresholded_c1,size(y_plot_rotated_smoothed_thresholded_c1));
        h5write(filename_hdf5,filename_rotated_averaged_smoothed_thresholded_c1,y_plot_rotated_smoothed_thresholded_c1);

        h5create(filename_hdf5,filename_rotated_averaged_smoothed_thresholded_c2,size(y_plot_rotated_smoothed_thresholded_c2));
        h5write(filename_hdf5,filename_rotated_averaged_smoothed_thresholded_c2,y_plot_rotated_smoothed_thresholded_c2);

        %plot along x diameter rotated averaged and thresholded crossection
        image_size_c1=size(image_rotated_average_c1);
        image_size_c2=size(image_rotated_average_c2);

        y_plot_c1=image_rotated_average_c1(round(image_size_c1(1,1)/2),:);
        x_plot_c1=linspace(1,image_size_c1(1,2),image_size_c1(1,2));

        y_plot_c2=image_rotated_average_c2(round(image_size_c2(1,1)/2),:);
        x_plot_c2=linspace(1,image_size_c2(1,2),image_size_c2(1,2));


        ipt_c1=findchangepts(double(y_plot_rotated_smoothed_thresholded_c1),MaxNumChanges=4);
        ipt_c2=findchangepts(double(y_plot_rotated_smoothed_thresholded_c2),MaxNumChanges=4);

       try
        fig7=figure
        subplot(3,2,1)
        plot(x_plot_rotated_c1,y_plot_rotated_c1)
        title('Red channel crossection')

        subplot(3,2,2)
        plot(x_plot_rotated_c2,y_plot_rotated_c2)
        title('Green channel crossection')

        subplot(3,2,3)
        plot(x_plot_rotated_smoothed_c1,y_plot_rotated_smoothed_c1)
        title('Red channel crossection smoothed')

        subplot(3,2,4)
        plot(x_plot_rotated_smoothed_c2,y_plot_rotated_smoothed_c2)
        title('Green channel crossection smoothed')

        subplot(3,2,5)
        plot(x_plot_rotated_smoothed_c1,y_plot_rotated_smoothed_thresholded_c1)
        hold on
        xline(ipt_c1,'r')
        hold off
        title('Red channel crossection thresholded')

        subplot(3,2,6)
        plot(x_plot_rotated_smoothed_c2,y_plot_rotated_smoothed_thresholded_c2)
        hold on
        xline(ipt_c2,'r')
        hold off
        title('Green channel crossection thresholded')

        exportgraphics(fig7,fullfile(folder_name_saving, sprintf('crossection_plot_%d.png',i)),'Resolution',300);
        close(fig7)

        %ring detection
        A=size(ipt_c1);
        B=size(ipt_c2);

        if A(1,2)==4
            ring_c1(i,1)=true;
        else
            ring_c1(i,1)=false;
        end

        if B(1,2)==4
            ring_c2(i,1)=true;
        else
            ring_c2(i,1)=false;
        end
        ring_c1_and_c2(i,1)= and(ring_c1(i,1),ring_c2(i,1));
       catch
        ring_c1(i,1)=0;
        ring_c2(i,1)=0;
        ring_c1_and_c2(i,1)=0;
        exportgraphics(fig7,fullfile(folder_name_saving, sprintf('crossection_plot_%d.png',i)),'Resolution',300);
        close(fig7)
        fprintf('changepoints in iteration %d are weird, skipped.\n', i);

       end

       clear filename_c1 filename_c2 image_cropped_c1 image_cropped_center_c1 image_cropped_c2 image_cropped_center_c2 cropping_c1 a cropping_c2 b fig3 image_cropped_centerpoint_c1 fig3 h_im_c1 circ_c1 mask_c1 masked_image_c1 masked_image_c2 fig4 filename_masked_c1 filename_masked_c2 image_rotated_c1 image_rotated_c2 image_rotated_average_c1 image_rotated_average_c2 fig5 fig6 filename_rotated_c1 filename_rotated_c2 filename_rotated_averaged_c1 filename_rotated_averaged_c2 image_size_rotated_c1 image_size_rotated_c2 image_size_c1 image_size_c2 ipt_c1 ipt_c2 fig7
    end 

    %saving the data of ring information
    filename_ring_c1=sprintf('/ring/ring_c1');
    filename_ring_c2=sprintf('/ring/ring_c2');
    filename_ring_c1_c2=sprintf('/ring/ring_c1_c2');

    h5create(filename_hdf5,filename_ring_c1,size(ring_c1));
    h5write(filename_hdf5,filename_ring_c1,uint8(ring_c1));
 

    h5create(filename_hdf5,filename_ring_c1_c2,size(ring_c1_and_c2));
    h5write(filename_hdf5,filename_ring_c1_c2,uint8(ring_c1_and_c2));

    h5create(filename_hdf5,filename_ring_c2,size(ring_c2));
    h5write(filename_hdf5,filename_ring_c2,uint8(ring_c2));

    clear filename_open image_c1 image_c2 filename_hdf5 folder_name_saving folder_name_saving_path gray_image_c1 gray_image_noborder_object_c1 gray_image_c2 gray_image_noborder_object_c2 centers_c1 radii_c1 fig1 fig2 radii_crop filename_ring_c1 filename_ring_c2 filename_ring_c1_c2 A B
end
