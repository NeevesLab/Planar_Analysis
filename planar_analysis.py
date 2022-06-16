import numpy as np
import vsi_metadata as v
import bioformats
import javabridge
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import thresholding as t


# combines in height profile into loops of z and t
def yz_timepoint(path,meta_number=None, cycle_vm=True,
                 image_channel=1,t_sample=1,
                 show=True, manual=True,thresh_val=300,zero_index=0,
                 binarization='ilastik',training_dir='Platelets.ilp'
                 ):
    
    # start javabridge
    if cycle_vm:
        javabridge.start_vm(class_path=bioformats.JARS)
    # read in metadata using bioformats and make ararys for t ans z slices
    metadata=v.extract_metadata(path,cycle_vm=False,
                                meta_number=meta_number)
    z_slices=np.arange(0,metadata['size_Z'])
    z_slices_scaled=z_slices*float(metadata['relative step width'])
    t_slices=np.arange(0,metadata['size_T'])
    t_slices=t_slices[::t_sample]
    t_slices_scaled=(t_slices-zero_index)*float(metadata['cycle time'])
    
    # find the indices at 20,40,60,80,and 100% timepoints
    t_slice_percentage=np.array([0,.2,.4,.6,.8,1])
    t_slice_num=float(len(t_slices))*t_slice_percentage
    t_slice_loop_indices=np.round(t_slice_num,0).astype(int)
    max_index=np.max(t_slice_loop_indices)
    # similarly, get 5 linearly spaced points from .2 to .8 to select along the y-z plane 
    ortho_selection=np.linspace(.3,.7,5)
    # iterate through time slices, stop before the last one
    master_planes_df=pd.DataFrame()
    for i in t_slice_loop_indices:
        print('\t-fraction',i/max_index)
        if i==max_index:
            i=i-1
        # declare surface_area array outside of z-stack loop
        area=np.zeros(len(z_slices))
        for j in range(len(z_slices)):
            # read image
            img=bioformats.load_image(path=path,z=z_slices[j],t=t_slices[i],
                                      rescale=False)
            # if the image consists of multiple channels take just one channel 
            if len(np.shape(img))>2:
                img=img[:,:,image_channel]
            
            # make image container to store the z-stack in memory
            if j==0:
                img_dims=np.shape(img)
                img_container=np.zeros([img_dims[0],img_dims[1],len(z_slices)])
                if i==0:
                    # generate the indices of the orthogonals we wanna call
                    ortho_indices=np.round(img_dims[1]*ortho_selection,0).astype(int)
            # store z-stack into memory
            img_container[:,:,j]=img
        # collect orthogonals frm
        ortho_stacks=np.zeros([img_dims[0],len(z_slices),len(ortho_selection)])
        for k in range(len(ortho_selection)):
            ortho_stacks[:,:,k]=img_container[:,ortho_indices[k],:]
        del img_container
        # edgefinder 
        ortho_thresh=np.zeros(np.shape(ortho_stacks))
        print(np.shape(ortho_thresh))
        for k in range(len(ortho_selection)):
            print(np.shape(ortho_stacks[:,:,k]))
            ortho_thresh[:,:,k]=t.binarize(ortho_stacks[:,:,k],binarization=binarization,
                                               training_dir=training_dir)
        z_scale=float(metadata['relative step width'])
        xy_scale=float(metadata['scale'])
        # part that will show the image comparing the edge detector 
        if show:
            fig, axes = plt.subplots(1, len(ortho_selection)*2,figsize=(20,10))
            fig_count=0
            size_rows=img_dims[0]*xy_scale
            size_z_stacks=float(metadata['size_Z'])*z_scale
            aspect=size_z_stacks/size_rows
            print("Mean {:.2f}".format(np.mean(ortho_stacks[:,:,0])),
                  "Min {:.2f}".format(np.min(ortho_stacks[:,:,0])),
                  "Max {:.2f}".format(np.max(ortho_stacks[:,:,0]))
                  )
            print("Mean {:.2f}".format(np.mean(ortho_thresh[:,:,0])))
            for k in range(len(ortho_selection)):
                fig
                plt.sca(axes[fig_count])
                plt.imshow(ortho_stacks[:,:,k],aspect=aspect)
                plt.title('OG')
                
                plt.sca(axes[fig_count+1])
                plt.imshow(ortho_thresh[:,:,k],aspect=aspect)
                plt.title('Threshold')
                fig_count+=2
            plt.tight_layout()
            plt.show()
        del ortho_stacks
        planes_df=pd.DataFrame()
        # compute the max height, lenght along y-z frame, and store with other relevenant metadata
        for k in range(len(ortho_selection)):
            thresh_slice=ortho_thresh[:,:,k]
            
            height_scaled=calculate_height(thresh_slice,xy_scale,z_scale)
            yz_length=np.shape(thresh_slice)[0]*xy_scale
            planes_df['height (um)']=height_scaled
            planes_df['y-z length (um)']=np.linspace(0,yz_length,len(height_scaled))
            planes_df['y-z selection']=[ortho_selection[k]]*len(planes_df)
            planes_df['time (s)']=[t_slices_scaled[i]]*len(planes_df)
            planes_df['t selection']=[i]*len(planes_df)
            planes_df['time (m)']=planes_df['time (s)']/60
            master_planes_df=master_planes_df.append(planes_df,ignore_index=True)
    if cycle_vm:
        javabridge.kill_vm()
    return master_planes_df

# function for computing height along y-z axis of thresholded images
# splits the image into 100 32 pixels-long segments and computes the height 
# by fitting a rectangle to the thresholded image, then returns the height of
# that rectangle
def calculate_height(img,xy_scale,z_scale,step=32,show=False):
    img=img.astype('uint8')
    rect_img=np.copy(img)
    shape=np.shape(img)
    length=shape[0]
    length_intervals=np.arange(0,length,step)
    height=np.zeros(len(length_intervals))
    height_count=0
    for l in length_intervals:
        sub_img=img[l:l+step,:]
        contours,_ = cv2.findContours(sub_img, 1, 2)
        if len(contours)==0:
            height[height_count]=0
        else:
            big_area=0
            for c in contours:
                area=cv2.contourArea(c)*xy_scale*z_scale
                if area>big_area:
                    big_area=area
                    big_contour=c
            if big_area<1:
                height[height_count]=0
            else:
                x,y,w,h = cv2.boundingRect(big_contour)
                if show:
                    rect_img = cv2.rectangle(rect_img,(x,y),(x+w,y+h),(0,255,0),2)
                
                
                height[height_count]=w
        height_count+=1
    if show:
        plt.imshow(rect_img)
        plt.show()
    return height

