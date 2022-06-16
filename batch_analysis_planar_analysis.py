import pandas as pd
import os
import bioformats
import javabridge
import batch_analysis_functions as b
import re
import planar_analysis as p

# read in data from spreadsheet
sheet_path="Flow Assay Log.xlsx"
sheet_path=sheet_path.replace('\\','/')
master_info_df=pd.read_excel(sheet_path,sheet_name='Kinetic')
dynamic_master=pd.DataFrame()


# -------info to copy from into df to results of analysis
# Generate a dictionary that allows us to specify the options we want to pass into our function to evaluate mean fluorescence and metrics
default_options = dict(
                 cycle_vm=False, # telling javabridge to stay on
                 image_channel=1, # selecting image channel 1 for first analysis
                 meta_stage_loop=True, # doesn't look for stage loop
                 t_sample=1, # sampling every nth timepoint
                 training_dir="Z:\Matt_Sorrells\2022 Ilastik Training Data\Y-Z Orthogonals\Platelets\Platelets.ilp",
                 meta_number=None,
                 zero_index=1
                             )
# options we want to modify for processing p-selectin data 
ps_options=dict(t_lag_level=5E4,
                training_dir="Z:/Matt_Sorrells/2022 Ilastik Training Data/Y-Z Orthogonals/P-selectin/P-selectin.ilp"
                )
# determine what metadata to pull in from spreadsheet
info=[]
for col in master_info_df.columns:
    if col not in default_options.keys():
        info.append(col)

current_path=os.getcwd()
javabridge.start_vm(class_path=bioformats.JARS)
# A for loop to loop through the cells in the excel sheet where I store all of the relevant information for the assays
for index, row in master_info_df.iterrows():
    kinetic_path = row['File Path']
    print('---------Row: ', str(index+2),' ',kinetic_path)
    # Only do analysis for cells where I specify a file location since some cells don't have this
    if not pd.isna(kinetic_path):
        # since a relative path is passed in from the excel sheet we need to add the absolute path 
        # to that 
        # from Cell Sense they dump the channels of all photos into the same folder but distinguish them by file type
        kinetic_path=kinetic_path.replace('\\','/')
        path= kinetic_path
        path=path+'.vsi'
        # We have an option in the excel sheet to pull the data from the old master csvs for a given assay if we don't need to analyze it
        if row['Analyze Arch Images']=='N':
           print('\t','Read in From Previous Analysis')
           old_dynamic_master=pd.read_csv('time_series_data.csv')
           dynamic_df=old_dynamic_master[old_dynamic_master['Assay ID']==row['Assay ID']]
        else:
            options=default_options
            # Optional things that are specified for the functions
            if not pd.isna(row['zero_index']):
                options['zero_index']=row['zero_index']
            # if a background is specified we feed that into the fluorescence processing 
            if not pd.isna(row['meta_number']):
                options['meta_number']=int(row['meta_number'])
            # Process DIO C6 Data
            print('DIO')
            dio=p.yz_timepoint(path,**options)
            for col in dio.columns:
                if col !='time (s)':
                    dio=dio.rename(columns={str(col):'DIO'+str(col)})
            print('P-selectin')
            digits=re.findall(r'\d+', path)
            last_digit=int(digits[-1])
            current_number=re.findall('(\d+)(?!.*\d)',path)[0]
            current_number=int(current_number)
            for key,val in ps_options.items():
                options[key]=val
            path=re.sub('(\d+)(?!.*\d)',str(current_number+1),path)
            ps=p.yz_timepoint(path,**options)
            ps=ps.drop(columns=['time (s)'])
            ps.columns=['PS '+ str(col) for col in ps.columns]
            dynamic_df=pd.concat([dio,ps],axis=1)
            
            # met_df=fl_m
            # store info about assay conditions into analsysi
            b.store_info(dynamic_df,row,info)
        # Store all of this data in dataframes so we can easily plot all of it together
        dynamic_master=dynamic_master.append(dynamic_df,ignore_index=True)
    dynamic_master.to_csv('time_series_data.csv',index=False)
        # metric_platelet_df=metric_platelet_df.append(m,ignore_index=True)        
javabridge.kill_vm()
