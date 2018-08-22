import time
from datetime import timedelta
import numpy as np
from skimage import  morphology
import cv2
import csv
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import os
import ast

def Segmentation(WorkingDirectory, ListTrainingDataFile, ListImageName, modelname, noiseReduction, numberOfClasses, classesNamesList, ROI, ListAreaNames, fusionClassesY_N, maskY_N, imageY_N, InfoY_N, NFMaskY_N, BiggestBlobY_N, chosenArea, ReferencePicture):
    """ Main function for the segmentation.
    
    Args:
          (string): 
            Address of the working directory
            (ex: WorkingDirectory='/Users/Name/Desktop/folder'
            
        ListTrainingDataFile (List of strings):
            List of the addresses of all the training data files that the user wants to use to train the machine learning model. 
            (ex: ListTrainingDataFile=['/Users/Name/Desktop/folder/TrainingData/TrainingDataIndividualFiles/trainData_PaddyRice_image1.csv','/Users/Name/Desktop/folder/TrainingData/TrainingDataIndividualFiles/TrainingDataIndividualFiles/trainData_Background_image1.csv'])
        
        ListImageName (List of strings):
            List of the addresses of all the pictures that the user wants to process.  
            (ex: ListImageName=['/Users/Name/Desktop/image.png','/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png'])
        
        modelname (string): 
            Name of the chosen model 
            ('Support Vector Machine (Sklearn)','Random Forest Classifier (Sklearn)','Classification and Regression Tree (Sklearn)')
        
        noiseReduction (int): 
            Maximal size of the area which will be removed in the function 'noiseRemoval' 
            (ex: noiseReduction=100)
        
        numberOfClasses (int):
            Number of classes 
            (ex: numberOfClasses=2)
        
        classesNamesList (List of strings):
             List of the names of the classes 
             (ex: classesNamesList=['PaddyRice','Background'] )
        
        ROI (string):
            'Whole picture' 
            or a **string** of a List of the coordinates of each region of interest (ROI) in the same order as ListAreaNames.
            For each ROI, the first two numbers are the coordinates of the top left corner and the other two are the coordinates of the bottum right corner.
            (ex: ROI='[[0,0,50,50],[50,0,100,50],[0,50,50,100]]' )
        
        ListAreaNames (List of strings):
             List of the names of the areas in the same order as the list ROI 
             (ex: ListAreaNames=['P1','P2','P1xP2'] ) 
            
        fusionClassesY_N (string): 
            Two possible values 'Y' or 'N'. If fusionClassesY_N='Y', the user have chosen to fusion two or more classes.
        
        maskY_N (string): 
            Two possible values 'Y' or 'N'. If maskY_N='Y', the user have chosen to save the mask (binary image if there is only two classes and colored flat image if more than two classes)
        
        imageY_N (string): 
            Two possible values 'Y' or 'N'. If imageY_N='Y', the user have chosen to save the reconstructed image (only show the class of interest if there is only two classes and  image+colored filter if there is more than two classes)
        
        InfoY_N (string): 
            Two possible values 'Y' or 'N'. If imageY_N='Y', the user have chosen to save the information file containing for each plant : 'Area/Plant','Image Name','Surface','Coverage', 'Aspect Ratio','Extent','Solidity', 'Equivalent Diameter', 'Main axe', 'Secondary axe'
        
        NFMaskY_N (string): 
            Two possible values 'Y' or 'N'. If maskY_N='Y', the user have chosen to save the mask before any noise reduction and morphological filtering.  
        
        BiggestBlobY_N(string): 
            Two possible values 'Y' or 'N'. If BiggestBlobY_N='Y', the user have chosen to only keep the biggest blob of the mask for analysis.
        
        chosenArea (string): 
            Name of the class of interest (the one that will be mesured) 
            (ex: 'PaddyRice')
        
        ReferencePicture (string):
            Address of the picture used to define the ROI and that will be used to decide if the other pictures have the right size 
            (ex: ReferencePicture='/Users/Name/Desktop/image.png')
        
    Return:
        ListImageWrongSize (List of strings):
            List of the addresses of the pictures that were not processed beacause their size was ddifferent from the size of the reference picture
            (ex: ListImageName=['/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png']
            
        ListRunningTimes (List float): 
            List of the running times for each picture in sec. 
            
        ListTestDataTimes (List float): 
            List of the times to createthe test data array (read the picture) for each picture in sec. 
            
        ListApplyModelTimes (List float): 
            List of the times to apply the model to the picture array for each picture in sec. 
            
        ListSaveOutputTimes (List float): 
            List of the times to save all the output for each picture in sec. 
    
    Note: 
        This funtion uses the functions : ApplyModelAndSaveOutput and TrainModel and is used in the function Execute in the main file.
       
    
    """
    ### Create the folder where the output will be saved 
    if maskY_N=='Y':
        if not os.path.exists(WorkingDirectory+'/Masks'):    
            os.mkdir(WorkingDirectory+'/Masks')
    if imageY_N=='Y':
        if not os.path.exists(WorkingDirectory+'/MaskedImages'):    
            os.mkdir(WorkingDirectory+'/MaskedImages')
    if NFMaskY_N=='Y':
        if not os.path.exists(WorkingDirectory+'/NonFilteredMasks'):    
            os.mkdir(WorkingDirectory+'/NonFilteredMasks')

    
    ### Import and format the training data from the training data files.
    trainDataTab=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0]])
    for file in ListTrainingDataFile:    
        f=open(file,"r",newline='') 
        TrainData = list(csv.reader(f))
        f.close()
        TrainData.remove(['Class', 'Image', 'x','y','B','G','R','H','S','V','L','a','b'])
        TrainData=np.asarray(TrainData)    
        trainDataTab=np.concatenate((trainDataTab, TrainData), axis=0)
    trainDataTab=np.delete(trainDataTab, (0), axis=0)
    if len(ListTrainingDataFile)>1: # if the user choose more than one file, a new file is saved combining all the selected files.
        np.savetxt(WorkingDirectory+'/trainData_'+str(numberOfClasses)+'classes.csv', trainDataTab, delimiter=",",header='Class,Image,x,y,B,G,R,H,S,V,L,a,b', comments='',fmt='%s')
    trainDataTab=np.delete(trainDataTab,1, 1)
    trainDataTab=np.delete(trainDataTab,1, 1)
    trainDataTab=np.delete(trainDataTab,1, 1)

    ### Format the list of ROI 
    if ROI!='Whole pictures':
        ROI=ast.literal_eval(ROI)

    
    ### Train the model     
    model=TrainModel(trainDataTab, modelname,classesNamesList)  

          
    
    ### Get the size of the reference picture with a 1 pixel difference to avoid any resizing issue
    FirstImage=cv2.imread(ReferencePicture)
    ShapeFirstImage=np.shape(FirstImage)
    a=ShapeFirstImage[0]
    b=ShapeFirstImage[1]
    c=ShapeFirstImage[2]
    ShapeFirstImage2=(a+1,b,c)
    ShapeFirstImage3=(a+1,b+1,c)
    ShapeFirstImage4=(a+1,b-1,c)
    ShapeFirstImage5=(a,b,c)
    ShapeFirstImage6=(a,b+1,c)
    ShapeFirstImage7=(a,b-1,c)    
    ShapeFirstImage8=(a-1,b,c)
    ShapeFirstImage9=(a-1,b+1,c)
    ShapeFirstImage10=(a-1,b-1,c)   

    ### List initialization 
    ListImageWrongSize=[]
    ListRunningTimes=[]
    ListTestDataTimes=[]
    ListApplyModelTimes=[]
    ListSaveOutputTimes=[]
    
    if BiggestBlobY_N=='Y':
        ListAirs=np.array([['Area/Plant','Image Name','Surface','Coverage', 'Aspect Ratio','Extent','Solidity', 'Equivalent Diameter', 'Main axe', 'Secondary axe']])      
    else:
        ListAirs=np.array([['Area/Plant','Image Name','Surface','Coverage']])      
    
    ### Main loop on the image list.
    for i in  ListImageName:
        start_time = time.monotonic() 
        TestImageBGR=cv2.imread(i) 
        ImageName=i.split('/')
        ImageName=ImageName[-1] 
        ImageName=ImageName.split('.')
        ImageName=ImageName[0] 
        ######################################THESE THREE LINES CAN BE USED TO ADD a TIME FILTER ( only keep the pictures between certain hours)
#        hour=float(ImageName[8:10])  #get the time the picture was taken from the name of the file
        hour=float(10)
        if 8<hour<16: # apply a time condition 
        ######################################
            if ROI!='Whole pictures':
                if np.shape(TestImageBGR)==ShapeFirstImage or np.shape(TestImageBGR)==ShapeFirstImage2 or np.shape(TestImageBGR)==ShapeFirstImage3 or np.shape(TestImageBGR)==ShapeFirstImage4 or np.shape(TestImageBGR)==ShapeFirstImage5 or np.shape(TestImageBGR)==ShapeFirstImage6 or np.shape(TestImageBGR)==ShapeFirstImage7 or np.shape(TestImageBGR)==ShapeFirstImage8 or np.shape(TestImageBGR)==ShapeFirstImage9 or np.shape(TestImageBGR)==ShapeFirstImage10  : # Test the size of the picture
                    for j in range(len(ROI)): 
                        #Crop the picture for each ROI
                        x1,y1,x2,y2=ROI[j]
                        if x1>x2:
                            a=x1
                            x1=x2
                            x2=a
                        if y1>y2:
                            a=y1
                            y1=y2
                            y2=a                        
                        croppedImagej=TestImageBGR[y1:y2,x1:x2]  
                        
                        NameArea=ListAreaNames[j] 
                        #Initialize the output names
                        OutputMaskName=''
                        OutputimageName=''
                        OutputNFMaskName=''
                        
                        #Create the output names and folders
                        if maskY_N=='Y': 
                            croppedMaskDirectoryArea=WorkingDirectory+'/Masks/'+NameArea                   
                            if not os.path.exists(croppedMaskDirectoryArea):    
                                os.mkdir(croppedMaskDirectoryArea)
                            OutputMaskName=croppedMaskDirectoryArea+'/'+ImageName+'_crop_'+NameArea+'_mask.png'
        
                        if imageY_N=='Y':    
                            croppedMaskedImagesDirectoryArea=WorkingDirectory+'/MaskedImages/'+NameArea                   
                            if not os.path.exists(croppedMaskedImagesDirectoryArea):    
                                os.mkdir(croppedMaskedImagesDirectoryArea) 
                            OutputimageName=croppedMaskedImagesDirectoryArea+'/'+ImageName+'_crop_'+NameArea+'_maskedImage.png'
                        
                        if NFMaskY_N=='Y':
                            croppedNonFilteredMaskDirectoryArea=WorkingDirectory+'/NonFilteredMasks/'+NameArea                   
                            if not os.path.exists(croppedNonFilteredMaskDirectoryArea):    
                                os.mkdir(croppedNonFilteredMaskDirectoryArea) 
                            OutputNFMaskName=croppedNonFilteredMaskDirectoryArea+'/'+ImageName+'_crop_'+NameArea+'_NFMask.png'
                            
                        # Segment the image with the function ApplyModelAndSaveOutput
                        ListAirs, ListTestDataTimes,ListApplyModelTimes,ListSaveOutputTimes=ApplyModelAndSaveOutput(model, modelname, croppedImagej, ImageName, NameArea, noiseReduction, numberOfClasses, classesNamesList, fusionClassesY_N, maskY_N, InfoY_N, imageY_N, NFMaskY_N, BiggestBlobY_N, chosenArea, OutputMaskName, OutputimageName, OutputNFMaskName, ListAirs, ListTestDataTimes,ListApplyModelTimes,ListSaveOutputTimes)
                       
                        
                        print(str(ImageName)+'   '+str(NameArea)+'  Done!') 
                else: #if the picture is not the right size 
                    ListImageWrongSize.append(i) 
                    print(str(ImageName)+'  Wrong size')
                    
            else: #if the user wants to use the whole pictures
                #Create the output names
                OutputMaskName=WorkingDirectory+'/Masks/'+ImageName+'_mask.png'
                OutputimageName=WorkingDirectory+'/MaskedImages/'+ImageName+'_maskedImage.png'
                OutputNFMaskName=WorkingDirectory+'/NonFilteredMasks/'+ImageName+'_NFMask.png'
                
                # Segment the image with the function ApplyModelAndSaveOutput
                ListAirs, ListTestDataTimes,ListApplyModelTimes,ListSaveOutputTimes=ApplyModelAndSaveOutput(model, modelname, TestImageBGR, ImageName, '', noiseReduction, numberOfClasses, classesNamesList, fusionClassesY_N, maskY_N, InfoY_N, imageY_N, NFMaskY_N, BiggestBlobY_N, chosenArea, OutputMaskName, OutputimageName, OutputNFMaskName, ListAirs, ListTestDataTimes,ListApplyModelTimes,ListSaveOutputTimes)
                
                
                print(str(ImageName)+'  Done!')
                
            end_time = time.monotonic()
            RunningTime=timedelta(seconds=end_time - start_time)
            sec=float(RunningTime.days*86400+RunningTime.seconds+RunningTime.microseconds/1000000)
            
            if i==ListImageName[0]: # get an estimation of the running time after the first picture is done
                print('Running time for 1 image =', RunningTime)
                print('Total running time estimation =', RunningTime*len(ListImageName))
            ListRunningTimes.append(sec)      
            
            
        else: # usefull only if you apply a time filter 
            ListImageWrongSize.append(i) 
            print(str(ImageName)+'  Wrong time')
        
    # Save the info file         
    if len(ListAirs)>1:
        np.savetxt(WorkingDirectory+'/'+'InformationFile.csv', ListAirs, delimiter=",", comments='', fmt='%s')   
    
    return ListImageWrongSize,ListRunningTimes, ListTestDataTimes,ListApplyModelTimes,ListSaveOutputTimes
        
############################################################################################    
        
def ApplyModelAndSaveOutput(model, modelname, imageArray, ImageName,NameArea, noiseReduction, numberOfClasses, classesNamesList, fusionClassesY_N, maskY_N, InfoY_N, imageY_N, NFMaskY_N, BiggestBlobY_N, chosenArea, OutputMaskName, OutputimageName, OutputNFMaskName, ListAirs, ListTestDataTimes,ListApplyModelTimes,ListSaveOutputTimes):
    """ Main function for the segmentation.
    
    Args:
        model (machine learning model): 
            Machine learning model trained with the function  'TrainModel'
        
        modelname (string): 
            Name of the chosen model 
            ('Support Vector Machine (Sklearn)','Random Forest Classifier (Sklearn)','Classification and Regression Tree (Sklearn)')
        
        imageArray (3 channels numpy array):
            Image array with for each pixel the B G R values of the pixel 
            (ex: imageArray=np.array([[[106,255,0],[0,50,13]...], [[106,255,0],[0,50,13]...],...])
            
        ImageName(string):
            Name of the image
            (ex: ImageName='image')
            
        NameArea (string):
            Name of the ROI 
            (ex: NameArea='P1')
        
        noiseReduction (int) 
            Maximal size of the area which will be removed in the function 'noiseRemoval' 
            (ex: noiseReduction=100)
        
        numberOfClasses (int):
            Number of classes 
            (ex: numberOfClasses=2)
        
        classesNamesList (List of strings):
             List of the names of the classes 
             (ex: classesNamesList=['PaddyRice','Background'] )
        
        ROI (List of list of int):
            List of the coordinates of each region of interest (ROI) in the same order as ListAreaNames.
            For each ROI, the first two numbers are the coordinates of the top left corner and the other two are the coordinates of the bottum right corner.
            (ex: ROI=[[0,0,50,50],[50,0,100,50],[0,50,50,100]] )
        
        ListAreaNames (List of strings):
             List of the names of the areas in the same order as the list ROI 
             (ex: ListAreaNames=['P1','P2','P1xP2'] ) 
            
        fusionClassesY_N (string): 
            Two possible values 'Y' or 'N'. If fusionClassesY_N='Y', the user have chosen to fusion two or more classes.
        
        maskY_N (string): 
            Two possible values 'Y' or 'N'. If maskY_N='Y', the user have chosen to save the mask (binary image if there is only two classes and colored flat image if more than two classes)
        
        imageY_N (string): 
            Two possible values 'Y' or 'N'. If imageY_N='Y', the user have chosen to save the reconstructed image (only show the class of interest if there is only two classes and  image+colored filter if there is more than two classes)
        
        InfoY_N (string): 
            Two possible values 'Y' or 'N'. If imageY_N='Y', the user have chosen to save the information file containing for each plant : 'Area/Plant','Image Name','Surface','Coverage', 'Aspect Ratio','Extent','Solidity', 'Equivalent Diameter', 'Main axe', 'Secondary axe'
        
        NFMaskY_N (string): 
            Two possible values 'Y' or 'N'. If maskY_N='Y', the user have chosen to save the mask before any noise reduction and morphological filtering.  
        
        BiggestBlobY_N(string): 
            Two possible values 'Y' or 'N'. If BiggestBlobY_N='Y', the user have chosen to only keep the biggest blob of the mask for analysis.

        chosenArea (string): 
            Name of the class of interest (the one that will be mesured) 
            (ex: 'PaddyRice')
        
        OutputMaskName (string):
            Address used to save the mask.
            (ex: OutputNFMaskName=/Users/Name/Desktop/folder/Masks/P1/image_crop_P1_mask.png')
      
        OutputimageName (string):
            Address used to save the masked image.
            (ex: OutputNFMaskName=/Users/Name/Desktop/folder/MaskedImages/P1/image_crop_P1_maskedImage.png')
       
        OutputNFMaskName (string):
            Address used to save the non-filtered mask.
            (ex: OutputNFMaskName=/Users/Name/Desktop/folder/NonFilteredMasks/P1/image_crop_P1_NFMask.png')
        
        ListAirs (List float)
            List of the areas (number of pixels) of the class of interest for each picture 
            (ex: ListAirs=[1500, 517, 641])
        
        ListTestDataTimes (List float): 
            List of the times to create the test data array (read the picture) for each picture in sec. 
            (ex: ListAirs=[2.1, 2.2, 2.1])
            
        ListApplyModelTimes (List float): 
            List of the times to apply the model to the picture array for each picture in sec. 
            (ex: ListApplyModelTimes=[3.2, 3.2, 3.0])
            
        ListSaveOutputTimes (List float): 
            List of the times to save all the output for each picture in sec. 
            (ex: ListSaveOutputTimes=[1.6, 1.7, 1.5])
       
    Return:
        
        ListAirs (List float)
            List of the areas (number of pixels) of the class of interest for each picture 
            (ex: ListAirs=[1500, 517, 641, 555])
            
        ListTestDataTimes (List float): 
            Argument list with one more element 
            (ex: ListAirs=[2.1, 2.2, 2.1, 2.3])
            
        ListApplyModelTimes (List float): 
            Argument list with one more element 
            (ex: ListApplyModelTimes=[3.2, 3.2, 3.0, 3.1])
            
        ListSaveOutputTimes (List float): 
            Argument list with one more element 
            (ex: ListSaveOutputTimes=[1.6, 1.7, 1.5, 1.6])

    """
    ### Create the test data array
    start_timeTestData = time.monotonic()
    
    TestData=creatTestData(imageArray)
    
    end_timeTestData = time.monotonic() 
    RunningTime=timedelta(seconds=end_timeTestData - start_timeTestData)
    sec=float(RunningTime.days*86400+RunningTime.seconds+RunningTime.microseconds/1000000)
    ListTestDataTimes.append(sec)
    
    ### Apply the model to the test data
    start_timeModel = time.monotonic()
    
    Resultmodel=ApplyModel(TestData, modelname, model)
    
    end_timeModel = time.monotonic() 
    RunningTime=timedelta(seconds=end_timeModel - start_timeModel)
    sec=float(RunningTime.days*86400+RunningTime.seconds+RunningTime.microseconds/1000000)
    ListApplyModelTimes.append(sec)
    
    ### Create and save the output 
    start_timeOutput = time.monotonic()
    
    Mask=Resultmodel.reshape(np.shape(imageArray)[0],np.shape(imageArray)[1])
    
    #Save the non filtered mask in shades of gray
    if NFMaskY_N=='Y':
        NFMask=Mask.astype('int')
        NFMask=(NFMask/(numberOfClasses-1))*255
        NFMask=NFMask.astype(int)
        cv2.imwrite(OutputNFMaskName,NFMask)
    
    # apply a noise reduction filter to the mask
    FilteredMask=noiseRemoval(Mask, noiseReduction, numberOfClasses)    
    
    if numberOfClasses>2 and fusionClassesY_N=='N' :
        #create a colored mask with 1 color=1class
        coloredMask=colorfilter(FilteredMask)
        if maskY_N=='Y':   
            cv2.imwrite(OutputMaskName,coloredMask)
    
        if imageY_N=='Y':
            MaskedImage=0.3*coloredMask+0.7*imageArray
            cv2.imwrite(OutputimageName,MaskedImage)
    
    else:
        # create a black and white mask with the class of interest in white
        BandWMask=FilteredMask*0
        List=[]
        for AreaName in chosenArea:
            if AreaName in classesNamesList:
                List.append(classesNamesList.index(AreaName))
                
        for AreaNumber in List:
            BandWMask[FilteredMask==(AreaNumber)]=255
        
        BandWMask=BandWMask.astype('uint8')
        
        #If the user choosed to only keep the biggest blob and do shape analysis
        if BiggestBlobY_N=='Y':
        
            # Detect the blobs and there contour in this black and white mask
            im2, contours, hierarchy = cv2.findContours(BandWMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            #find the biggest blob, erase the others and keep the smaller black blobs in the biggest white blob
            if contours!=[]:
                surfaceMainBlob = 0
                contourMainBlob=[]
                RankMainBlob=0
                for i in range(len(contours)):
                    if cv2.contourArea(contours[i])>surfaceMainBlob:
                        contourMainBlob=contours[i]
                        surfaceMainBlob=cv2.contourArea(contours[i])
                        RankMainBlob=i
                
                ListSecondaryBlod=[]
                
                for i in range(len(hierarchy[0])):
                    if hierarchy[0,i][3] ==RankMainBlob:
                        ListSecondaryBlod.append(contours[i])       
                
                FilteredMask2=imageArray*0
                L=[]
                L.append(contourMainBlob)
                FilteredMask2=cv2.drawContours(FilteredMask2, L, 0, (255,255,255), -1)
                FilteredMask2=cv2.drawContours(FilteredMask2, ListSecondaryBlod, -1, (0,0,0), -1)
                
                #Save the final mask
                if maskY_N=='Y':   
                    cv2.imwrite(OutputMaskName,FilteredMask2)
    
                # calculate some of the properties of the main blob 
                hull = cv2.convexHull(contourMainBlob)
                rect = cv2.minAreaRect(contourMainBlob)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                axes=rect[1]
                axe1=axes[0]
                axe2=axes[1]
                
                if axe1<axe2:
                    a=axe1
                    axe1=axe2
                    axe2=a
            
                # Save the masked image and draw some of the blob properties (convexhull, rectangle, main axes...)        
                if imageY_N=='Y':
                    FilteredMask3=FilteredMask2
                    FilteredMask3[FilteredMask2==255]=1
                    FilteredMask3[FilteredMask2==0]=0.1            
                    MaskedImage=FilteredMask3*imageArray
                    
                    MaskedImage=cv2.drawContours(MaskedImage,[box],0,(0,255,0),1)
                    MaskedImage=cv2.ellipse(MaskedImage,rect,(0,255,0),1)
                    
                    x1,y1=box[0]
                    x2,y2=box[1]
                    x3,y3=box[2]
                    x4,y4=box[3]
                                
                    l1x1=int((x3+x2)/2)
                    l1y1=int((y3+y2)/2)
                    
                    l1x2=int((x4+x1)/2)
                    l1y2=int((y4+y1)/2) 
                    
                    l2x1=int((x1+x2)/2)
                    l2y1=int((y1+y2)/2)
                    
                    l2x2=int((x4+x3)/2)
                    l2y2=int((y4+y3)/2)  
                    
                    MaskedImage=cv2.line(MaskedImage,(l1x1,l1y1),(l1x2,l1y2),(255,255,0),1) # blue
                    MaskedImage=cv2.line(MaskedImage,(l2x1,l2y1),(l2x2,l2y2),(255,255,255),1) # white
                    L=[]
                    L.append(hull)
                    MaskedImage=cv2.drawContours(MaskedImage, L, 0, (0,0,255), 1)
                    
                    cv2.imwrite(OutputimageName,MaskedImage)
                
                #Save the information in ListAirs
                if InfoY_N=='Y':
                    for i in ListSecondaryBlod:
                        surfaceSecondaryBlobi=cv2.contourArea(i)
                        surfaceMainBlob=surfaceMainBlob-surfaceSecondaryBlobi
                        
                        
                    x,y,w,h = cv2.boundingRect(contourMainBlob)
                    aspect_ratio = float(w)/h
                    rect_area = w*h
                    extent = float(surfaceMainBlob)/rect_area
                    equi_diameter = np.sqrt(4*surfaceMainBlob/np.pi)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(surfaceMainBlob)/hull_area
                                    
                    TotalSurface=len(imageArray)*len(imageArray[0])
                    ListAirs=np.vstack([ListAirs, [NameArea, ImageName , surfaceMainBlob, surfaceMainBlob/TotalSurface, aspect_ratio, extent, solidity,equi_diameter, axe1, axe2]])
                        
            
            else: #if No blob is found, just save a black rectangle
                FilteredMask2=imageArray*0
                if maskY_N=='Y':   
                    cv2.imwrite(OutputMaskName,FilteredMask2)
                if imageY_N=='Y':
                    cv2.imwrite(OutputimageName,FilteredMask2)
                if InfoY_N=='Y':
                    ListAirs=np.vstack([ListAirs, [NameArea, ImageName , 0, 0, 0, 0, 0, 0,0,0]])
        
        #If the user decided to keep all the blobes and not do the shape analysis
        else:
            #Save the final mask
            if maskY_N=='Y':   
                cv2.imwrite(OutputMaskName,BandWMask)
        
            # Save the masked image and draw some of the blob properties (convexhull, rectangle, main axes...)        
            if imageY_N=='Y':
                FilteredMask3=np.zeros((len(BandWMask),len(BandWMask[0]),3))
                FilteredMask3[BandWMask==255]=[1,1,1]
                FilteredMask3[BandWMask==0]=[0.1,0.1,0.1]           
                MaskedImage=FilteredMask3*imageArray
                
                cv2.imwrite(OutputimageName,MaskedImage)
            
            #Save the information in ListAirs
            if InfoY_N=='Y':
                surfaceClassOfInterest=np.sum(BandWMask)/255
                TotalSurface=len(imageArray)*len(imageArray[0])
                ListAirs=np.vstack([ListAirs, [NameArea, ImageName , surfaceClassOfInterest, surfaceClassOfInterest/TotalSurface]])
                        
      
        
            
                
    end_timeOutput = time.monotonic() 
    RunningTime=timedelta(seconds=end_timeOutput - start_timeOutput)
    sec=float(RunningTime.days*86400+RunningTime.seconds+RunningTime.microseconds/1000000)
    ListSaveOutputTimes.append(sec)
    return ListAirs, ListTestDataTimes,ListApplyModelTimes,ListSaveOutputTimes
    
        
############################################################################################    
      
    
def creatTestData(testImage):
    """ Function to create the test data array.
    
    Args:
        testImage (numpy array ): 
            Image array with for each pixel the B G R values of the pixel
            (ex: imageArray=np.array([[[106,255,0],[0,50,13]...], [[106,255,0],[0,50,13]...],...])
    
    Return:
        fusion (numpy array): 
            Array containing for each pixel RGB HSV and Lab. One pixel per line 
            (ex: )
    
    Note:
       This function  is used in the function 'ApplyModelAndSaveOutput' in this file. 
    
    """
    
    bgrArray = testImage[:,:,0:3] # make sure that there is only 3 channels (BGR) per pixel
        
    bgrArray=bgrArray.reshape(np.shape(bgrArray)[0]*np.shape(bgrArray)[1],1,3) # transform the rectangular array into a column with one pixel per row
    bgrArray=bgrArray.astype('uint8')
    
    #Calculate other color properties from the bgr values
    hsvArray = cv2.cvtColor(bgrArray,cv2.COLOR_BGR2HSV)

    LabArray=cv2.cvtColor(bgrArray,cv2.COLOR_BGR2Lab) 
    
    #Save everything in a big array
    fusion=np.concatenate((bgrArray, hsvArray), axis=1)
    fusion=np.concatenate((fusion, LabArray),axis=1)
    fusion=fusion.reshape((len(fusion),9))

    return fusion

############################################################################################    
    
def TrainModel(trainData, modelname,classesNamesList):

    """ Function to create and train the machine learning model
    
    Args: 
        trainData (numpy array): 
            Array with for each pixel the class and the BGR HSV Lab values of the pixel 
            (ex:  trainData=np.array([[PaddyRice, 116, 179, 147,  45, 90, 179, 177, 106, 157],[Background,132, 121, 123, 125, 21, 132, 131, 131, 122]]))
            
        modelname (string): 
            Name of the chosen model 
            ('Support Vector Machine (Sklearn)','Random Forest Classifier (Sklearn)','Classification and Regression Tree (Sklearn)')
    
        classesNamesList (List of strings): 
            List of the names of the classes 
            (ex: classesNamesList=['PaddyRice','Background'] )
    
    Return:
        model (machine learning model): 
            machine learning model trained with the trainingData array.
            (ex of use: result=model.predict(testData) with testData an array containg the BGR HSV Lab values of each pixel )
    
    Note: 
        This function  is used in the function 'ApplyModelAndSaveOutput' in this file.
    """

    Response=trainData[:,0]
    Response=Response.astype('str')
    ResponseNumber=[]
    for i in range(len(Response)):
        for j in range(len(classesNamesList)):
            if Response[i]==classesNamesList[j]:        
                ResponseNumber.append(j)
    trainData=trainData[:,1:10] 
    trainData=trainData.astype('float32')
    
    if modelname=='Support Vector Machine (Sklearn)':
        model = svm.SVC()
        model.fit(trainData, ResponseNumber) 
    
    if modelname=='Random Forest Classifier (Sklearn)':
        model = RandomForestClassifier(bootstrap= True, n_estimators= 100, oob_score=True)
        model.fit(trainData, ResponseNumber)
        
    if modelname=='Classification and Regression Tree (Sklearn)':
        model = tree.DecisionTreeClassifier()
        model.fit(trainData, ResponseNumber)
    
        
    return model

    
def ApplyModel(testData, modelname, model):
    """ Function to apply the model created with the function  'TrainModel' to the testData created with the function  'creatTestData'
    
    Args: 
        testData (numpy array): 
            Array with for each pixel the B G R HSV Lab values of the pixel 
            (ex:  trainData=np.array([[116, 179, 147,  45, 90, 179, 177, 106, 157], [132, 121, 123, 125, 21, 132, 131, 131, 122]])   )
    
        modelname (string): 
            Name of the chosen model 
            ('Support Vector Machine (Sklearn)','Random Forest Classifier (Sklearn)','Classification and Regression Tree (Sklearn)')
    
        model (machine learning model): 
            Machine learning model trained with the function  'TrainModel'
    
    Return:
        result2 (numpy array containing int): 
            Array of 1 column containing the number of the class corresponding to each pixel 
            (ex: result2= np.array([0,0,1,1]) with 0 corresponding to 'PaddyRice' and 1 corresponding to 'Background' if classesNamesList=[PaddyRice,Background] )
    
    Note: 
        This function  is used in the function 'ApplyModelAndSaveOutput' in this file and uses the output of the function 'TrainModel' (model) and the function 'creatTestData' (testData). The third argument (modelname) is chosen in the combo box of the GUI in the main file. 
    """
    testData=testData.astype('float32')
    
    if modelname=='Support Vector Machine (Sklearn)':
        a,result = model.predict(testData, cv2.ml.ROW_SAMPLE)

    if modelname=='Random Forest Classifier (Sklearn)':
        result = model.predict(testData)
        
    if modelname=='Classification and Regression Tree (Sklearn)':
        result = model.predict(testData)

    
    result2=result.reshape(np.shape(result)[0])
    result2=result2.astype('int')
    return result2



############################################################################################    
        
    
def noiseRemoval(array, minSize, classes): 
    """ Function to remove the smallest area of the picture and reduce noise.
    
    Args:
        array (numpy array): 
            Image array with for each pixel a number corresponding to the class 
            (ex: array= np.array([[0,0,1,1],[1,0,0,1],[1,0,1,0]]) with 0 corresponding to 'PaddyRice' and 1 corresponding to 'Background'
        
        minSize (int):
            Maximal size of the area which will be removed.
            (ex: minSize=100)
        
        classes (int):
            number of classes  
            (ex: classes=2)                                                                  
    Return:
        img (numpy array): 
            Filtered (=no more noise) image array with for each pixel a number corresponding to the class 
            (ex: array= np.array([[0,0,1,1],[1,0,0,1],[1,0,1,0]]) with 0 corresponding to 'PaddyRice' and 1 corresponding to 'Background' 
    
    Note:
       This function  is used in the function 'ApplyModelAndSaveOutput' in this file. 
  
    """
    img=array.astype('int')
    for i in range(classes):
        B=(img!=i) # return a bool array
        B = morphology.remove_small_objects(B, min_size=minSize, connectivity=1) 
        img[B==False]=i
        
    return img


############################################################################################    
    
    
def colorfilter(FilteredMask):
    """ Function to transform a one channel array to a three channel colored array.
    
    Args:
        FilteredMask (numpy array): 
            Image array with for each pixel a number corresponding to the class 
            (ex: array= np.array([[0,0,1,1],[1,0,0,1],[1,0,1,0]]) with 0 corresponding to 'PaddyRice' and 1 corresponding to 'Background'
                                                                            
    Return:
        FilteredMask3channels (numpy array): 
            3 channels array with for each pixel a color in BGR value corresponding to a class  
            (ex: array= np.array([[[0,0,0],[0,0,0],[255,255,255],[255,255,255]],[[255,255,255],[0,0,0],[0,0,0],[255,255,255]],[[255,255,255],[0,0,0],[255,255,255],[0,0,0]]]) with [0,0,0] (black) corresponding to 'PaddyRice' and [255,255,255] (white) corresponding to 'Background' 
    
    Note:
       This function  is used in the function 'ApplyModelAndSaveOutput' in this file. 
    
    """
    FilteredMask3channels=np.stack((FilteredMask,)*3, axis=-1)
    for i in range(len(FilteredMask)):
        for j in range(len(FilteredMask[0])):
            if FilteredMask[i,j]==0:
                FilteredMask3channels[i,j]=[0,0,0]
            if FilteredMask[i,j]==1:
                FilteredMask3channels[i,j]=[255,255,255]
            if FilteredMask[i,j]==2:
                FilteredMask3channels[i,j]=[255,0,0]
            if FilteredMask[i,j]==3:
                FilteredMask3channels[i,j]=[0,255,0]
            if FilteredMask[i,j]==4:
                FilteredMask3channels[i,j]=[0,0,255]
    return FilteredMask3channels
        

