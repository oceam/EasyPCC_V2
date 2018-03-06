import cv2
import numpy as np
import os


def draw_line(event,x,y,flags,param):
    """ Function to draw the line on the picture and save the coordinates in a list.
    
    Args:
        event (-): 
            Mouse event given by the function cv2.setMouseCallback 
            (ex: event=cv2.EVENT_LBUTTONDOWN or cv2.EVENT_MOUSEMOVE or cv2.EVENT_LBUTTONUP)
        
        x and y (int):
            coordinates of the mouse given by the function cv2.setMouseCallback
            (ex: x=50, y=60)
        
        flags (-):
            parameter given by the function cv2.setMouseCallback  
            
        param (List):
            List of all the other parameters that the user wants to use in the function. It is the only way to pass extra paramters to the function when using cv2.setMouseCallback 
            Here param=[cls, img, nameImage, coeff, color]
                cls (class): 
                    Class in which the function is used 
                    (Here cls= TrainingData)
                img (3 channels numpy array) : 
                    Image array with for each pixel the B G R values of the pixel
                    (ex: imageArray=np.array([[[106,255,0],[0,50,13]...], [[106,255,0],[0,50,13]...],...])
                nameImage (string):
                    Name of the image (without the full address)
                    (ex: nameImage='image1')
                coeff (float):
                    Coefficient use to resize the image for the display.
                    (ex: coeff=1.3)
                color(int triplet):
                    BGR values of the color used to draw the lines on the picture
                    (ex: color=(0,0,255) )
                    
    Return:
        No return, the coordinates list is directly saved in an attribut of the class TrainingData (TrainingData.coordinates).
    
    Note:
       This function  is used in the function 'SelectOneClass' in the class TrainingData. It is important that this function  is written **before** the class.
    
    """
    (cls, img, nameImage, coeff, color)=param
    if event == cv2.EVENT_LBUTTONDOWN: #when the button is pushed, the first pixel is recorded and a circle is drawn on the pixure
        cls.drawing = True
        cls.coordinates = cls.coordinates+[[int(x/coeff),int(y/coeff),nameImage]] # for each selected pixels, the neighboring pixels are also recorded (four pixels in total)
        cls.coordinates = cls.coordinates+[[int(x/coeff)+1,int(y/coeff),nameImage]] 
        cls.coordinates = cls.coordinates+[[int(x/coeff)+1,int(y/coeff)+1,nameImage]] 
        cls.coordinates = cls.coordinates+[[int(x/coeff),int(y/coeff)+1,nameImage]] 
        cls.numberpixels=cls.numberpixels+4 
        cv2.circle(img,(x,y),2,color,-1) #draw the circle
        cls.ix,cls.iy=x,y #save the last coordinates

    elif event == cv2.EVENT_MOUSEMOVE: #If the mouse stay pushed and the mouse moves, the pixels are recorded
        if cls.drawing == True:
            cls.coordinates = cls.coordinates+[[int(x/coeff),int(y/coeff),nameImage]] 
            cls.coordinates = cls.coordinates+[[int(x/coeff)+1,int(y/coeff),nameImage]] 
            cls.coordinates = cls.coordinates+[[int(x/coeff)+1,int(y/coeff)+1,nameImage]] 
            cls.coordinates = cls.coordinates+[[int(x/coeff),int(y/coeff)+1,nameImage]] 
            cls.numberpixels=cls.numberpixels+4
            cv2.line(img,(cls.ix,cls.iy),(x,y),color,2) # draw a line 
            cls.ix,cls.iy=x,y 

                
    elif event == cv2.EVENT_LBUTTONUP: #When the button is released, the last pixel is recorded and the mouse stop drawing
        cls.drawing = False 
        cls.ix,cls.iy=x,y
        
class TrainingData:
    def __init__(self):
        #initialization of the class 
        self.coordinates=[]
        self.drawing=False
        self.numberpixels=0
        
    def SelectOneClass(self,ImageNameList, workingDirectory, NameClasse, color,displaySize):
        """ Function to draw the line on the picture and save the coordinates in a list.
        
        Args:
            self (class): 
                Allow the function to use the class attributes 
            
            ListImageName (List of strings):
                List of the addresses of all the pictures that the user wants to use to create the training data set.  
                (ex: ListImageName=['/Users/Name/Desktop/image.png','/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png'])
            
            WorkingDirectory (string): 
                Address of the working directory
                (ex: WorkingDirectory='/Users/Name/Desktop/folder')
                
            NameClasse (string):
                Name of the class that the user is going to select pixels for. 
                (ex: NameClasse='PaddyRice')
                
            color(int triplet):
                BGR values of the color used to draw the lines on the picture
                (ex: color=(0,0,255) )
                  
            displaySize (int):
                Integer use to calculate the resizing coefficient for the display of the pictures
                (ex: displaySize=500)
                
                
                
        Return:
            ListTrainingDataFile (List of strings):
                List of the addresses of the training data files that was created. One file per pictures and class. 
                (ex: ListTrainingDataFile= ['/Users/Name/Desktop/folder/TrainingData/TrainingDataIndividualFiles/trainData_PaddyRice_image1.csv','/Users/Name/Desktop/folder/TrainingData/TrainingDataIndividualFiles/TrainingDataIndividualFiles/trainData_Background_image1.csv'])
                
        
        Note:
           This function  is part of the class TrainingData and is used in the function 'CreatTrainingData' in the class 'TrainingWindow' in the file MainFileGUI.py
        
        """
        

        ## initialize the list
        self.coordinates=[]
        ListTrainingDataFile=[]
        
        ##Creeat the folder that will be use to save the output
        if not os.path.exists(workingDirectory+'/SeeTheSelectedPixelsOnThePictures'):    
            os.mkdir(workingDirectory+'/SeeTheSelectedPixelsOnThePictures')
        if not os.path.exists(workingDirectory+'/TrainingDataIndividualFiles'):    
            os.mkdir(workingDirectory+'/TrainingDataIndividualFiles')
       
        ## Loop on the list of image 
        for i in range(len(ImageNameList)):
            
            ########## First Part :Selecting the pixels##################   
            
            self.numberpixels=0
            #Extract the name of the image from its address
            ImageName=ImageNameList[i]
            ImageNamebis=ImageName.split('/')
            ImageNamebis=ImageNamebis[-1] 
            ImageNamebis=ImageNamebis.split('.')
            ImageNamebis=ImageNamebis[0]
            
            # Open the image
            img1 = cv2.imread(ImageName)
            
            #resize the image according to the screen size 
            if len(img1[:,0])>=len(img1[0,:]) : #if the picture Height is bigger than its width 
                H=displaySize
                coeff=H/len(img1[:,0])
                W=int(coeff*len(img1[0,:]))
            else:
                W=displaySize
                coeff=W/len(img1[0,:])
                H=int(coeff*len(img1[:,0]))
            
            img = cv2.resize(img1, (W, H)) 
            
            
            #Open a window and display the image
            nameWindow='Select pixels for the class: '+NameClasse
            cv2.namedWindow(nameWindow,cv2.WINDOW_NORMAL) #define the name of the window
            cv2.resizeWindow(nameWindow, W,H+20) 
            
            #Call the function to select the pixels
            cv2.setMouseCallback(nameWindow,draw_line, param=(self, img, ImageNamebis,coeff, color)) #call the function 
            
            
            #Update the image display continually until 'E' is pressed
            while(1): 
                cv2.imshow(nameWindow,img)
                key = cv2.waitKey(20) & 0xFF
                
                #Start over the labelling of the last displayed picture when the key 'q' is pressed
                if key==ord('q'): 
                    img = cv2.resize(img1, (W, H)) 
                    cv2.namedWindow(nameWindow,cv2.WINDOW_NORMAL) #define the name of the window
                    cv2.resizeWindow(nameWindow, W,H+20) 
                    cv2.setMouseCallback(nameWindow,draw_line, param=(self, img, i,coeff, color)) #call the function 

                    del self.coordinates[-self.numberpixels:-1]
                    del self.coordinates[-1]
                
                # when the key 'e' is pressed, save a the image with the line as a reminder and go to the next picture 
                if key==ord('e'): #To exit and stop the drawing mode PRESS 'E'
                    cv2.imwrite(workingDirectory+'/SeeTheSelectedPixelsOnThePictures/'+NameClasse+'_'+ImageNamebis+'_selectedpixels.png',img)
                    break

            ############### Part 2 Create the csv file and caluclate colors values################
            
            fusion=[]   
            
            if self.coordinates!=[]:
                img=cv2.imread(ImageName)   
                bgrList=[]
                classesList=[]
                ImgNameList=[]
                coordinates=[]
                for k in range(len(self.coordinates)): # for each pixel :
                    (xk,yk,imgNamek)=self.coordinates[k]# get the coordinates and the category
                    if xk<len(img[0]) and yk<len(img):
                        # Get the BGR value
                        b=img[yk,xk,0] 
                        g=img[yk,xk,1]
                        r=img[yk,xk,2]
                        bgrList=bgrList+[[[b,g,r]]]
                        classesList=classesList+[[NameClasse]]
                        ImgNameList=ImgNameList+[[imgNamek]]
                        coordinates=coordinates+[[xk,yk]]
                
                bgrArray=np.asarray(bgrList)
                bgrArray=bgrArray.astype('uint8')
                
                classesArray=np.asarray(classesList) 
                ImgNameArray=np.asarray(ImgNameList)
                coordinates=np.asarray(coordinates)
                
                hsvArray = cv2.cvtColor(bgrArray,cv2.COLOR_BGR2HSV) # calculate the HSV values for each pixels from the BGR values
                
                LabArray=cv2.cvtColor(bgrArray,cv2.COLOR_BGR2Lab) # calculate the Lab values for each pixels from the BGR values
            
                
                #Concatenate and reshape all the informartions in one big array 
                fusion=np.concatenate((bgrArray, hsvArray), axis=1)
                fusion=np.concatenate((fusion, LabArray),axis=1)
                fusion=fusion.reshape((len(fusion),9))
                fusion=np.concatenate((coordinates, fusion), axis=1)
                fusion=np.concatenate((ImgNameArray, fusion), axis=1)
                fusion=np.concatenate((classesArray, fusion), axis=1)
                
                #Save the array 
                np.savetxt(workingDirectory+'/TrainingDataIndividualFiles/trainData_'+str(NameClasse)+'_'+ImageNamebis+'.csv', fusion, delimiter=",",header='Class,Image,x,y,B,G,R,H,S,V,L,a,b', comments='',fmt='%s')
                ListTrainingDataFile.append(workingDirectory+'/TrainingDataIndividualFiles/trainData_'+str(NameClasse)+'_'+ImageNamebis+'.csv')
            
            #Reset the coordinates list for the next picture
            self.coordinates=[]
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        
        return ListTrainingDataFile 
            
    
    

    
    
    
