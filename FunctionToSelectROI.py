import cv2

coordinaterectangle=[] 
drawing=False # true if mouse is pressed
i=0
FirstDone=False


def SelectROI(ImageName,SameSize,displaySize):
    """ Function to draw the line on the picture and save the coordinates in a list.

    Args:        
        ImageName (string):
            Addresses of the picture that the user wants to use to select the regions of interest (ROI).  
            (ex: ImageName='/Users/Name/Desktop/image.png')
        
        SameSize (string): 
            Two possible values 'Y' or 'N'. If imageY_N='Y', the user have chosen to use ROIs all of the same size.
              
        displaySize (int):
            Integer use to calculate the resizing coefficient for the display of the pictures
            (ex: displaySize=500)
            
    Return:
        coordinaterectangle (List of lists of int):
            List of the coordinates of each ROI. For each ROI, the first two numbers are the coordinates of the top left corner and the other two are the coordinates of the bottom right corner.
            (ex: coordinaterectangle=[[0,0,50,50],[50,0,100,50],[0,50,50,100]] )  
            
    
    Note:
       This function  is used in the function 'SelectArea' in the class 'SelectAreaWindow' in the file MainFileGUI.py
    
    """

    global coordinaterectangle, drawing, i,FirstDone
    
    #### First possibility : SameSize=='Y', the user wants to have same size rectangles.
    
    if SameSize=='Y':
        #Initialize the variables
        coordinaterectangle=[] 
        drawing=False # true if mouse is pressed
        FirstDone=False
        i=0
        img1 = cv2.imread(ImageName) #The image is opened 
        
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
        
        #Choose the name of the window
        cv2.namedWindow('Select the region of interest',cv2.WINDOW_NORMAL) 
        
        #call the function to draw the rectangles on the pictures
        cv2.setMouseCallback('Select the region of interest',draw_rectangle, param=(img,coeff)) 
        
        #Update the image display continually until the first rectangle is drawn
        while FirstDone==False:
            cv2.imshow('Select the region of interest',img)
            key = cv2.waitKey(20) & 0xFF
            if key==ord('e'): #To exit and stop the drawing mode PRESS 'E'
                break
         
        if coordinaterectangle!=[]:
            #Once the first rectangle is drawn, its hight and width are calculated
            x1,y1,x2,y2=coordinaterectangle[0]
            x1=x1*coeff
            x2=x2*coeff
            y1=y1*coeff
            y2=y2*coeff
            W=x2-x1
            H=y2-y1    
            
            cv2.namedWindow('Select the region of interest',cv2.WINDOW_NORMAL) 
            
            # Another function is used
            cv2.setMouseCallback('Select the region of interest',Same_Rectangle, param=(img, coeff, H, W)) 
            
            #Update the image display continually until the key 'e' is pressed
            while(1):
                cv2.imshow('Select the region of interest',img)
                key = cv2.waitKey(20) & 0xFF
                if key==ord('e'): 
                    break
     
        #Close the window
        cv2.destroyAllWindows()# There is a problem with spyder and destroyAllWindows doesn't work alone> These few lines work for some reason...
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        i=0

    #### Second possibility : SameSize=='N', the user wants to have different size rectangles. Same thing as the first part but only the draw_rectangle function is used. 

    else:
        coordinaterectangle=[] 
        drawing=False # true if mouse is pressed
        FirstDone=False
        i=0
        img1 = cv2.imread(ImageName) #The image is opened 
       
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
        
        cv2.namedWindow('Select the region of interest',cv2.WINDOW_NORMAL) #define the name of the window

        cv2.setMouseCallback('Select the region of interest',draw_rectangle, param=(img,coeff)) #call the function
        
        #Update the image display continually until the key 'e' is pressed
        while(1):
            cv2.imshow('Select the region of interest',img)
            key = cv2.waitKey(20) & 0xFF
            if key==ord('e'): #To exit and stop the drawing mode PRESS 'E'
                break
     
        cv2.waitKey(0) # There is a problem with spyder and destroyAllWindows doesn't work alone> These few lines work for some reason when a keyboard piece is pressed 
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        i=0

    return coordinaterectangle

def draw_rectangle(event,x,y,flags,param):
    """ Function to draw rectangles on the picture and save the coordinates in a list.
    
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
            List of all the other parameters that the user wants to use in the function. It is the only way to pass extra parameters to the function when using cv2.setMouseCallback 
            Here param=[img, coeff]
                    (Here cls= TrainingData)
                img (3 channels numpy array) : 
                    Image array with for each pixel the B G R values of the pixel
                    (ex: imageArray=np.array([[[106,255,0],[0,50,13]...], [[106,255,0],[0,50,13]...],...])
                coeff (float):
                    Coefficient use to resize the image for the display.
                    (ex: coeff=1.3)
                    
    Return:
        coordinaterectangle (List of lists of int):
            List of the coordinates of each ROI. For each ROI, the first two numbers are the coordinates of the top left corner and the other two are the coordinates of the bottum right corner.
            (ex: coordinaterectangle=[[0,0,50,50],[50,0,100,50],[0,50,50,100]] )  
    
    Note:
       This function is used in the function 'SelectROI' in the file FunctionToSelectROI. 
    
    """

    (img, coeff)=param
    global ix,iy, drawing, coordinaterectangle, i,FirstDone
    if event == cv2.EVENT_LBUTTONDOWN: #when the button is pushed, the first pixel is recorded and a circle is draw on the picture
        drawing = True
        coordinaterectangle=coordinaterectangle+[[0,0,0,0]]
        coordinaterectangle[i][0]=int(x/coeff)
        coordinaterectangle[i][1]=int(y/coeff)
        cv2.circle(img, (x,y), 3, (0,0,255), 3)
        ix,iy=x,y


    elif event == cv2.EVENT_MOUSEMOVE: #If the mouse stay pushed and the mouse moves, the pixels are recorded
        if drawing == True:
            cv2.line(img, (ix,iy), (x,iy), (0,0,255), 1)
            cv2.line(img, (ix,iy), (ix,y), (0,0,255), 1)
                
    elif event == cv2.EVENT_LBUTTONUP: #When the button is released, the last pixel is recorded and the mouse stop drawing
        drawing = False
        coordinaterectangle[i][2]=int(x/coeff)
        coordinaterectangle[i][3]=int(y/coeff)

        cv2.rectangle(img, (ix,iy), (x,y), (0,0,255), 3)
        i=i+1
        FirstDone=True
        
    return coordinaterectangle

def Same_Rectangle(event,x,y,flags,param):
    """ Function to draw rectangles of a chosen size on the picture by clicking in the center of the area and save the coordinates in a list.
    
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
            List of all the other parameters that the user wants to use in the function. It is the only way to pass extra parameters to the function when using cv2.setMouseCallback 
            Here param=[img, coeff, H, W]
                    (Here cls= TrainingData)
                img (3 channels numpy array) : 
                    Image array with for each pixel the B G R values of the pixel
                    (ex: imageArray=np.array([[[106,255,0],[0,50,13]...], [[106,255,0],[0,50,13]...],...])
                coeff (float):
                    Coefficient use to resize the image for the display.
                    (ex: coeff=1.3)
                H and W (int):
                    Hight and width of the rectangle
                     
                    
    Modify:
        coordinaterectangle (List of lists of int):
            List of the coordinates of each ROI. For each ROI, the first two numbers are the coordinates of the top left corner and the other two are the coordinates of the bottum right corner.
            (ex: coordinaterectangle=[[0,0,50,50],[50,0,100,50],[0,50,50,100]] )  
    
    Note:
       This function is used in the function 'SelectROI' in the file FunctionToSelectROI. 
    
    """
    (img, coeff, H,W)=param
    global coordinaterectangle, i
    if event == cv2.EVENT_LBUTTONDOWN: #when the button is pushed, a rectangle is drawn around the coordinates.
        x1=int(x-(W/2))
        y1=int(y-(H/2))
        x2=int(x+(W/2))
        y2=int(y+(H/2))
        
        #If the rectangle coordinates are outside of the picture, the rectangle is moved so that the edge of the rectangle is on the side of the window
        if x1<=0:
            x1=int(1)
            x2=x1+int(W)
        if y1<=0:
            y1=int(1)
            y2=y1+int(H)

        if x2>=len(img[0]):
            x2=int(len(img[0])-1)
            x1=int(x2-W)
        if y2>=len(img):
            y2=int(len(img)-1)
            y1=int(y2-H)
        
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
        coordinaterectangle=coordinaterectangle+[[0,0,0,0]]
        coordinaterectangle[i][0]=int(x1/coeff)
        coordinaterectangle[i][1]=int(y1/coeff)
        coordinaterectangle[i][2]=int(x2/coeff)
        coordinaterectangle[i][3]=int(y2/coeff)
        i=i+1
        
    