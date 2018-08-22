import sys
from PyQt5.QtWidgets import ( QCheckBox,QScrollArea, QGridLayout, QFrame, QSpinBox, QComboBox, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QApplication, QWidget, QLabel, QFileDialog, QDialog)
import os
import time
from datetime import timedelta
import numpy as np
import cv2
import csv
from FunctionToCreateTrainingData import TrainingData
from FunctionForSegmentation import Segmentation
from FunctionToSelectROI import SelectROI
import ast
from random import randint

class TrainingWindow(QDialog):
    """ This class contain the GUI for the window used to create the training data. 
    This window can be opened from the class 'MainWindow' by clicking on the button 'Create new set of training data' (MainWindow.button_newTrainingData)
    This class has two parameters: 
        WorkingDirectory (string): 
            Address of the working directory
            (ex: WorkingDirectory='/Users/Name/Desktop/folder'
        DisplaySize (int):
            Integer use to calculate the resizing coefficient for the display of the pictures
            (ex: displaySize=500)
    """

    def __init__(self, parent=None, workingDirectory=None, DisplaySize=None):
        
        super(TrainingWindow, self).__init__(parent)
        self.displaySize=DisplaySize
        self.workingDirectory = workingDirectory
        self.classes=2
        self.init_TrainingWindow()
        
    def init_TrainingWindow(self):
        """Function to set up the GUI and link the buttons to their corresponding functions """
        #Name the window
        self.setWindowTitle('Create training data')
        
        #Create all the widgets
        self.label_newTrainingDataInfo = QLabel('When you press start, the first picture will appear. \n First select some pixels of the foreground/plant which will appear in red. \n When you are done, press "E" on your keyboard and select a few pixels of the background. \n When your are done press "e" again to save the data and press the space bar to pass to the next picture. \n Repeat until all the pictures have been annotated. ')
        self.label_newTrainingData = QLabel('Choose the pictures you want to use for creating the training data :')
        
        self.label_ModeTrainingPictures = QLabel('Mode:')
        self.label_NumberPictures = QLabel('Path to the training pictures:')
        self.text_SelectTrainingPicture = QLabel()
        self.button_ListImages=QPushButton('Show the whole List',default=False, autoDefault=False)
        
        self.button_SelectTrainingPictures = QPushButton('Look for some picture(s)', default=False, autoDefault=False)
        self.button_SelectTrainingPicturesDirectory= QPushButton('Look for a directory', default=False, autoDefault=False)
        self.clear_button = QPushButton('Clear', default=False, autoDefault=False)
        
        self.label_empty = QLabel() #just to help with the layout
        self.label_NumberOfClasses = QLabel('Choose the number of classes you want to use:')
        self.spinbox_numberOfClasses = QSpinBox()
        self.spinbox_numberOfClasses.setMinimum(2)
        self.spinbox_numberOfClasses.setMaximum(5)
        self.spinbox_numberOfClasses.setValue(2)
        self.button_NameClasses=QPushButton('Name the classes',default=False, autoDefault=False)
        self.label_NameClasses = QLabel('')
        
        self.cancel_button = QPushButton('Cancel', default=False, autoDefault=False)
        self.button_newTrainingDataStart = QPushButton('START',default=False, autoDefault=False)
        
        #Create the layout
        h_box1 = QHBoxLayout()
        h_box1.addWidget(self.button_SelectTrainingPicturesDirectory)
        h_box1.addWidget(self.button_SelectTrainingPictures)
        h_box1.addWidget(self.clear_button)
        
        h_boxbutton = QHBoxLayout()
        h_boxbutton.addWidget(self.cancel_button)
        h_boxbutton.addWidget(self.button_newTrainingDataStart)
        
        h_boxclasses = QHBoxLayout()
        h_boxclasses.addWidget(self.label_NumberOfClasses)
        h_boxclasses.addWidget(self.spinbox_numberOfClasses)
        h_boxclasses.addWidget(self.button_NameClasses)
        
        v_box = QVBoxLayout()
        v_box.addWidget(self.label_newTrainingData)
        v_box.addWidget(self.label_ModeTrainingPictures)
        v_box.addWidget(self.label_NumberPictures)
        v_box.addWidget(self.text_SelectTrainingPicture)
        v_box.addWidget(self.button_ListImages)
        v_box.addLayout(h_box1)
        v_box.addWidget(self.label_empty)
        v_box.addLayout(h_boxclasses)
        v_box.addWidget(self.label_NameClasses)
        v_box.addLayout(h_boxbutton)
        
        self.setLayout(v_box)
        
        self.show()
        
        self.button_ListImages.hide()
        
        # Link the buttons and spinbox to their corresponding functions
        self.button_ListImages.clicked.connect(self.ShowListImages)
        self.button_SelectTrainingPictures.clicked.connect(self.lookForTrainingPicture)
        self.button_SelectTrainingPicturesDirectory.clicked.connect(self.lookForTrainingPictureDirectoryFunction)
        self.clear_button.clicked.connect(self.clear_text)
        self.button_newTrainingDataStart.clicked.connect(self.CreateTrainingData)
        self.cancel_button.clicked.connect(self.cancel)
        self.button_NameClasses.clicked.connect(self.NameClassesFunction)
        self.spinbox_numberOfClasses.valueChanged.connect(self.spinboxChangedFunction)
        
        #Initialize the different variables
        self.newfilecreated='N'
        self.listPictureNames=[]
        self.classesNamesList=[]
        self.classes=self.spinbox_numberOfClasses.value()
        for i in range(self.classes):
            self.classesNamesList.append('Class_'+str(i+1))
            self.label_NameClasses.setText(self.label_NameClasses.text()+'Class '+str(i+1)+': '+self.classesNamesList[i]+'\n')
    
    def spinboxChangedFunction(self):
        """ This function is part of the class 'TrainingWindow'. \n
        It is linked to the change of value of the spinbox self.spinbox_numberOfClasses. \n
        When the value of the spinbox changes, the attribute self.classes changes according to the value of the spinbox and the label self.label_NameClasses changes. """
        self.label_NameClasses.setText('')
        self.classesNamesList=[]
        self.classes=self.spinbox_numberOfClasses.value()
        for i in range(self.classes):
            self.classesNamesList.append('Class_'+str(i+1))
            self.label_NameClasses.setText(self.label_NameClasses.text()+'Class '+str(i+1)+': '+self.classesNamesList[i]+'\n')
                
    def ShowListImages(self):
        """ This function is part of the class 'TrainingWindow'. \n
        It is linked to the click of the button 'Show the whole list' (self.button_ListImages)  \n
        When the button is clicked, the window 'ListNameWindow' is opened"""

        ListWin=ListNameWindow(parent=self, List=self.listPictureNames)
        ListWin.exec_()       
        
    def lookForTrainingPicture(self):
        """ This function is part of the class 'TrainingWindow'. \n
        It is linked to the click of the button 'Look for some picture(s)' (self.button_SelectTrainingPictures)  \n
        This function allow the user to choose one or several pictures which will be used to create the training data set."""

        self.button_ListImages.hide()
        self.text_SelectTrainingPicture.setText('')
        
        #Open a dialog window to choose the pictures
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.listPictureNames, _ = QFileDialog.getOpenFileNames(self,"Select the training picture(s)", "",".png or jpg (*.png *.jpg *.JPG *.jpeg)", options=options)
        
        
        if self.listPictureNames: #if the user has selected at least 1 file
            for i in range(len(self.listPictureNames)): #Only display the first 5 file addresses
                if i<5:
                    self.text_SelectTrainingPicture.setText(self.text_SelectTrainingPicture.text()+'\n'+str(self.listPictureNames[i]))
                if i==5 and len(self.listPictureNames)>5:
                    self.text_SelectTrainingPicture.setText(self.text_SelectTrainingPicture.text()+'\n ...')                
        
        if len(self.listPictureNames)>5: #If more than 5 pictures are selected, the button 'Show the whole List' (self.button_ListImages) appears and allow the user to see the whole list in another window.
            self.button_ListImages.show()
            
        self.label_ModeTrainingPictures.setText('Mode: Files')
        
        #Count the number of picture
        if len(self.listPictureNames)==0 or len(self.listPictureNames)==1:
            self.label_NumberPictures.setText('Path to the training pictures: ('+str(len(self.listPictureNames))+' picture)')
        else :
            self.label_NumberPictures.setText('Path to the training pictures: ('+str(len(self.listPictureNames))+' pictures)')
            
    def lookForTrainingPictureDirectoryFunction(self):
        """ This function is part of the class 'TrainingWindow'. \n
        It is linked to the click of the button 'Look for a directory' (self.button_SelectTrainingPicturesDirectory)  \n
        This function allow the user to choose one directory containing the pictures which will be used to create the training data set."""

        #Open a dialog window to choose the pictures
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        DirectoryName= str(QFileDialog.getExistingDirectory(self, "Select Directory with the training picture(s)"))
        
        if DirectoryName!='':           #if the user has selected a directory
            self.text_SelectTrainingPicture.setText(DirectoryName)
            self.label_ModeTrainingPictures.setText('Mode: Directory')
            
            #Save the list of the addresses of the pictures in the directory in self.listPictureNames
            ListImageinDirectory=os.listdir(DirectoryName)      
            ListImage=[]        
            for i in range(len(ListImageinDirectory)):
                a=ListImageinDirectory[i].split('.')
                if 'jpeg'in a or 'png'in a or 'jpg'in a or 'JPG' in a:   #only take the images
                    ListImage.append(DirectoryName+'/'+ListImageinDirectory[i]) 
            
            self.listPictureNames=ListImage
            
            #Count the number of picture
            if len(self.listPictureNames)==0 or len(self.listPictureNames)==1:
                self.label_NumberPictures.setText('Path to the directory: ('+str(len(self.listPictureNames))+' picture)')
            else :
                self.label_NumberPictures.setText('Path to the directory: ('+str(len(self.listPictureNames))+' pictures)')
            
            self.button_ListImages.show() #the button 'Show the whole List' (self.button_ListImages) appears and allow the user to see the list of the pictures in the directory in another window.

    
    def clear_text(self):
        """ This function is part of the class 'TrainingWindow'. \n
        It is linked to the click of the button 'Clear' (self.clear_button)  \n
        Erase the content of the label self.text_SelectTrainingPicture"""

        self.label_NumberPictures.setText('Path to the training pictures:')
        self.text_SelectTrainingPicture.clear()
        self.listPictureNames=[]
    
    def cancel(self):
        """ This function is part of the class 'TrainingWindow'. \n
        It is linked to the click of the button 'Cancel' (self.cancel_button)  \n
        Close the window without saving any file """
        self.newfilecreated='N'
        self.ListTrainingDataFile=[]
        self.hide()
    
    def NameClassesFunction(self):
        """ This function is part of the class 'TrainingWindow'. \n
        It is linked to the click of the button 'Name the classes' (self.button_NameClasses)  \n
        Open a new window from the class 'WindowNameClasses' and save the name chosen in this window"""
        self.classes=self.spinbox_numberOfClasses.value()
        #Open the window
        WindNames=WindowNameClasses(parent=self, classes=self.classes, classesNamesList=self.classesNamesList)
        WindNames.exec_()
        
        #Save the names in self.classesNamesList and display them inthe label self.label_NameClasses
        self.classesNamesList=WindNames.NamesList       
        self.label_NameClasses.setText('')
        for i in range(self.classes):
            self.label_NameClasses.setText(self.label_NameClasses.text()+'Class '+str(i+1)+': '+self.classesNamesList[i]+'\n')
        
    def CreateTrainingData(self):   
        """ This function is the main function of the class 'TrainingWindow'. \n
        It is linked to the click of the button 'START' (self.button_newTrainingDataStart)  \n
        Use the function 'SelectOneClass' of the class 'TrainingData' from the file FunctionToCreateTrainingData.py"""    
        
        #Error message if one of the parameter is missing
        if self.listPictureNames==[] or self.label_NumberPictures.text()=='Path to the directory: (0 picture)':
            messageWin=MessageWindow(parent=self, WorkingDirectory='OK', trainingData='OK', listPictureNames='OK', ROI='OK', text_InfoTestPictures='OK', trainingDataPicture=[], NamesList='Y', selectedpixels='Y',Window='N')
            messageWin.exec_() 
        
        else :   
            
            self.ListTrainingDataFile=[]
            
            # Create the directory where the output will be saved
            TrainingDataDirectory=self.workingDirectory+'/TrainingData'
            if not os.path.exists(TrainingDataDirectory):    
                os.mkdir(TrainingDataDirectory)

            colorList=[(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)] # 5 different colors, one for each class
            
            for i in range(self.classes):
                #Open a small window explaining what to do
                InfoWin=InfoWindowTrainindData(parent=None, Class=self.classesNamesList[i])
                InfoWin.exec_()
                #Call the function for each class
                t=TrainingData()
                ListFile=TrainingData.SelectOneClass(t,self.listPictureNames,TrainingDataDirectory,self.classesNamesList[i],colorList[i],self.displaySize)
                self.ListTrainingDataFile.extend(ListFile)
            
            #Error message if no pixels have been selected
            if self.ListTrainingDataFile==[]:
                messageWin=MessageWindow(parent=self, WorkingDirectory='OK', trainingData='OK', listPictureNames='OK', ROI='OK', text_InfoTestPictures='OK', trainingDataPicture='OK', NamesList='Y', selectedpixels='N', Window='N')
                messageWin.exec_() 
            
            #Save a file compiling all the individual training data files for easier use
            else:
                self.newfilecreated='Y'
                
                trainDataTab=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0]])
                for file in self.ListTrainingDataFile:    
                    f=open(file,"r",newline='') 
                    TrainData = list(csv.reader(f))
                    f.close()
                    TrainData.remove(['Class', 'Image', 'x','y','B','G','R','H','S','V','L','a','b'])
                    TrainData=np.asarray(TrainData)    
                    trainDataTab=np.concatenate((trainDataTab, TrainData), axis=0)
                trainDataTab=np.delete(trainDataTab, (0), axis=0)
                np.savetxt(TrainingDataDirectory+'/trainData_'+str(self.classes)+'classes.csv', trainDataTab, delimiter=",",header='Class,Image,x,y,B,G,R,H,S,V,L,a,b', comments='',fmt='%s')
                
                self.ListTrainingDataFile=[TrainingDataDirectory+'/trainData_'+str(self.classes)+'classes.csv']
                
                #Last message window to inform the user that all the pictures for all the class have been red.
                InfoWin2=YourDoneWindow(parent=None, Name=TrainingDataDirectory+'/trainData_'+str(self.classes)+'classes.csv')
                InfoWin2.exec_()
                
                #Close the training data window and go back to the main window
                self.hide()
    

        

class InfoWindowTrainindData(QDialog):
    """ This class contain the GUI for a message window. 
    This window will be opened from the class 'TrainingWindow' by the function red when the button 'START' (MainWindow.button_newTrainingDataStart) is pushed. \n
    This class has one parameter: 
        Class (string):
            Name of the class for which the user is going to select pixels
            (ex: Class='PaddyRice')
    """

    def __init__(self, parent=None, Class=None):
        super(InfoWindowTrainindData, self).__init__(parent)
        self.Class=Class
        self.init_InfoWindowTrainindData()

    def init_InfoWindowTrainindData(self):
        #Name the window
        self.setWindowTitle('Tutorial')
        
        #Create the widgets
        self.button_close = QPushButton('OK') 
        self.label_Info = QLabel('Select some pixels for the class:      '+ str(self.Class) +'\n Press "E" when you are done to go to the next picture \n Press "Q" if you want to start the picture again')
        
        #Create the layout
        v_box=QVBoxLayout()
        v_box.addWidget(self.label_Info)
        v_box.addWidget(self.button_close)
        
        self.setLayout(v_box)
        
        self.show()
        
        #connect the button to its function
        self.button_close.clicked.connect(self.close)
    
    def close(self):
        # Function to close this window
        self.hide()

class YourDoneWindow(QDialog):
    """ This class contain the GUI for a message window. 
    This window will be opened from the class 'TrainingWindow' by the function red when the button 'START' (MainWindow.button_newTrainingDataStart) is pushed. \n
    This class has one parameter: 
        Name (string):
            Address of the compiled training data file 
            (ex: Name='/Users/Name/Desktop/folder/TrainingData/trainData_2classes.csv')
    """

    def __init__(self, parent=None, Name=None):
        super(YourDoneWindow, self).__init__(parent)
        self.Name=Name
        self.init_InfoWindowTrainindData()

    def init_InfoWindowTrainindData(self):
        #Name the window
        self.setWindowTitle('Info')
        
        #Create the widgets
        self.button_close = QPushButton('OK') 
        self.label_Info = QLabel('You are done ! \n The training data file has been saved at the address: \n'+ str(self.Name))
       
        #Create the layout
        v_box=QVBoxLayout()
        v_box.addWidget(self.label_Info)
        v_box.addWidget(self.button_close)
        
        self.setLayout(v_box)
        
        self.show()
        
        #connect the button to its function
        self.button_close.clicked.connect(self.close)
    
    def close(self):
        # Function to close this window
        self.hide()
        
        
class WindowNameClasses(QDialog):
    """ This class contain the GUI for the window used to give a name to the classes. \n
    This window can be opened from class 'TrainingWindow' by clicking on the button 'Name the classes' (self.button_NameClasses) \n
    This class has two parameters: 
        classes (int):
            Number of classes 
            (ex: classes=2)
            
        classesNamesList (List of strings):
             List of the names of the classes 
             (ex: classesNamesList=['PaddyRice','Background'] )
    """

    def __init__(self, parent=None, classes=None, classesNamesList=None):
        super(WindowNameClasses, self).__init__(parent)
        self.classes=classes
        self.classesNamesList=classesNamesList
        self.init_WindowNameClasses()
        
    def init_WindowNameClasses(self):
        #Name the window
        self.setWindowTitle('Name the classes')
        
        #Initialize the lists
        self.NamesList=[]
        self.Line=[]
        self.InitNameList=[]  
        
        
        #Create the widgets
        self.label_Info = QLabel('Choose a name for each class (No space):')        
        for i in range(self.classes):
            label = QLabel('Class '+str(i+1))
            LineEdit= QLineEdit(self.classesNamesList[i])
            h_box=QHBoxLayout()
            h_box.addWidget(label)
            h_box.addWidget(LineEdit)
            self.Line.append(h_box)
            self.InitNameList.append(LineEdit)
       
        self.button_close = QPushButton('OK') 
        
        #Create the layout
        v_box=QVBoxLayout()
        v_box.addWidget(self.label_Info)
        for line in self.Line:
            v_box.addLayout(line)
        v_box.addWidget(self.button_close)
        self.setLayout(v_box)
        self.show()

        #connect the button to its function
        self.button_close.clicked.connect(self.close)
    
    def close(self):
        # Function to close this window
        self.NamesList=[]
        for i in range(self.classes):
            name=self.InitNameList[i].text()
            self.NamesList.append(name)
        
        #Error message if one classes has no name
        if '' in self.NamesList:
            messageWin=MessageWindow(parent=self, WorkingDirectory='OK', trainingData='OK', listPictureNames='OK', ROI='OK', text_InfoTestPictures='OK', trainingDataPicture='OK', NamesList='Class', selectedpixels='Y', Window='N')
            messageWin.exec_() 
        else:
            self.hide()

    
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
        
    
class DirectoryWindow(QDialog):
    """ This class contain the GUI for the window used to create a new directory. 
    This window can be opened from the class 'MainWindow' by clicking on the button 'Create a new directory' (MainWindow.button_CreateNewWorkingDirectory)
    """
    def __init__(self, parent=None):
        super(DirectoryWindow, self).__init__(parent)
        
        self.init_DirectoryWindow()
        
    def init_DirectoryWindow(self):
        #Name the window
        self.setWindowTitle('Create a new Directory')
        
        ####First Part : default layout
        #Create the widgets
        self.label_newWorkingDirectory = QLabel('Choose the directory where you want to Create the Directory and its name:')
        self.label_empty = QLabel() #just to help with the layout
        self.text_newWorkingDirectoryPath = QLabel('Directory/')
        self.button_newWorkingDirectoryPath = QPushButton('Choose the path', default=False, autoDefault=False) 
        self.clear_button = QPushButton('Clear', default=False, autoDefault=False)
       
        self.cancel_button = QPushButton('Cancel', default=False, autoDefault=False)
        
        self.text_newWorkingDirectoryName = QLineEdit()
        self.label_newWorkingDirectoryName = QLabel('Choose the name of the Directory')
        
        self.button_newWorkingDirectorySave = QPushButton('SAVE', default=False, autoDefault=False)
       
        #Create the layout
        v_boxlabel = QVBoxLayout()
        v_boxlabel.addWidget(self.label_newWorkingDirectory)
        v_boxlabel.addWidget(self.label_empty) 
         
        v_boxPath = QVBoxLayout()
        v_boxPath.addWidget(self.text_newWorkingDirectoryPath)
        v_boxPath.addWidget(self.button_newWorkingDirectoryPath)
        
        v_boxName = QVBoxLayout()
        v_boxName.addWidget(self.text_newWorkingDirectoryName)
        v_boxName.addWidget(self.label_newWorkingDirectoryName) 
        
        h_box = QHBoxLayout()
        h_box.addLayout(v_boxlabel)
        h_box.addLayout(v_boxPath)
        h_box.addLayout(v_boxName)
        
        h_boxButton=QHBoxLayout()
        h_boxButton.addWidget(self.cancel_button)
        h_boxButton.addWidget(self.clear_button)
        h_boxButton.addWidget(self.button_newWorkingDirectorySave)
                              
        v_box = QVBoxLayout()
        v_box.addLayout(h_box)
        v_box.addLayout(h_boxButton)
    
        
        self.windowlayout=v_box
        #connect the button to its function
        self.clear_button.clicked.connect(self.clear_text)
        self.button_newWorkingDirectoryPath.clicked.connect(self.ChooseWorkingDirectoryPathFunction)
        self.button_newWorkingDirectorySave.clicked.connect(self.CreateNewWorkingDirectoryFunction)
        self.cancel_button.clicked.connect(self.cancel)      
        
        
        ####Second Part : layout with warnings
        #Create the widgets
        self.label_NoPath = QLabel('Please enter the path of the Directory')
        self.label_Confimation = QLabel('')
        self.button_ConfirmationYes = QPushButton('Yes')
        self.button_ConfirmationNo = QPushButton('No', default=False, autoDefault=False) 
        
        #Create the layout
        self.h_boxConfirmation=QHBoxLayout()
        self.h_boxConfirmation.addWidget(self.label_NoPath)
        self.h_boxConfirmation.addWidget(self.label_Confimation)
        self.h_boxConfirmation.addWidget(self.button_ConfirmationYes)
        self.h_boxConfirmation.addWidget(self.button_ConfirmationNo)
        
        self.windowlayout.addLayout(self.h_boxConfirmation)
        
        #connect the buttons to their function
        self.button_ConfirmationYes.clicked.connect(self.button_ConfirmationYesFunction)
        self.button_ConfirmationNo.clicked.connect(self.button_ConfirmationNoFunction)

        #Hide the warnings as long as they are not needed
        self.label_NoPath.hide()
        self.label_Confimation.hide()
        self.button_ConfirmationYes.hide()
        self.button_ConfirmationNo.hide()
        
        self.label_ChooseAnotherName = QLabel('Please choose another name')
        self.windowlayout.addWidget(self.label_ChooseAnotherName)
        self.label_ChooseAnotherName.hide()
        
        #Set and show the layout in the window
        self.setLayout(self.windowlayout)
        
        self.show()
        
        
    def ChooseWorkingDirectoryPathFunction(self):
        """ This function is part of the class 'DirectoryWindow'. \n
        It is linked to the click of the button 'Choose the path' (self.button_newWorkingDirectoryPath)  \n
        This function allow the user to choose one directory where the new directory will be saved."""
        
        #Open a dialog window to choose the directory
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        DirectoryName= str(QFileDialog.getExistingDirectory(self, "Select a directory"))
        
        #Print the address in the label self.text_newWorkingDirectoryPath
        self.text_newWorkingDirectoryPath.setText(DirectoryName)
        
    def CreateNewWorkingDirectoryFunction(self):
        """ This function is part of the class 'DirectoryWindow'. \n
        It is linked to the click of the button 'SAVE' (self.button_newWorkingDirectorySave)  \n
        This function Creates a new directory at the chosen address."""
        
        NewDirectoryName=self.text_newWorkingDirectoryPath.text()+'/'+ self.text_newWorkingDirectoryName.text()
        
        #Error message if no address or name have been chosen
        if self.text_newWorkingDirectoryPath.text()=='Directory/' or self.text_newWorkingDirectoryPath.text()=='':
            self.label_NoPath.show()    
        
        else : 
            #Create the directory if it doesn't already exist
            self.label_NoPath.hide()
            if not os.path.exists(NewDirectoryName):    
                os.mkdir(NewDirectoryName)
                self.hide()
            #If it already exists, warning message appears    
            if os.path.exists(NewDirectoryName):
                        self.label_Confimation.show()
                        self.label_Confimation.setText('The Directory '+ self.text_newWorkingDirectoryPath.text()+'/'+ self.text_newWorkingDirectoryName.text()+' already exist. Would you like to use it ?')
                        self.button_ConfirmationYes.show()
                        self.button_ConfirmationNo.show()

    
    def button_ConfirmationYesFunction(self):
        """ This function is part of the class 'DirectoryWindow'. \n
        It is linked to the click of the button 'Yes' (self.button_ConfirmationYes)  \n
        This function keep the chosen address and close the window."""

        self.hide()
    
    def button_ConfirmationNoFunction(self):
        """ This function is part of the class 'DirectoryWindow'. \n
        It is linked to the click of the button 'No' (self.button_ConfirmationNo)  \n
        This function clear the chosen address and display a message asking to choose another name."""
        self.label_Confimation.hide()
        self.button_ConfirmationYes.hide()
        self.button_ConfirmationNo.hide()
        self.label_ChooseAnotherName.show()
        self.text_newWorkingDirectoryName.clear() 
        
    def clear_text(self):
        """ This function is part of the class 'DirectoryWindow'. \n
        It is linked to the click of the button 'Clear' (self.clear_button)  \n
        This function clear the chosen address """
        self.text_newWorkingDirectoryPath.clear()
        self.text_newWorkingDirectoryName.clear()
    
    def cancel(self):
        """ This function is part of the class 'DirectoryWindow'. \n
        It is linked to the click of the button 'Cancel' (self.cancel_button)  \n
        Close the window without saving any address """
        self.text_newWorkingDirectoryPath.clear()
        self.text_newWorkingDirectoryName.clear()
        self.hide()
        
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
   
     
class RegionOfInterestWindow(QDialog):
    """ This class contain the GUI for the window used to choose the regions of interest (ROI). \n
    This window can be opened from the class 'MainWindow' by clicking on the button 'Select one or several ROI' (MainWindow.button_RegionOfInterest) \n
    This class has three parameters: 
        WorkingDirectory (string): 
            Address of the working directory
            (ex: WorkingDirectory='/Users/Name/Desktop/folder')
            
        ListImageName (List of strings):
            List of the addresses of all the pictures that the user wants to process.  
            (ex: ListImageName=['/Users/Name/Desktop/image.png','/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png'])

        DisplaySize (int):
            Integer use to calculate the resizing coefficient for the display of the pictures
            (ex: displaySize=500)
    """

    def __init__(self, parent=None, workingDirectory=None, ImageNames=None, DisplaySize=None):
        super(RegionOfInterestWindow, self).__init__(parent)
        
        self.displaySize=DisplaySize
        self.ImageNames = ImageNames
        if '.DS_Store' in self.ImageNames:
            self.ImageNames.remove('.DS_Store')
        self.workingDirectory=workingDirectory
        self.init_RegionOfInterestWindow()
        
    def init_RegionOfInterestWindow(self):
        #Name the window
        self.setWindowTitle('Select one or several Regions of interest (ROI)')
        
        # Create the widgets
        self.label_Info = QLabel('Select the Region(s) of Interest and choose if you want to save the cropped pictures.  \n Note: saving the cropped pictures might take a lot of space and time if you are using \n big and/or many pictures')
        self.label_NbROI = QLabel('Coordinates of the selected region(s): ')
        
        
        self.button_start=QPushButton('Select the area(s)', default=False, autoDefault=False) 
        
        self.button_center=QPushButton('Use the center', default=False, autoDefault=False) 
        self.label_center=QLabel('Pourcentage of the image:')
        self.spinbox_center = QSpinBox()
        self.spinbox_center.setMinimum(5)
        self.spinbox_center.setMaximum(100)
        self.spinbox_center.setSingleStep(5)
        self.spinbox_center.setValue(50)
        self.spinbox_center.hide()
        self.label_center.hide()
        h_boxCenter=QHBoxLayout()
        h_boxCenter.addWidget(self.label_center)
        h_boxCenter.addWidget(self.spinbox_center)
                
        
        self.button_random=QPushButton('Choose random rectangle', default=False, autoDefault=False) 
        self.spinbox_nbRandom = QSpinBox()
        self.spinbox_nbRandom.setMinimum(1)
        self.spinbox_nbRandom.setMaximum(100)
        self.spinbox_nbRandom.setSingleStep(1)
        self.spinbox_nbRandom.setValue(1)
        self.spinbox_nbRandom.hide()
        self.label_nbRandom=QLabel('Number of random rectangles:')
        h_boxnbRandom=QHBoxLayout()
        h_boxnbRandom.addWidget(self.label_nbRandom)
        h_boxnbRandom.addWidget(self.spinbox_nbRandom)
        self.label_nbRandom.hide()
        self.spinbox_nbRandom.hide()
        
        img1 = cv2.imread(self.ImageNames[0])
        self.spinbox_sizeRandomH = QSpinBox()
        self.spinbox_sizeRandomH.setMinimum(1)
        self.spinbox_sizeRandomH.setMaximum(len(img1))
        self.spinbox_sizeRandomH.setSingleStep(5)
        self.spinbox_sizeRandomH.setValue(10)
        self.label_sizeRandomH=QLabel('Hight of the random rectangles:')
        self.label_pixelsH=QLabel('pixels')
        h_boxSizeRandomH=QHBoxLayout()
        h_boxSizeRandomH.addWidget(self.label_sizeRandomH)
        h_boxSizeRandomH.addWidget(self.spinbox_sizeRandomH)
        h_boxSizeRandomH.addWidget(self.label_pixelsH)
        
        self.label_sizeRandomH.hide()
        self.spinbox_sizeRandomH.hide()
        self.label_pixelsH.hide()
        
        
        self.spinbox_sizeRandomW = QSpinBox()
        self.spinbox_sizeRandomW.setMinimum(1)
        self.spinbox_sizeRandomW.setMaximum(len(img1[0]))
        self.spinbox_sizeRandomW.setSingleStep(5)
        self.spinbox_sizeRandomW.setValue(10)
        self.label_sizeRandomW=QLabel('Width of the random rectangles:')
        self.label_pixelsW=QLabel('pixels')
        h_boxSizeRandomW=QHBoxLayout()
        h_boxSizeRandomW.addWidget(self.label_sizeRandomW)
        h_boxSizeRandomW.addWidget(self.spinbox_sizeRandomW)
        h_boxSizeRandomW.addWidget(self.label_pixelsW)
        self.label_sizeRandomW.hide()
        self.spinbox_sizeRandomW.hide()
        self.label_pixelsW.hide()
        
        
        self.CheckBox_Randomsize = QCheckBox('Choose a random size for the ROI')
        self.CheckBox_Randomsize.setChecked(True)
        self.CheckBox_Randomsize.hide()
        
        self.CheckBox_RandomNumber = QCheckBox('Choose a random number of ROI')
        self.CheckBox_RandomNumber.setChecked(True)
        self.CheckBox_RandomNumber.hide()
        
        self.button_OK=QPushButton('OK')
        self.button_cancel=QPushButton('Cancel', default=False, autoDefault=False)
        self.button_clear=QPushButton('Clear', default=False, autoDefault=False)
        self.text_rectangle=QLabel()
        self.CheckBox_SaveCroppedPictures = QCheckBox('Save the cropped pictures')
        self.CheckBox_UseAsAMask = QCheckBox('Use the rectangle(s) as a mask')
        self.CheckBox_UseAsAMask.setChecked(True)
        
        self.button_NameROI=QPushButton('Name/View the areas', default=False, autoDefault=False) 
        
        #Create the layout

        h_boxButton1=QHBoxLayout()
        h_boxButton1.addWidget(self.button_cancel)
        h_boxButton1.addWidget(self.button_OK)
        
        v_boxCenter=QVBoxLayout()
        v_boxCenter.addLayout(h_boxCenter)
        v_boxCenter.addWidget(self.button_center)
        
        v_boxRandom=QVBoxLayout()
        v_boxRandom.addWidget(self.CheckBox_RandomNumber)
        v_boxRandom.addLayout(h_boxnbRandom)
        v_boxRandom.addWidget(self.CheckBox_Randomsize)
        v_boxRandom.addLayout(h_boxSizeRandomH)
        v_boxRandom.addLayout(h_boxSizeRandomW)
        v_boxRandom.addWidget(self.button_random)        
        
        h_boxButton2=QHBoxLayout()
        h_boxButton2.addWidget(self.button_start)
        h_boxButton2.addLayout(v_boxCenter)
        h_boxButton2.addLayout(v_boxRandom)

        h_boxClear=QHBoxLayout()
        h_boxClear.addWidget(self.button_clear)
        h_boxClear.addWidget(self.button_NameROI)

        h_boxCheckBox=QHBoxLayout()
        h_boxCheckBox.addWidget(self.CheckBox_SaveCroppedPictures)
        h_boxCheckBox.addWidget(self.CheckBox_UseAsAMask)        
        
        v_box=QVBoxLayout()
        v_box.addWidget(self.label_Info)
        v_box.addLayout(h_boxButton2)
        v_box.addWidget(self.label_NbROI)
        v_box.addWidget(self.text_rectangle)
        v_box.addLayout(h_boxClear)
        v_box.addLayout(h_boxCheckBox)
        v_box.addLayout(h_boxButton1)
        
        self.setLayout(v_box)
        
        self.show()
        
        #connect the buttons to their function

        self.CheckBox_SaveCroppedPictures.clicked.connect(self.uncheckUseAsAMask)
        self.CheckBox_UseAsAMask.clicked.connect(self.uncheckSaveCroppedPictures)
        self.button_cancel.clicked.connect(self.cancel)
        self.button_clear.clicked.connect(self.clear_text)
        self.button_start.clicked.connect(self.SelectArea)
        self.button_OK.clicked.connect(self.Validation)
        self.button_NameROI.clicked.connect(self.NameArea)
        self.button_center.clicked.connect(self.centerFunction)
        self.button_random.clicked.connect(self.randomFunction)
        self.CheckBox_Randomsize.clicked.connect(self.CheckRandomsizeFunction)
        self.CheckBox_RandomNumber.clicked.connect(self.CheckRandomNumberFunction)
        
        self.AreaNamesList=[] 
        
        self.coordinaterectangle=[]

            
    def centerFunction(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the button 'Use the center' (self.button_center)  \n
        This function calculates the coordinates of a rectangle in the center of the picture whose area is equal to a chosen pourcentage of the total area. """

        self.coordinaterectangle=[]
        
        self.spinbox_center.show()
        self.label_center.show()
        
        #Read the image and calculate the coordinates of a rectangle in the center.
        img1 = cv2.imread(self.ImageNames[0])
        a=((100-self.spinbox_center.value())/2)/100
        x1=int(len(img1[0])*a)
        y1=int(len(img1)*a)
        x2=int(len(img1[0])*(1-a))
        y2=int(len(img1)*(1-a))
    
        self.coordinaterectangle=[[x1,y1,x2,y2]]
        self.text_rectangle.setText(str(self.coordinaterectangle))
        self.AreaNamesList=['Center'+str(self.spinbox_center.value())]
        
        #Hide the not-needed widgets
        self.label_sizeRandomW.hide()
        self.spinbox_sizeRandomW.hide()
        self.label_pixelsW.hide()
        self.label_sizeRandomH.hide()
        self.spinbox_sizeRandomH.hide()
        self.label_pixelsH.hide()
        self.label_nbRandom.hide()
        self.spinbox_nbRandom.hide()        
        self.CheckBox_Randomsize.hide()
        self.CheckBox_RandomNumber.hide()
        
        self.CheckBox_RandomNumber.setChecked(True)
        self.CheckBox_Randomsize.setChecked(True)
        
        nbROI=1
        self.label_NbROI.setText('Coordinates of the selected region: ('+str(nbROI)+' region)')
        self.RefImg=self.ImageNames[0]
        
    
    def randomFunction(self):   
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the button 'Choose random rectangle' (self.button_random)  \n
        This function calculates random coordinates of a random or chosen number rectangle(s) with chosen or random dimensions """

        self.coordinaterectangle=[]
        self.CheckBox_RandomNumber.show()
        self.CheckBox_Randomsize.show()
        self.spinbox_center.hide()
        self.label_center.hide()
        
        img1 = cv2.imread(self.ImageNames[0])
        
        if self.CheckBox_Randomsize.isChecked():
            W=randint(1, len(img1[0])-1)
            self.spinbox_sizeRandomW.setValue(W)
            H=randint(1, len(img1)-1)
            self.spinbox_sizeRandomH.setValue(H)
        
        if self.CheckBox_RandomNumber.isChecked():
            areaBox=self.spinbox_sizeRandomW.value()*self.spinbox_sizeRandomH.value()
            areaImage=len(img1[0])*len(img1)
            maxBox=int(areaImage/areaBox)
            if maxBox>20:
                n=randint(0, 20)
            else:
                n=randint(0, maxBox)
            self.spinbox_nbRandom.setValue(n)
        
        H=self.spinbox_sizeRandomH.value()
        W=self.spinbox_sizeRandomW.value()
        n=self.spinbox_nbRandom.value()
        
        for i in range(n):
            x=randint(0, len(img1[0]))
            y=randint(0, len(img1))
            x1=int(x-(W/2))
            y1=int(y-(H/2))
            x2=int(x+(W/2))
            y2=int(y+(H/2))
            if x1<=0:
                x1=int(1)
                x2=x1+int(W)
            if y1<=0:
                y1=int(1)
                y2=y1+int(H)
    
            if x2>=len(img1[0]):
                x2=int(len(img1[0])-1)
                x1=int(x2-W)
            if y2>=len(img1):
                y2=int(len(img1)-1)
                y1=int(y2-H)
        
            self.coordinaterectangle.append([x1,y1,x2,y2])
            
        if len(self.coordinaterectangle)>5:
            a=0
            string=''
            for i in range(int(len(self.coordinaterectangle)/5)):
                string=string+str(self.coordinaterectangle[a:a+5])+'\n'
                a=a+5
            self.text_rectangle.setText(string)
        else: 
            self.text_rectangle.setText(str(self.coordinaterectangle))
        
        self.AreaNamesList=[]
        for i in range(len(self.coordinaterectangle)):
            self.AreaNamesList.append('Area_'+str(i+1))     
        
        nbROI=len(self.coordinaterectangle)
        if nbROI==0 or nbROI==1:
            self.label_NbROI.setText('Coordinates of the selected region: ('+str(nbROI)+' region)')
        else:
            self.label_NbROI.setText('Coordinates of the selected region: ('+str(nbROI)+' regions)')
        
        self.RefImg=self.ImageNames[0]
            
    def CheckRandomsizeFunction(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the check box 'Choose a random size for the ROI' (self.CheckBox_Randomsize)  \n
        This function make other widgets appear or disappear """

        if self.CheckBox_Randomsize.isChecked():
            self.label_sizeRandomH.hide()
            self.spinbox_sizeRandomH.hide()
            self.label_pixelsH.hide()
            self.label_sizeRandomW.hide()
            self.spinbox_sizeRandomW.hide()
            self.label_pixelsW.hide()
        else:
            self.label_sizeRandomH.show()
            self.spinbox_sizeRandomH.show()
            self.label_pixelsH.show()
            self.label_sizeRandomW.show()
            self.spinbox_sizeRandomW.show()
            self.label_pixelsW.show()
            
    def CheckRandomNumberFunction(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the check box 'Choose a random number of ROI' (self.CheckBox_RandomNumber)  \n
        This function make other widgets appear or disappear """
        if self.CheckBox_RandomNumber.isChecked():
            self.label_nbRandom.hide()
            self.spinbox_nbRandom.hide()
        else:
            self.label_nbRandom.show()
            self.spinbox_nbRandom.show()
        
    def uncheckSaveCroppedPictures(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the check box 'Use the rectangle(s) as a mask' (self.CheckBox_UseAsAMask)  \n
        Make sure that self.CheckBox_UseAsAMask and self.CheckBox_SaveCroppedPictures are not checked at the same time """
        self.CheckBox_SaveCroppedPictures.setChecked(False)
    
    def uncheckUseAsAMask(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the check box 'Save the cropped pictures' (self.CheckBox_SaveCroppedPictures)  \n
        Make sure that self.CheckBox_UseAsAMask and self.CheckBox_SaveCroppedPictures are not checked at the same time """
        self.CheckBox_UseAsAMask.setChecked(False)
    
    def clear_text(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the check box 'Clear' (self.button_clear)  \n
        Clear the list of coordinates """
        self.text_rectangle.clear()
    
    def cancel(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the check box 'Cancel' (self.button_cancel)  \n
        Close the window without saving any coordinates """
        self.text_rectangle.setText('')
        self.coordinaterectangle=[]
        self.hide()
    
        
    def SelectArea(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the button 'Select the area(s)' (self.button_start)  \n
        Open the SelectAreaWindow """
        
        #Hide the not-needed widgets
        self.label_sizeRandomW.hide()
        self.spinbox_sizeRandomW.hide()
        self.label_pixelsW.hide()
        self.label_sizeRandomH.hide()
        self.spinbox_sizeRandomH.hide()
        self.label_pixelsH.hide()
        self.label_nbRandom.hide()
        self.spinbox_nbRandom.hide()
        self.CheckBox_Randomsize.hide()
        self.CheckBox_RandomNumber.hide()
        self.CheckBox_RandomNumber.setChecked(True)
        self.CheckBox_Randomsize.setChecked(True)
        self.spinbox_center.hide()
        self.label_center.hide()
        
        #Open the SelectAreaWindow
        messageWin=SelectAreaWindow(parent=None, ImageNames=self.ImageNames, DisplaySize=self.displaySize)
        messageWin.exec_()
        
        #Get the selected coordinate from SelectAreaWindow
        self.RefImg=messageWin.Img
        self.coordinaterectangle=messageWin.coordinaterectangle

        #Display the list of coordinates with maximum 5 ROI per line 
        if len(self.coordinaterectangle)>5:
            a=0
            string=''
            for i in range(int(len(self.coordinaterectangle)/5)):
                string=string+str(self.coordinaterectangle[a:a+5])+'\n'
                a=a+5
            self.text_rectangle.setText(string)
        else: 
            self.text_rectangle.setText(str(self.coordinaterectangle))
        
        #Create default names for the ROI
        self.AreaNamesList=[]
        for i in range(len(self.coordinaterectangle)):
            self.AreaNamesList.append('Area_'+str(i+1))
        
        #Count the number of ROI
        nbROI=len(self.coordinaterectangle)
        if nbROI==0 or nbROI==1:
            self.label_NbROI.setText('Coordinates of the selected region: ('+str(nbROI)+' region)')
        else:
            self.label_NbROI.setText('Coordinates of the selected region: ('+str(nbROI)+' regions)')
     
              
    def NameArea(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the check box 'Name/View the areas' (self.button_NameROI)  \n
        Open the WindowNameROI to allow the user to choose names for the ROI """
        
        if self.coordinaterectangle!=[]:
            ImageName=self.ImageNames[0]           
            WindNames=WindowNameROI(parent=self, coordinates=self.coordinaterectangle, ImageName=ImageName, AreaNamesList=self.AreaNamesList)
            WindNames.exec_()
            self.AreaNamesList=WindNames.NamesList
        
        #Error message if no ROI has been chosen
        else:
            messageWin=MessageWindow(parent=self, WorkingDirectory='OK', trainingData='OK', listPictureNames='OK', ROI='', text_InfoTestPictures='OK', trainingDataPicture='OK', NamesList='Y', selectedpixels='Y', Window='N')
            messageWin.exec_() 
    
    def Validation(self):
        """ This function is part of the class 'RegionOfInterestWindow'. \n
        It is linked to the click of the check box 'OK' (self.button_OK)  \n
        Save the coordinates and the cropped pictures if needed """

        coordinateList=str(self.text_rectangle.text())
        
        ###Error message if no ROI has been chosen
        if coordinateList=='':
            messageWin=MessageWindow(parent=self, WorkingDirectory='OK', trainingData='OK', listPictureNames='OK', ROI='', text_InfoTestPictures='OK', trainingDataPicture='OK', NamesList='Y', selectedpixels='Y', Window='N')
            messageWin.exec_() 
        
        ### Save informations about the ROI
        else :   
            
            ImgName=self.ImageNames[0]
            
            # Create a directory where the ROI informations will be saved
            ROIDirectory=self.workingDirectory+'/RegionsOfInterest'
            if not os.path.exists(ROIDirectory):  # Create a directory
                os.mkdir(ROIDirectory)
            
            #Save the reference in low resolution with red squares corresponding to the ROI as a reminder
            for i in range (len(self.coordinaterectangle)):  
                img1 = cv2.imread(ImgName) 
                x1,y1,x2,y2=self.coordinaterectangle[i]
                b=400
                a=b/len(img1[:,0])
                ResizedImage = cv2.resize(img1, (int(a*len(img1[0,:])), b))
                cv2.rectangle(ResizedImage, (int(x1*a),int(y1*a)), (int(x2*a),int(y2*a)), (0,0,255), 3) 
                AreaName=self.AreaNamesList[i]
                name=ROIDirectory+'/'+AreaName+'.png'
                cv2.imwrite(name,ResizedImage)
            
            #Save a text file with the name of the reference picture, the name and coordinates of the ROI
            img1 = cv2.imread(ImgName)
            dimRefImg=np.shape(img1)

            ListSaveROI=[[ImgName,str(dimRefImg),'','','']]
            for i in range(len(self.coordinaterectangle)):
                ListSaveROI.append([self.AreaNamesList[i],self.coordinaterectangle[i][0],self.coordinaterectangle[i][1],self.coordinaterectangle[i][2],self.coordinaterectangle[i][3]])
                
            np.savetxt(ROIDirectory+'/Coordinates_ROI.txt', ListSaveROI, delimiter=" ", comments='',fmt='%s')
            
            ## If the user doesn't want to save the cropped pictures, close the window now
            if self.CheckBox_UseAsAMask.isChecked():
                self.hide()
            
            
            ## Save the cropped pictures
            else : 
                # Error message if no working directory as been chosen
                if self.workingDirectory=='':
                    messageWin=MessageWindow(parent=self, WorkingDirectory='', trainingData='OK', listPictureNames='OK', ROI='OK', text_InfoTestPictures='OK', trainingDataPicture='OK', NamesList='Y', selectedpixels='Y', Window='N')
                    messageWin.exec_() 
                
                else:
                    #Create a directory where the cropped pictures will be saved
                    croppedPicturesDirectory=self.workingDirectory+'/croppedPictures'
                    if not os.path.exists(croppedPicturesDirectory):    
                        os.mkdir(croppedPicturesDirectory)
                        
                    #Create a sub-directoryies, one per ROI  
                    for n in range(len(self.coordinaterectangle)):
                        AreaName=self.AreaNamesList[n]
                        croppedPicturesDirectoryArea=croppedPicturesDirectory+'/'+AreaName
                        if not os.path.exists(croppedPicturesDirectoryArea):    
                            os.mkdir(croppedPicturesDirectoryArea)
                    
                    # Crop and save the pictures
                    for imgName in self.ImageNames:
                        img=cv2.imread(imgName)
                        ImageNamebis=imgName.split('/')
                        ImageNamebis=ImageNamebis[-1] 
                        ImageNamebis=ImageNamebis.split('.')
                        ImageNamebis=ImageNamebis[0] 
                        
                        for j in range(len(self.coordinaterectangle)):
                            AreaName=self.AreaNamesList[j]
                            
                            x1,y1,x2,y2=self.coordinaterectangle[j]
                            croppedImagej=img[y1:y2,x1:x2]
                            name=self.workingDirectory+'/croppedPictures/'+AreaName+'/'+ImageNamebis+'_crop_'+AreaName+'.png'
                            cv2.imwrite(name,croppedImagej)
                    
                    #Close the window
                    self.hide()
                
class WindowNameROI(QDialog):
    """ This class contain the GUI for the window used to choose the names regions of interest (ROI). \n
    This window can be opened from the class 'RegionOfInterestWindow' by clicking on the button 'Name/View the areas' (RegionOfInterestWindow.button_NameROI) \n
    This class has three parameters: 
        coordinates (List of lists of int):
            List of the coordinates of each ROI. For each ROI, the first two numbers are the coordinates of the top left corner and the other two are the coordinates of the bottom right corner.
            (ex: coordinates=[[0,0,50,50],[50,0,100,50],[0,50,50,100]] )  
            
        ImageName (string):
            Reference image address
            (ex: ImageName='/Users/Name/Desktop/image.png')

        AreaNamesList (List of strings):
             List of the names of the areas in the same order as the list ROI 
             (ex: AreaNamesList=['Area_1','Area_2','Area_3'] )    
        
        This Window allow the user to change the names of the ROI
        """

    def __init__(self, parent=None, coordinates=None, ImageName=None, AreaNamesList=None):
        super(WindowNameROI, self).__init__(parent)
        self.coordinates=coordinates
        self.ImageName=ImageName
        self.AreaNamesList=AreaNamesList
        self.init_WindowNameROI()
        
    def init_WindowNameROI(self):
        #Name the window
        self.setWindowTitle('Name the region of interest')
        
        #Initialize the list of the new names
        self.NamesList=[]
        
        #Create the widgets
        self.label_Info = QLabel('Choose a name for each region of interest (No space): \n To exit the "View" mode, press any key')

        self.Line=[]
        self.InitNameList=[]
        self.buttonList=[]
        for i in range(len(self.coordinates)):
            #Create and connects the 'View' button. The number of button depends on the number of ROI
            label = QLabel('Region '+str(i+1))
            LineEdit= QLineEdit(self.AreaNamesList[i])
            button = QPushButton('View '+str(i+1), default=False, autoDefault=False) 
            button.clicked.connect(self.viewPicture)
            h_box=QHBoxLayout()
            h_box.addWidget(label)
            h_box.addWidget(LineEdit)
            h_box.addWidget(button)
            self.Line.append(h_box)
            self.InitNameList.append(LineEdit)
            self.buttonList.append(button)
        
        self.button_close = QPushButton('OK') 
        
        #Create the layout
        v_box=QVBoxLayout()
        v_box.addWidget(self.label_Info)
        for line in self.Line:
            v_box.addLayout(line)
        v_box.addWidget(self.button_close)
        self.setLayout(v_box)
        self.show()
        
        #Connect the 'OK' button to its function
        self.button_close.clicked.connect(self.close)
    
    def viewPicture(self):
        """ This function is part of the class 'WindowNameROI'. \n
        It is linked to the click of the buttons 'View_number' from the list self.buttonList  \n
        Save the coordinates and the cropped pictures if needed """
        button = self.sender() #get the text of the button linked to the function
        #This text is 'View_'+the number of the ROI so we can extract the number whith the next lines :
        n=button.text()
        n=n.split(' ')
        n=n[1]
        n=int(n)
        
        img1 = cv2.imread(self.ImageName) #The image is opened 
        #Resize the image 
        b=500
        a=b/len(img1[:,0])
        img = cv2.resize(img1, (int(a*len(img1[0,:])), b))   
        
        #Get the coordinates and draw the rectangle on the resized picture
        x1,y1,x2,y2=self.coordinates[n-1]
        x1=int(x1*a)
        x2=int(x2*a)
        y1=int(y1*a)
        y2=int(y2*a)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3) 
                     
        #Display the picture
        cv2.namedWindow('Area_'+str(n),cv2.WINDOW_NORMAL) #define the name of the window
        cv2.imshow('Area_'+str(n),img)
        
        cv2.waitKey(0) # Press any keyboard to read the next line and close the window where the picture is displayed
        cv2.destroyAllWindows() # There is a problem with spyder and destroyAllWindows doesn't work alone>These fiew lines work for some reason
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)

    
    def close(self):
        """ This function is part of the class 'WindowNameROI'. \n
        It is linked to the click of the button 'OK' (self.button_close)  \n
        Save the names and close the window """
        # Create the list
        self.NamesList=[]
        for i in range(len(self.coordinates)):
            name=self.InitNameList[i].text()
            self.NamesList.append(name)
        
        # if one of the area doesn't have any name, an error message is raised
        if '' in self.NamesList:
            messageWin=MessageWindow(parent=self, WorkingDirectory='OK', trainingData='OK', listPictureNames='OK', ROI='OK', text_InfoTestPictures='OK', trainingDataPicture='OK', NamesList='Region of Interest', selectedpixels='Y', Window='N')
            messageWin.exec_() 
        else:
            #Else, close the window
            self.hide()


class SelectAreaWindow(QDialog):
    """ This class contain the GUI for the window used to start the selection of the regions of interest (ROI). \n
    This window can be opened from the class 'RegionOfInterestWindow' by clicking on the button 'Select the area(s)' (RegionOfInterestWindow.button_start)  \n
    This class has two parameters: 
        ImageNames (List of strings):
            List of the addresses of all the pictures that the user wants to process.  
            (ex: ImageNames=['/Users/Name/Desktop/image.png','/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png'])

        DisplaySize (int):
            Integer use to calculate the resizing coefficient for the display of the pictures
            (ex: DisplaySize=500)  
    
    """

    def __init__(self, parent=None, ImageNames=None, DisplaySize=None):
        super(SelectAreaWindow, self).__init__(parent)
        self.displaySize=DisplaySize
        self.Img=''
        self.ImageNames=ImageNames
        self.init_SelectAreaWindow()
        
    def init_SelectAreaWindow(self):
        # Name the window
        self.setWindowTitle('Select the region(s) of interest')

        #Create the widgets
        self.label_Info = QLabel('Draw a first rectangle of the size you want around the first region of interest. \n Then just click on the center of the next region of interest and a rectangle of the same size as the first one will be drawn automatically. \n When you are done press "E" to close the window')
        
        self.label_sameRectangle= QLabel('Would you like to use regions all of the same size ?')
        self.CheckBox_SameSize = QCheckBox('Yes')
        self.CheckBox_SameSize.setChecked(True)
        
        self.label_textPicture = QLabel('Picture use to select the ROI(s):' )
        
        self.label_Picture = QLabel(str(self.ImageNames[0]))
        self.button_choosePicture=QPushButton('Choose another picture', default=False, autoDefault=False) 
        self.button_SelectArea=QPushButton('START', default=False, autoDefault=False) 
        self.button_cancel = QPushButton('Cancel',default=False, autoDefault=False) 
        
        #Create the layout
        h_boxSize=QHBoxLayout()
        h_boxSize.addWidget(self.label_sameRectangle)
        h_boxSize.addWidget(self.CheckBox_SameSize)  
        
        h_boxPicture=QHBoxLayout()
        h_boxPicture.addWidget(self.label_textPicture)
        h_boxPicture.addWidget(self.label_Picture)  
        h_boxPicture.addWidget(self.button_choosePicture)
        
        h_boxbutton=QHBoxLayout()
        h_boxbutton.addWidget(self.button_cancel)
        h_boxbutton.addWidget(self.button_SelectArea)
        
        self.v_box=QVBoxLayout()
        self.v_box.addWidget(self.label_Info)
        self.v_box.addLayout(h_boxPicture)
        self.v_box.addLayout(h_boxSize)
        self.v_box.addLayout(h_boxbutton)

        self.setLayout(self.v_box)
        
        self.show()
        
        #Connect the buttons to their functions
        self.button_choosePicture.clicked.connect(self.SelectPicture)
        self.button_SelectArea.clicked.connect(self.SelectArea)
        self.button_cancel.clicked.connect(self.cancel)
        
        #Initialize variable and list
        self.coordinaterectangle=[]
        self.Img=''
        
    def SelectPicture(self):
        """ This function is part of the class 'SelectAreaWindow'. \n
        It is linked to the click of the button 'Choose another picture' (self.button_choosePicture)  \n
        Allow the user to choose another reference picture than the one proposed. """
        #Open a dialog window
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select a picture", "",".png or jpg (*.png *.jpg *.JPG *.jpeg)", options=options)
        
        #Print the address of the selected picture in the label self.label_Picture
        self.label_Picture.setText(str(fileName))
        
    
    def cancel(self):
        """ This function is part of the class 'SelectAreaWindow'. \n
        It is linked to the click of the button 'Cancel' (self.button_cancel)  \n
        Close the window without saving any coordinates """

        self.coordinaterectangle=''
        self.hide()
    
    def SelectArea(self):
        """ This function is part of the class 'SelectAreaWindow'. \n
        It is linked to the click of the button 'START' (self.button_SelectArea)  \n
        Use the function SelectROI of the file FunctionToSelectROI.py to start the selection of the ROI and save the coordinates """
        
        #Decide the reference picture
        if self.label_Picture.text()=='': 
            self.Img=self.ImageNames[0]
        else:    
            self.Img=self.label_Picture.text()
        
        #Call the function 
        if self.CheckBox_SameSize.isChecked():
            self.coordinaterectangle=SelectROI(self.Img,'Y',self.displaySize)
        else:
            self.coordinaterectangle=SelectROI(self.Img,'N',self.displaySize)
        
        #close the window
        self.hide()

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

class ListNameWindow(QDialog):
    """ This class contain the GUI for the window used to start the show the list of the picture addresses when there is more than 5 pictures. \n
    This window can be opened from the class 'MainWindow' by clicking on the button 'Show the whole List' (MainWindow.button_ListImages)  \n
    This class has one parameter: 
        List (List of strings):
            List of the addresses of all the pictures that the user wants to process.  
            (ex: List=['/Users/Name/Desktop/image.png','/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png'])
    """

    def __init__(self, parent=None, List=None):
        super(ListNameWindow, self).__init__(parent)
        ListNameWindow.setFixedSize(self,500, 500)
        self.listPictureNames=List
        self.init_ListNameWindow()
        
    def init_ListNameWindow(self):
        #Name the window
        self.setWindowTitle('Test images List')
        
        #Create the widgets
        self.label_NoWorkingDirectory = QLabel('Selected images:')
        self.button_close = QPushButton('OK') 
        self.label_ImageNames = QLabel('')
        self.scrollArea=QScrollArea(self)
        
        for i in self.listPictureNames: #print the list in the label self.label_ImageNames
            self.label_ImageNames.setText(self.label_ImageNames.text()+'\n'+i)        
        self.scrollArea.setWidget(self.label_ImageNames)
        
        #Create the layout
        self.h_box=QVBoxLayout()
        self.h_box.addWidget(self.label_ImageNames)
        self.h_box.addWidget(self.scrollArea)
        
        self.v_box=QVBoxLayout()
        self.v_box.addWidget(self.label_NoWorkingDirectory)
        self.v_box.addWidget(self.scrollArea)
        self.v_box.addWidget(self.button_close)
        
        self.setLayout(self.v_box)
        
        self.show()
        
        #Connect the button to its function
        self.button_close.clicked.connect(self.close)
    
    def close(self):
        """ This function is part of the class 'ListNameWindow'. \n
        It is linked to the click of the button 'OK' (self.button_close)  \n
        Close the window """

        self.hide()
        
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

class MessageWindow(QDialog):
    """ This class contain the GUI for the window used to display all the warning and error messages of the program . \n
    This window can be opened from almost all the windows when one parameter is missing or wrong \n
    This class has nine parameters: 
        WorkingDirectory (string): 
            Address of the working directory
            (ex: WorkingDirectory='/Users/Name/Desktop/folder'
        
        trainingData (List of strings):
            List of the addresses of all the training data files that the user wants to use to train the machine learning model. 
            (ex: trainingData=['/Users/Name/Desktop/folder/TrainingData/TrainingDataIndividualFiles/trainData_PaddyRice_image1.csv','/Users/Name/Desktop/folder/TrainingData/TrainingDataIndividualFiles/TrainingDataIndividualFiles/trainData_Background_image1.csv'])
        
        listPictureNames (List of strings):
            List of the addresses of all the pictures that the user wants to process.  
            (ex: listPictureNames=['/Users/Name/Desktop/image.png','/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png'])
        
        ROI (string):
            'Whole picture' 
            or a **string** of a List of the coordinates of each region of interest (ROI) in the same order as ListAreaNames.
            For each ROI, the first two numbers are the coordinates of the top left corner and the other two are the coordinates of the bottom right corner.
            (ex: ROI='[[0,0,50,50],[50,0,100,50],[0,50,50,100]]' )
        
        text_InfoTestPictures(string):
            This string give the number of pictures selected
            
        trainingDataPicture (List of strings):
            List of the addresses of all the pictures that the user wants use to create training data files.  
            (ex: trainingDataPicture=['/Users/Name/Desktop/image.png','/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png'])
        NamesList (List of strings):
             List of the names of the ROI in the same order as the list ROI 
             (ex: NamesList=['P1','P2','P1xP2'] ) 
        selectedpixels (string):
            'Y' or 'N'
        Window (string):
            'Y' or 'N'
    
    """
    def __init__(self, parent=None, WorkingDirectory=None, trainingData=None, listPictureNames=None, ROI=None, text_InfoTestPictures=None, trainingDataPicture=None, NamesList=None, selectedpixels=None, Window=None):
        super(MessageWindow, self).__init__(parent)
        self.WorkingDirectory=WorkingDirectory
        self.trainingData=trainingData
        self.listPictureNames=listPictureNames 
        self.ROI=ROI 
        self.text_InfoTestPictures=text_InfoTestPictures
        self.trainingDataPicture=trainingDataPicture
        self.NamesList=NamesList
        self.selectedpixels=selectedpixels
        self.Window=Window
        self.init_MessageWindow()
        
    def init_MessageWindow(self):
        self.setWindowTitle('Warning')

        self.label_NoWorkingDirectory = QLabel('Select or create a working directory')
        self.label_NoTrainingData = QLabel('Select or create a training data set ')
        self.label_NoTestData = QLabel('Choose the picture(s) you want to segment')
        self.label_NoAreaData = QLabel('Choose the working area')
        self.label_NoTrainingPicture = QLabel('Choose the picture(s) you want to use to create the training Data')
        self.label_NoNameROI = QLabel('Please choose a name for every '+self.NamesList)
        self.label_SelectedPixels = QLabel("You haven't selected any pixels. \n Please try again.")
        self.label_Window = QLabel("Please close the other window (except the main window)")
        self.button_close = QPushButton('OK') 
        
        self.v_box=QVBoxLayout()
        self.v_box.addWidget(self.label_NoNameROI)
        self.v_box.addWidget(self.label_NoTrainingPicture)
        self.v_box.addWidget(self.label_NoWorkingDirectory)
        self.v_box.addWidget(self.label_NoTrainingData)
        self.v_box.addWidget(self.label_NoTestData)
        self.v_box.addWidget(self.label_NoAreaData)
        self.v_box.addWidget(self.label_SelectedPixels)
        self.v_box.addWidget(self.label_Window)
        self.v_box.addWidget(self.button_close)
        
        self.label_SelectedPixels.hide()
        self.label_NoTrainingPicture.hide()
        self.label_NoNameROI.hide()
        self.label_NoWorkingDirectory.hide()
        self.label_NoTrainingData.hide()
        self.label_NoTestData.hide()
        self.label_NoAreaData.hide()
        self.label_Window.hide()
        
        if self.selectedpixels=='N':
            self.label_SelectedPixels.show()
        
        if self.NamesList=='Region of Interest' or self.NamesList=='Class' :
            self.label_NoNameROI.show()
        
        if self.trainingDataPicture==[] :
            self.label_NoTrainingPicture.show()
            
        if self.WorkingDirectory=='' or self.WorkingDirectory=='/':
            self.label_NoWorkingDirectory.show()
            
        if self.trainingData=='':
            self.label_NoTrainingData.show()
            
        if self.listPictureNames==[] or self.text_InfoTestPictures=='Path to the directory: (0 picture)':
            self.label_NoTestData.show()
            
        if  self.ROI=='':
            self.label_NoAreaData.show()
        
        if  self.Window=='Y':
            self.label_Window.show()

        self.setLayout(self.v_box)
        
        self.show()
        
        self.button_close.clicked.connect(self.close)
    
    def close(self):
        self.hide()
            
    
    
        
        
        
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
        
        
class MainWindow(QWidget):
    """ This class contain the GUI and the functions for the Main window."""

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        self.init_mainWindow()
        
    def init_mainWindow(self):
        """ This function initialize the class MainWindow'. \n
        It is organized by block, each block corresponding to 1 line of the final GUI"""

        self.setWindowTitle('Segmentation')
        self.resize(1000, 400)
        
        # Select the size of the screen
        self.label_screenSize = QLabel('Select the size of your screen:')
        self.label_screenSize2 = QLabel('(if the size is not in the list, please choose a smaller size)')
        self.comboBox_screenSize = QComboBox()
        self.comboBox_screenSize.addItem('1024 x 640 pixels')
        self.comboBox_screenSize.addItem('1280 x 800 pixels')
        self.comboBox_screenSize.addItem('1440 x 900 pixels')
        self.comboBox_screenSize.addItem('1680 x 1050 pixels')        
        self.comboBox_screenSize.addItem('2048 x 1152 pixels')
        self.comboBox_screenSize.addItem('2560 x 1140 pixels')
        
        self.displaySize=400
        
        self.comboBox_screenSize.activated.connect(self.ScreenSizeFunction)
        
        # Choose or create Working Directory 
        self.label_WorkingDirectory = QLabel('Choose or create a working directory:')
        self.text_InfoWorkingDirectory = QLabel('Selected working directory:')
        self.text_WorkingDirectory = QLabel()
        self.button_ChooseWorkingDirectory = QPushButton('Choose a directory')
        self.button_CreateNewWorkingDirectory = QPushButton('Create a new directory')        
        self.label_PathDesktop = QLabel('You have chosen the Desktop as working directory. \n Please consider that this may be not very convenient \n especially if you are planning to analyse many pictures')
        self.label_PathDesktop.hide()
              
        self.button_ChooseWorkingDirectory.clicked.connect(self.ChooseWorkingDirectoryFunction)
        self.button_CreateNewWorkingDirectory.clicked.connect(self.CreateNewWorkingDirectoryFunction)
        
        v_boxl1_button = QVBoxLayout()
        v_boxl1_button.addWidget(self.button_CreateNewWorkingDirectory)
        v_boxl1_button.addWidget(self.button_ChooseWorkingDirectory)
        
        v_boxl1_text = QVBoxLayout()
        v_boxl1_text.addWidget(self.text_InfoWorkingDirectory)
        v_boxl1_text.addWidget(self.text_WorkingDirectory)
        self.WorkingDirectory=''

        # Choose or create the Training Data
                
        self.label_TrainingData = QLabel('Choose or create a training Data set:')
        self.text_InfoTrainingData = QLabel('Training data set path:')
        self.text_TrainingData = QLabel()
        self.text_NbClasses = QLabel('Number of classes:')
        self.button_newTrainingData = QPushButton('Create a new set of training data')
        self.button_oldTrainingData = QPushButton('Use a training data file')

        self.button_newTrainingData.clicked.connect(self.newTrainingDataFunction)
        self.button_oldTrainingData.clicked.connect(self.SelectTrainingDataFileFunction)
        
        v_boxl2_button = QVBoxLayout()
        v_boxl2_button.addWidget(self.button_newTrainingData)
        v_boxl2_button.addWidget(self.button_oldTrainingData)

        v_boxl2_text = QVBoxLayout()
        v_boxl2_text.addWidget(self.text_InfoTrainingData)
        v_boxl2_text.addWidget(self.text_TrainingData)
        v_boxl2_text.addWidget(self.text_NbClasses)

        #Choose test pictures
                
        self.label_ChooseTestPictures = QLabel('Choose the pictures to be processed:')
        self.text_ModeTestPictures = QLabel('Mode:')
        self.text_InfoTestPictures = QLabel('Path to the test picture(s):')
        self.text_TestPictures = QLabel()
        self.button_ListImages = QPushButton('Show the whole List')

        self.button_ChooseTestPicturesFiles = QPushButton('Select one or several picture(s)')
        self.button_ChooseTestPicturesDirectory = QPushButton('Select a whole Directory')
        
        self.button_ListImages.clicked.connect(self.ShowListImages)
        self.button_ChooseTestPicturesFiles.clicked.connect(self.lookForTestPictureFilesFunction)
        self.button_ChooseTestPicturesDirectory.clicked.connect(self.lookForTestPictureDirectoryFunction)
        
        v_boxl3_button = QVBoxLayout()
        v_boxl3_button.addWidget(self.button_ChooseTestPicturesFiles)
        v_boxl3_button.addWidget(self.button_ChooseTestPicturesDirectory)
        
        v_boxl3_text = QVBoxLayout()
        v_boxl3_text.addWidget(self.text_ModeTestPictures)
        v_boxl3_text.addWidget(self.text_InfoTestPictures)  
        v_boxl3_text.addWidget(self.text_TestPictures)  
        v_boxl3_text.addWidget(self.button_ListImages)  
        self.button_ListImages.hide()
        
        #  Choose the  ROI

        self.label_RegionOfInterest = QLabel('Choose the Region(s) Of Interest (ROI):')
        self.text_RegionOfInterest = QLabel('Whole pictures')
        self.text_InfoRegionOfInterest = QLabel('Region(s) Of Interest (ROI:')
        self.button_WholePicture= QPushButton('Use the whole picture(s)')
        self.button_RegionOfInterest = QPushButton('Select one or several ROI')
        self.button_ROIFile = QPushButton('Use coordinates from a file')
    
        self.button_WholePicture.clicked.connect(self.UseTheWholePictureFunction)
        self.button_RegionOfInterest.clicked.connect(self.SelectRegionOfInterestFunction)
        self.button_ROIFile.clicked.connect(self.UseACoordinatesFileFunction)
        
        v_boxl4_button = QVBoxLayout()
        v_boxl4_button.addWidget(self.button_RegionOfInterest)
        v_boxl4_button.addWidget(self.button_WholePicture)
        v_boxl4_button.addWidget(self.button_ROIFile)
        
        v_boxl4_text = QVBoxLayout()
        v_boxl4_text.addWidget(self.text_InfoRegionOfInterest)
        v_boxl4_text.addWidget(self.text_RegionOfInterest)
        
        self.ListAreaNames=[]
        
        
        #Choose the machine learning model 
        
        self.label_model = QLabel('Choose the machine learning model:')
        self.comboBox_model = QComboBox()
        self.comboBox_model.addItem('Classification and Regression Tree (Sklearn)')
        self.comboBox_model.addItem('Random Forest Classifier (Sklearn)')
        self.comboBox_model.addItem('Support Vector Machine (Sklearn)')

        
        #Noise reduction
        
        self.label_noiseReduction = QLabel('Size filter for the noise reduction:')
        self.spinbox_noiseReduction = QSpinBox()
        self.spinbox_noiseReduction.setMinimum(0)
        self.spinbox_noiseReduction.setMaximum(10000)
        self.spinbox_noiseReduction.setSingleStep(100)
        self.spinbox_noiseReduction.setValue(100)
        
#######################Only keep the biggest blob and do shape analysis
        self.label_BiggestBlob= QLabel('Region filter:')
        self.CheckBox_BiggestBlobYes=QCheckBox('Only keep the biggest region and do shape analysis')
        self.CheckBox_BiggestBlobNo=QCheckBox('Keep all the regions')
        self.CheckBox_BiggestBlobYes.setChecked(True)
        self.CheckBox_BiggestBlobNo.setChecked(False)
        
        v_box_BiggestBlob=QVBoxLayout()
        v_box_BiggestBlob.addWidget(self.CheckBox_BiggestBlobYes)
        v_box_BiggestBlob.addWidget(self.CheckBox_BiggestBlobNo)
        
        self.CheckBox_BiggestBlobYes.stateChanged.connect(self.CheckBox_BiggestBlobYesFunction)
        self.CheckBox_BiggestBlobNo.stateChanged.connect(self.CheckBox_BiggestBlobNoFunction)
        
        # Choose the output
        self.label_chosenOutput= QLabel('Choose the outputs:')
        self.CheckBox_infoFile=QCheckBox('Information file (surface, coverage + shape analysis)')
        self.CheckBox_NonFilteredMask=QCheckBox('Non filtered mask')
        self.CheckBox_BlackandWhiteMask=QCheckBox('Black and White mask')
        self.CheckBox_ReconstructedImage=QCheckBox('Reconstructed image')
        self.CheckBox_ColoredMask=QCheckBox('Colored mask')
        
        
        self.CheckBox_ReconstructedImage.setChecked(True)
        self.CheckBox_BlackandWhiteMask.setChecked(True)
        self.CheckBox_infoFile.setChecked(True)
        self.CheckBox_NonFilteredMask.setChecked(True)

        v_box_Output=QVBoxLayout()
        v_box_Output.addWidget(self.CheckBox_infoFile)
        v_box_Output.addWidget(self.CheckBox_NonFilteredMask)
        v_box_Output.addWidget(self.CheckBox_ReconstructedImage)
        v_box_Output.addWidget(self.CheckBox_BlackandWhiteMask)
        v_box_Output.addWidget(self.CheckBox_ColoredMask)
        

        #Choose the classes of interest
        
        self.label_chosenClassesForSurface= QLabel('Choose the class of interest:')
        
        self.CheckBox_fusion=QCheckBox('Fusion some of the classes')
        
        self.CheckBox1=QCheckBox('Class_1')
        self.CheckBox2=QCheckBox('Class_2')
        self.CheckBox3=QCheckBox('Class_3')
        self.CheckBox4=QCheckBox('Class_4')
        self.CheckBox5=QCheckBox('Class_5')
        
        self.CheckBox1.setChecked(True)
        
        self.CheckBox_ColoredMask.hide()
        self.CheckBox_fusion.hide()
        self.CheckBox3.hide()
        self.CheckBox4.hide()
        self.CheckBox5.hide() 

        h_box_Checkboxes=QHBoxLayout()
        h_box_Checkboxes.addWidget(self.CheckBox1)
        h_box_Checkboxes.addWidget(self.CheckBox2)
        h_box_Checkboxes.addWidget(self.CheckBox3)
        h_box_Checkboxes.addWidget(self.CheckBox4)
        h_box_Checkboxes.addWidget(self.CheckBox5)
        
        v_box_Checkboxes=QVBoxLayout()
        v_box_Checkboxes.addWidget(self.CheckBox_fusion)
        v_box_Checkboxes.addWidget(self.label_chosenClassesForSurface)
        v_box_Checkboxes.addLayout(h_box_Checkboxes)
        
        self.ListClassesForSurface=[]
        
        self.CheckBox1.stateChanged.connect(self.ListClassesForSurfaceFunction)
        self.CheckBox2.stateChanged.connect(self.ListClassesForSurfaceFunction)
        self.CheckBox3.stateChanged.connect(self.ListClassesForSurfaceFunction)
        self.CheckBox4.stateChanged.connect(self.ListClassesForSurfaceFunction)
        self.CheckBox5.stateChanged.connect(self.ListClassesForSurfaceFunction)
        
        self.CheckBox_fusion.stateChanged.connect(self.CheckFusionFunction)        
        
        # Last line: execute 
        self.button_execute = QPushButton('Execute')
      
        self.button_execute.clicked.connect(self.Execute)    
                
        
        # separation lines 
        self.Line=[]
        for i in range(24):
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            self.Line.append(line)

        #create the global layout using a grid
        grid=QGridLayout()
        
        grid.addWidget(self.label_screenSize, 0, 0)
        grid.addWidget(self.comboBox_screenSize,0,1)
        grid.addWidget(self.label_screenSize2,0,2)
        
        
        grid.addWidget(self.Line[0],1,0)
        grid.addWidget(self.Line[1],1,1)
        grid.addWidget(self.Line[2],1,2)

        grid.addWidget(self.label_WorkingDirectory, 2, 0)
        grid.addLayout(v_boxl1_button,2,1)
        grid.addLayout(v_boxl1_text, 2, 2)
        
        grid.addWidget(self.Line[3],3,0)
        grid.addWidget(self.Line[4],3,1)
        grid.addWidget(self.Line[5],3,2)
        
        
        grid.addWidget(self.label_TrainingData, 4, 0)
        grid.addLayout(v_boxl2_text, 4, 2)
        grid.addLayout(v_boxl2_button, 4, 1)
        
        grid.addWidget(self.Line[6],5,0)
        grid.addWidget(self.Line[7],5,1)
        grid.addWidget(self.Line[8],5,2)
        
        grid.addWidget(self.label_ChooseTestPictures, 6, 0 )
        grid.addLayout(v_boxl3_button, 6, 1)                        
        grid.addLayout(v_boxl3_text, 6,2)
        
        grid.addWidget(self.Line[9],7,0)
        grid.addWidget(self.Line[10],7,1)
        grid.addWidget(self.Line[11],7,2)
        
        grid.addWidget(self.label_RegionOfInterest, 8,0)
        grid.addLayout (v_boxl4_button, 8, 1)
        grid.addLayout(v_boxl4_text, 8, 2)
        
        grid.addWidget(self.Line[12],9,0)
        grid.addWidget(self.Line[13],9,1)
        grid.addWidget(self.Line[14],9,2)
        
        grid.addWidget(self.label_model, 10,0)
        grid.addWidget(self.comboBox_model,10,1)

        grid.addWidget(self.Line[15],11,0)
        grid.addWidget(self.Line[16],11,1)
        grid.addWidget(self.Line[17],11,2)
        
        grid.addWidget(self.label_noiseReduction,12,0)
        grid.addWidget(self.spinbox_noiseReduction,12,1)
        
        grid.addWidget(self.Line[18],13,1)
        grid.addWidget(self.Line[19],13,2)
        grid.addWidget(self.Line[20],13,0)
        
        grid.addWidget(self.label_chosenOutput, 14,0)
        grid.addLayout(v_box_Checkboxes,14,1)
        grid.addLayout(v_box_Output,14,2)
        
        grid.addWidget(self.Line[21],15,1)
        grid.addWidget(self.Line[22],15,2)
        grid.addWidget(self.Line[23],15,0)
        
        grid.addWidget(self.label_BiggestBlob,16,0)
        grid.addLayout(v_box_BiggestBlob,16,1)
    
        grid.addWidget(self.button_execute,17,2)
        
        #Create a frame
        Frame= QFrame()
        Frame.setLayout(grid)
        
        # Add a scrollArea  to the frame
        self.scrollArea=QScrollArea()
        self.scrollArea.setWidget(Frame)
        self.scrollArea.setWidgetResizable(True)

        Layout=QHBoxLayout()
        Layout.addWidget(self.scrollArea)
        self.setLayout(Layout)
        
        self.show()
        
        #Initialize the variables
        self.listPictureNames=[]
        self.ROI='Whole pictures'
        self.trainingData=''
        self.WorkingDirectory=''
        self.classes=2
        self.classesNamesList=['Class_1','Class_2']
        
        
   
    def ScreenSizeFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the change of the combo box self.comboBox_screenSize \n
        It decides the maximum size for the display of pictures depending on the size of the screen """
        if self.comboBox_screenSize.currentText()=='1024 x 640 pixels':
            self.displaySize=400
        if self.comboBox_screenSize.currentText()=='1280 x 800 pixels':
            self.displaySize=600
        if self.comboBox_screenSize.currentText()=='1440 x 900 pixels':
            self.displaySize=700
        if self.comboBox_screenSize.currentText()=='1680 x 1050 pixels':
            self.displaySize=800
        if self.comboBox_screenSize.currentText()=='2048 x 1152 pixels':
            self.displaySize=900
        if self.comboBox_screenSize.currentText()=='2560 x 1140 pixels':
            self.displaySize=900
        
    def ChooseWorkingDirectoryFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Choose a directory' (self.button_ChooseWorkingDirectory)  \n
        Allow the user to select a directory as working directory """
        #Open a dialog window 
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        DirectoryName= str(QFileDialog.getExistingDirectory(self, "Select the Working Directory"))
        
        #Save the address of the chosen directory 
        self.text_WorkingDirectory.setText(DirectoryName)
        self.WorkingDirectory=DirectoryName
        
    def CreateNewWorkingDirectoryFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Create a new directory' (self.button_CreateNewWorkingDirectory)  \n
        Open the window 'DirectoryWindow' and save the new address"""
        
        #Open the 'DirectoryWindow'  
        widget = DirectoryWindow(self)
        widget.exec_()
        
        #Save the new address
        if widget.text_newWorkingDirectoryPath.text()!='Directory/' and widget.text_newWorkingDirectoryPath.text()!='':
            self.text_WorkingDirectory.setText(widget.text_newWorkingDirectoryPath.text()+'/'+ widget.text_newWorkingDirectoryName.text())
            self.WorkingDirectory=widget.text_newWorkingDirectoryPath.text()+'/'+ widget.text_newWorkingDirectoryName.text()
    
    
    def newTrainingDataFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Create a new set of training data' (self.button_newTrainingData)  \n
        Open the window 'TrainingWindow' """
        
        #Error message if no working directory has been chosen
        if self.WorkingDirectory=='':
            messageWin=MessageWindow(parent=self, WorkingDirectory=self.WorkingDirectory, trainingData='OK', listPictureNames='OK', ROI='OK', text_InfoTestPictures='OK',trainingDataPicture='OK', NamesList='Y', selectedpixels='Y')
            messageWin.exec_() 

        else :   
            self.text_TrainingData.setText('The training data window is still open, please close it before doing anything else.') # Warning as long as the window is opened
            trainWin = TrainingWindow(parent=self, workingDirectory=self.WorkingDirectory, DisplaySize=self.displaySize)
            trainWin.exec_()
            self.text_TrainingData.setText('')
            
            #Save the number of classes and their names and the address of the combine training file
            self.classes=trainWin.classes
            self.classesNamesList=trainWin.classesNamesList
            if trainWin.newfilecreated=='Y':
                self.text_TrainingData.setText(self.WorkingDirectory+'/TrainingData/trainData_'+str(self.classes)+'classes.csv'+'\n A new set of Training Data have been created')
                self.trainingData=self.WorkingDirectory+'/TrainingData/trainData_'+str(self.classes)+'classes.csv'
            self.text_NbClasses.setText('Number of classes: '+str(int(self.classes)))
            self.ListtrainingData=trainWin.ListTrainingDataFile
            
        #Adapt the number and names of the checkbox 
        if self.classes==2:
            self.CheckBox1.setText(self.classesNamesList[0])
            self.CheckBox2.setText(self.classesNamesList[1])
            self.CheckBox1.show()
            self.CheckBox2.show()
            self.CheckBox3.hide()
            self.CheckBox4.hide()
            self.CheckBox5.hide()
            self.CheckBox3.setChecked(False)
            self.CheckBox4.setChecked(False)
            self.CheckBox5.setChecked(False)
            self.CheckBox_BlackandWhiteMask.show()
            self.CheckBox_ColoredMask.hide()
            self.CheckBox_ColoredMask.setChecked(False)
            self.CheckBox_fusion.hide()
            self.CheckBox_BiggestBlobYes.setChecked(True)
            self.CheckBox_BiggestBlobNo.setChecked(False)
            self.label_BiggestBlob.show()
            self.CheckBox_BiggestBlobYes.show()
            self.CheckBox_BiggestBlobNo.show()
            self.Line[21].show()
            self.Line[22].show()
            self.Line[23].show()
            
        if self.classes>2:
            self.CheckBox_infoFile.hide()
            self.CheckBox_infoFile.setChecked(False)
            self.label_chosenClassesForSurface.hide()
            self.CheckBox_BlackandWhiteMask.hide()
            self.CheckBox_ColoredMask.show()
            self.CheckBox_ColoredMask.setChecked(True)
            self.CheckBox_fusion.show()
            self.CheckBox1.hide()
            self.CheckBox2.hide()
            self.CheckBox_BiggestBlobYes.setChecked(False)
            self.CheckBox_BiggestBlobNo.setChecked(True)
            self.label_BiggestBlob.hide()
            self.CheckBox_BiggestBlobYes.hide()
            self.CheckBox_BiggestBlobNo.hide()
            self.Line[21].hide()
            self.Line[22].hide()
            self.Line[23].hide()
            

        
            
        
    def SelectTrainingDataFileFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Use a training data file' (self.button_oldTrainingData)  \n
        Allow the user to use an already created training data file """
       
        #Open a dialog window
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.ListtrainingData, _ = QFileDialog.getOpenFileNames(self,"Select the training data file", "",".csv (*.csv)", options=options)
        
        #Extract the name of the classes in the file and their number 
        self.classesNamesList=[]
        for file in self.ListtrainingData:
            if self.ListtrainingData!='':
                f=open(file,"r",newline='') 
                TrainData = list(csv.reader(f))
                f.close()
                TrainData.remove(['Class', 'Image', 'x','y','B','G','R','H','S','V','L','a','b'])
                TrainData=np.asarray(TrainData)  
                classes=TrainData[:,0]
                
                for Class in classes:
                    if Class in self.classesNamesList:
                        self.classesNamesList=self.classesNamesList                  
                    else:
                        self.classesNamesList.append(Class)
        self.classes=len(self.classesNamesList)
        self.text_TrainingData.setText(str(self.ListtrainingData))
        self.text_NbClasses.setText('Number of classes: '+str(int(self.classes)))

        #Adapt the number and names of the checkbox 
        if self.classes==2:
            self.CheckBox1.setText(self.classesNamesList[0])
            self.CheckBox2.setText(self.classesNamesList[1])
            self.CheckBox1.show()
            self.CheckBox2.show()
            self.CheckBox3.hide()
            self.CheckBox4.hide()
            self.CheckBox5.hide()
            self.CheckBox3.setChecked(False)
            self.CheckBox4.setChecked(False)
            self.CheckBox5.setChecked(False)
            self.CheckBox_BlackandWhiteMask.show()
            self.CheckBox_ColoredMask.hide()
            self.CheckBox_ColoredMask.setChecked(False)
            self.CheckBox_fusion.hide()
            self.CheckBox_BiggestBlobYes.setChecked(True)
            self.CheckBox_BiggestBlobNo.setChecked(False)
            self.label_BiggestBlob.show()
            self.CheckBox_BiggestBlobYes.show()
            self.CheckBox_BiggestBlobNo.show()
            self.Line[21].show()
            self.Line[22].show()
            self.Line[23].show()
            
        if self.classes>2:
            self.CheckBox_infoFile.hide()
            self.CheckBox_infoFile.setChecked(False)
            self.label_chosenClassesForSurface.hide()
            self.CheckBox_BlackandWhiteMask.hide()
            self.CheckBox_ColoredMask.show()
            self.CheckBox_ColoredMask.setChecked(True)
            self.CheckBox_fusion.show()
            self.CheckBox1.hide()
            self.CheckBox2.hide()
            self.CheckBox_BiggestBlobYes.setChecked(False)
            self.CheckBox_BiggestBlobNo.setChecked(True)
            self.label_BiggestBlob.hide()
            self.CheckBox_BiggestBlobYes.hide()
            self.CheckBox_BiggestBlobNo.hide()
            self.Line[21].hide()
            self.Line[22].hide()
            self.Line[23].hide()
            
        
        
    def lookForTestPictureFilesFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Select one or several picture(s)' (self.button_ChooseTestPicturesFiles)  \n
        Allow the user to choose the pictures he/she would like to test """

        self.text_TestPictures.setText('')
        self.button_ListImages.hide()
        
        #Open a dialog window to select the files
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.listPictureNames, _ = QFileDialog.getOpenFileNames(self,"Select the test picture(s)", "",".png or jpg (*.png *.jpg *.JPG *.jpeg)", options=options)
        
        #Print the first five addresses in the label self.text_TestPictures
        if self.listPictureNames:
            for i in range(len(self.listPictureNames)):
                if i<5:
                    self.text_TestPictures.setText(self.text_TestPictures.text()+'\n'+str(self.listPictureNames[i]))
                if i==5 and len(self.listPictureNames)>5:
                    self.text_TestPictures.setText(self.text_TestPictures.text()+'\n ...')                

        #If there is more than 5 pictures, show the button to the class 'ListNameWindow'
        if len(self.listPictureNames)>5:
            self.button_ListImages.show()

        self.text_ModeTestPictures.setText('Mode: Files')
        
        #Count the number of pictures
        if len(self.listPictureNames)==0 or len(self.listPictureNames)==1:
            self.text_InfoTestPictures.setText('Path to the test pictures: ('+str(len(self.listPictureNames))+' picture)')
        else :
            self.text_InfoTestPictures.setText('Path to the test pictures: ('+str(len(self.listPictureNames))+' pictures)')
        
        if len(self.listPictureNames)!=0:
            self.RefImg=self.listPictureNames[0]
            
            
    def ShowListImages(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Show the whole List' (self.button_ListImages)  \n
        Open the window 'ListNameWindow' """

        ListWin=ListNameWindow(parent=self, List=self.listPictureNames)
        ListWin.exec_()
            
    def lookForTestPictureDirectoryFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Select a whole Directory' (self.button_ListImages)  \n
        Allow the user to choose the all the pictures of a directory"""

        self.text_TestPictures.setText('')
        self.button_ListImages.hide()
        
        #Open a dialog window and select the directory
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        DirectoryName= str(QFileDialog.getExistingDirectory(self, "Select Directory with the test picture(s)"))
        
        #Make the list of all the picture inside the directory
        if DirectoryName!='':           
            self.text_TestPictures.setText(DirectoryName)
            self.text_ModeTestPictures.setText('Mode: Directory')
            
            ListImageinDirectory=os.listdir(DirectoryName)      
            ListImage=[]        
            for i in range(len(ListImageinDirectory)):
                a=ListImageinDirectory[i].split('.')
                if 'jpeg'in a or 'png'in a or 'jpg'in a or 'JPG' in a:   #only take the images
                    ListImage.append(DirectoryName+'/'+ListImageinDirectory[i]) 
            
            self.listPictureNames=ListImage
            
            #Count the number of picture
            if len(self.listPictureNames)==0 or len(self.listPictureNames)==1:
                self.text_InfoTestPictures.setText('Path to the directory: ('+str(len(self.listPictureNames))+' picture)')
            else :
                self.text_InfoTestPictures.setText('Path to the directory: ('+str(len(self.listPictureNames))+' pictures)')
            self.button_ListImages.show()
            
            self.RefImg=self.listPictureNames[0]
            
    def UseTheWholePictureFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Use the whole picture(s)' (self.button_WholePicture)  \n
        Allow the user to choose to process the whole pictures instead of ROI"""

        self.ROI='Whole pictures'
        self.text_RegionOfInterest.setText('Whole pictures')
        
    def SelectRegionOfInterestFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button  'Select one or several ROI' (self.button_RegionOfInterest)  \n
        Open the window 'RegionOfInterestWindow' and save the ROI and their names"""
        
        #Error message if one of the parameter is missing
        if self.listPictureNames==[] or self.text_InfoTestPictures.text()=='Path to the directory: (0 picture)' or self.WorkingDirectory=='' or self.WorkingDirectory=='/':
            messageWin=MessageWindow(parent=self, WorkingDirectory=self.WorkingDirectory, trainingData='OK', listPictureNames=self.listPictureNames, ROI='OK', text_InfoTestPictures=self.text_InfoTestPictures.text(), trainingDataPicture='OK', NamesList='Y', selectedpixels='Y', Window='N')
            messageWin.exec_() 
        else :   
            # Open the window 'RegionOfInterestWindow' 
            self.text_RegionOfInterest.setText('The ROI window is opened. Please close it before doing anything else.')
            RegionOfInterestWin = RegionOfInterestWindow(parent=self, workingDirectory=self.WorkingDirectory, ImageNames=self.listPictureNames, DisplaySize=self.displaySize)
            RegionOfInterestWin.exec_() 
            self.text_RegionOfInterest.setText('')
            
            #Get the coordinates and save them 
            coordinates=RegionOfInterestWin.coordinaterectangle
            self.ListAreaNames=RegionOfInterestWin.AreaNamesList
            if coordinates!=[]:
                self.RefImg=RegionOfInterestWin.RefImg
                self.ROI=str(coordinates)
                if RegionOfInterestWin.CheckBox_UseAsAMask.isChecked():  
                    if len(self.ROI)==0 or len(self.ROI)==1: #count the number of ROI
                        self.text_RegionOfInterest.setText('The selected area(s) will be used as ROI  ('+str(len(coordinates))+' region)')
                    else :
                        self.text_RegionOfInterest.setText('The selected area(s) will be used as ROI  ('+str(len(coordinates))+' regions)')
                else :
                    if len(self.ROI)==0 or len(self.ROI)==1:
                        self.text_RegionOfInterest.setText('The pictures have been cropped  ('+str(len(coordinates))+' region)')
                    else:
                        self.text_RegionOfInterest.setText('The pictures have been cropped  ('+str(len(coordinates))+' regions)')
            else:
                self.text_RegionOfInterest.setText('')
                    
    def UseACoordinatesFileFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the button 'Use coordinates from a file' (self.button_ROIFile)  \n
        Allow the user to choose a already created file containing the coordinates and name of the ROI"""

        #Open a dialog window to select the file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        CoordinatesFile, _ = QFileDialog.getOpenFileName(self,"Select the file containing the coordinates", "",".txt (*.txt)", options=options)
        
        #Extract from the file the info
        if CoordinatesFile!='':
            f=open(CoordinatesFile,"r",newline='')      
            content = f.readlines() 
            f.close()
            if content!=[]:
                for i in range(len(content)):
                    content[i]=content[i].replace('\n','')
                    content[i]=content[i].split(' ')
                    #Extract the name of the reference picture on the first line
                    if i==0:
                        self.RefImg=content[i][0]
                        
                    else:
                        #extract the coordinates and names of the ROI
                        self.ListAreaNames.append(content[i][0])
                        content[i].remove(content[i][0])
                        for j in range(len(content[i])):
                            content[i][j]=int(content[i][j])
                del content[0]
                self.ROI=str(content)
                
                #Count the number of ROI
                if len(content)==0 or len(content)==1:
                    self.text_RegionOfInterest.setText('Path to the file: ('+str(len(content))+' region) \n'+CoordinatesFile)
                else:
                    self.text_RegionOfInterest.setText('Path to the file: ('+str(len(content))+' regions) \n'+CoordinatesFile)
            
            else: #if the file is empty
                self.ROI=''
                self.text_RegionOfInterest.setText('The file you have selected is empty')
        
    
    def CheckBox_BiggestBlobYesFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the checkbox 'Only keep the biggest blob/region and do shape analysis' (self.CheckBox_BiggestBlobYes)  \n
        This function is made so that both of the checkboxes are not checked at the same time"""
        
        if self.CheckBox_BiggestBlobYes.isChecked()==True: 
            self.CheckBox_BiggestBlobNo.setChecked(False)
            self.CheckBox_infoFile.setText('Information file (surface, coverage + shape analysis)')
        
        if self.CheckBox_BiggestBlobYes.isChecked()==False: 
            self.CheckBox_BiggestBlobNo.setChecked(True)
            self.CheckBox_infoFile.setText('Information file (surface, coverage)')
    
    def CheckBox_BiggestBlobNoFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the checkbox 'Keep all the regions' (self.CheckBox_BiggestBlobNo)  \n
        This function is made so that both of the checkboxes are not checked at the same time"""
        
        if self.CheckBox_BiggestBlobNo.isChecked()==True: 
            self.CheckBox_BiggestBlobYes.setChecked(False)
            self.CheckBox_infoFile.setText('Information file (surface, coverage)')
            
        if self.CheckBox_BiggestBlobNo.isChecked()==False: 
            self.CheckBox_BiggestBlobYes.setChecked(True)
            self.CheckBox_infoFile.setText('Information file (surface, coverage + shape analysis)')
    
    
    def CheckFusionFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the checkbox 'Fusion some of the classes' (self.CheckBox_fusion)  \n
        Changes the widgets according to the user choice to fusion some classes or not"""


        if self.CheckBox_fusion.isChecked()==True:
            self.CheckBox1.setText(self.classesNamesList[0])
            self.CheckBox2.setText(self.classesNamesList[1])
            self.CheckBox1.show()
            self.CheckBox2.show()
            self.CheckBox_BlackandWhiteMask.show()
            self.CheckBox_infoFile.setChecked(True)
            self.CheckBox_ColoredMask.hide()
            self.CheckBox_infoFile.show()
            self.CheckBox_ColoredMask.setChecked(False)
            self.CheckBox_fusion.show()
            self.label_chosenClassesForSurface.setText('Choose the classes of interest:')
            self.CheckBox_BiggestBlobYes.setChecked(True)
            self.CheckBox_BiggestBlobNo.setChecked(False)
            self.label_BiggestBlob.show()
            self.CheckBox_BiggestBlobYes.show()
            self.CheckBox_BiggestBlobNo.show()
            self.Line[21].show()
            self.Line[22].show()
            self.Line[23].show()
            self.CheckBox_infoFile.setText('Information file (surface, coverage + shape analysis)')

            if self.classes==3:
                self.CheckBox1.setText(self.classesNamesList[0])
                self.CheckBox2.setText(self.classesNamesList[1])
                self.CheckBox3.setText(self.classesNamesList[2])
                self.CheckBox3.show()
                self.CheckBox4.hide()   
                self.CheckBox5.hide()
                self.CheckBox4.setChecked(False)
                self.CheckBox5.setChecked(False)
            if self.classes==4:
                self.CheckBox1.setText(self.classesNamesList[0])
                self.CheckBox2.setText(self.classesNamesList[1])
                self.CheckBox3.setText(self.classesNamesList[2])
                self.CheckBox4.setText(self.classesNamesList[3])
                self.CheckBox3.show()
                self.CheckBox4.show() 
                self.CheckBox5.hide()
                self.CheckBox5.setChecked(False)
            if self.classes==5:
                self.CheckBox1.setText(self.classesNamesList[0])
                self.CheckBox2.setText(self.classesNamesList[1])
                self.CheckBox3.setText(self.classesNamesList[2])
                self.CheckBox4.setText(self.classesNamesList[3])
                self.CheckBox5.setText(self.classesNamesList[4])
                self.CheckBox3.show()
                self.CheckBox4.show()                     
                self.CheckBox5.show()
            
        else:
            self.CheckBox_infoFile.hide()
            self.CheckBox_infoFile.setChecked(False)
            self.label_chosenClassesForSurface.hide()
            self.CheckBox_BlackandWhiteMask.hide()
            self.CheckBox_ColoredMask.show()
            self.CheckBox_ColoredMask.setChecked(True)
            self.CheckBox_fusion.show()
            self.CheckBox1.hide()
            self.CheckBox2.hide()
            self.CheckBox3.hide()
            self.CheckBox4.hide()
            self.CheckBox5.hide()
            self.CheckBox_BiggestBlobYes.setChecked(False)
            self.CheckBox_BiggestBlobNo.setChecked(True)
            self.label_BiggestBlob.hide()
            self.CheckBox_BiggestBlobYes.hide()
            self.CheckBox_BiggestBlobNo.hide()
            self.Line[21].hide()
            self.Line[22].hide()
            self.Line[23].hide()
        
    def ListClassesForSurfaceFunction(self):
        """ This function is part of the class 'MainWindow'. \n
        It is linked to the click of the checkboxes 'Class_1' to 'Class_5' (self.CheckBox1 to self.CheckBox2)  \n
        Create a list of the classes the user wants to fusion """
        self.ListClassesForSurface=[]
        
        if self.CheckBox1.isChecked()==True:
            self.ListClassesForSurface.append(self.classesNamesList[0])            
        if self.CheckBox2.isChecked()==True:
            self.ListClassesForSurface.append(self.classesNamesList[1])            
        if self.CheckBox3.isChecked()==True:
            self.ListClassesForSurface.append(self.classesNamesList[2])            
        if self.CheckBox4.isChecked()==True:
            self.ListClassesForSurface.append(self.classesNamesList[3])
        if self.CheckBox5.isChecked()==True:
            self.ListClassesForSurface.append(self.classesNamesList[4])
            
        
            
            
    def Execute(self):    
        """ This function is the main and final function of the class 'MainWindow'. \n
        It is linked to the click of the button 'Execute' (self.button_execute)  \n
        Use the function 'Segmentation' from the file FunctionForSegmentation.py to process the pictures"""

        self.Window='N'
        #If one of the sub window is opened
        if self.text_RegionOfInterest.text()=='The ROI window is opened. Please close it before doing anything else.' or self.text_TrainingData.text=='The training data window is still open, please close it before doing anything else.':
            self.Window='Y'
        
        #Message Error if something is missing or wrong
        if self.WorkingDirectory=='' or self.WorkingDirectory=='/' or self.ListtrainingData==[]or self.listPictureNames==[] or self.ROI=='' or self.text_InfoTestPictures.text()=='Path to the directory: (0 picture)' or self.Window=='Y' :

        #if self.WorkingDirectory=='' or self.WorkingDirectory=='/' or self.trainingData=='' or self.listPictureNames==[] or self.ROI=='' or self.text_InfoTestPictures.text()=='Path to the directory: (0 picture)' :
            messageWin=MessageWindow(parent=self, WorkingDirectory=self.WorkingDirectory, trainingData=self.trainingData, listPictureNames=self.listPictureNames, ROI=self.ROI, text_InfoTestPictures=self.text_InfoTestPictures.text(), trainingDataPicture='OK', NamesList='Y', selectedpixels='Y', Window=self.Window)
            messageWin.exec_() 
        
        else :
            #prepare the parameters
            if self.CheckBox_fusion.isChecked()==True:
                fusion='Y'
            else:
                fusion='N'
                
            if self.CheckBox_ColoredMask.isChecked()==True or self.CheckBox_BlackandWhiteMask.isChecked()==True:
                mask='Y'
            else:
                mask='N'
            
            if self.CheckBox_ReconstructedImage.isChecked()==True:
                reconstructedimage='Y'
            else:
                reconstructedimage='N'
            
            if self.CheckBox_infoFile.isChecked()==True:
                info='Y'
            else:
                info='N'
                
            if self.CheckBox_NonFilteredMask.isChecked()==True:
                NFMask='Y'
            else:
                NFMask='N'
            
            if self.CheckBox_BiggestBlobYes.isChecked()==True:
                BiggestBlob='Y'
            else:
                BiggestBlob='N'
            
            if self.ListClassesForSurface==[]:
                self.ListClassesForSurface.append(self.classesNamesList[0])
            
            if self.RefImg==[]:
                self.RefImg=self.listPictureNames[0]
            
            #Initialize the list 
            ListImageWrongSize=[]
            
            #Call the function
            start_time = time.monotonic() 
            
            ListImageWrongSize, ListRunningTimes, ListTestDataTimes,ListApplyModelTimes,ListSaveOutputTimes=Segmentation(self.WorkingDirectory, self.ListtrainingData, self.listPictureNames, self.comboBox_model.currentText(), self.spinbox_noiseReduction.value(), self.classes, self.classesNamesList, self.ROI ,self.ListAreaNames, fusion, mask, reconstructedimage, info, NFMask, BiggestBlob, self.ListClassesForSurface, self.RefImg)
            
            end_time = time.monotonic()
            time_all=timedelta(seconds=end_time - start_time)
            
            #Calculate the mean time of each step 
            MeanRunningTime=np.mean(ListRunningTimes)
            MeanRunningTimeTestData=np.mean(ListTestDataTimes)
            MeanRunningTimeModel=np.mean(ListApplyModelTimes)
            MeanRunningTimeOutput=np.mean(ListSaveOutputTimes)
            
            #Save all the parameter in a text file called Activity_Report.txt
            ReferencePicture=cv2.imread(self.RefImg) 
            sizefirstImage=np.shape(ReferencePicture)

            if self.ROI!='Whole pictures':
                self.ROI2=ast.literal_eval(self.ROI)
                x1,y1,x2,y2=self.ROI2[0]
                H=y2-y1
                W=x2-x1
                sizeROI=(H,W)
            else:
                sizeROI=sizefirstImage
                
            report=('Activity report: '+
                    '\n \n Working directory: '+str(self.WorkingDirectory)+
                    '\n \n Training data: '+str(self.ListtrainingData)+
                    '\n \n Number of classes: '+str(self.classes)+
                    '\n \n Classes name: '+ str(self.classesNamesList)+
                    '\n \n Fusion of classes: '+str(fusion)+ 
                    '\n \n Classe(s) of interest: '+str(self.ListClassesForSurface)+
                    '\n \n Number of pictures tested: '+str(len(self.listPictureNames))+
                    '\n \n Size of the pictures: '+str(sizefirstImage)+
                    '\n \n Model: '+str(self.comboBox_model.currentText())+
                    '\n \n Number of regions of interest:'+str(len(self.ListAreaNames))+
                    '\n \n Regions of interest coordinates: '+str(self.ROI)+
                    '\n \n Region names: '+str(self.ListAreaNames)+
                    '\n \n Size of the regions of interest: '+str(sizeROI)+
                    '\n \n Noise reduction: '+str(self.spinbox_noiseReduction.value())+
                    '\n \n Mask saved: '+str(mask)+
                    '\n \n Reconstructed image saved: '+ str(reconstructedimage)+
                    '\n \n Information file saved: '+str(info)+
                    '\n \n Only keep the biggest region: '+str(BiggestBlob)+
                    '\n \n Reference picture used for choosing the region of interest: '+str(self.RefImg)+
                    '\n \n Total Running time: '+str(time_all)+
                    '\n \n Mean running time for each pictures: '+ str(MeanRunningTime)+'sec' +
                    '\n \n Mean time to create the test data: '+str(MeanRunningTimeTestData)+'sec' +
                    '\n \n Mean time to apply the model: '+str(MeanRunningTimeModel)+'sec' +
                    '\n \n Mean time to save the outputs: '+str(MeanRunningTimeOutput)+'sec' +
                    '\n \n Pictures which have not been processed because of their size: '+str(ListImageWrongSize)+
                    '\n \n Pictures List: '+str(self.listPictureNames) )
            
            report=np.array([report])
            np.savetxt(self.WorkingDirectory+'/Activity_Report.txt', report, delimiter="\n", comments='',fmt='%s')
            
            
            
            #Open the final window 
            messageWin=MessageWindowEndOfProgram(parent=self, ListImageWrongSize=ListImageWrongSize, ReferencePicture=self.RefImg)
            messageWin.exec_() 
            
                
            

class MessageWindowEndOfProgram(QDialog):
    """ This class contain the GUI for the final window informing the user that all the pictures have been processed \n
    This window is opened from the class 'MainWindow' at the end of the function 'Execute'  \n
    This class has two parameters: 
        ListImageWrongSize (List of strings):
            List of the addresses of all the pictures that couldn't be processed.  
            (ex: ListImageWrongSize=['/Users/Name/Desktop/image.png','/Users/Name/Desktop/image2.png','/Users/Name/Desktop/image3.png'])
        ReferencePicture (strings):
            Address of the reference pictures used to select the ROI 
            (ex: ReferencePicture='/Users/Name/Desktop/image.png'
    """

    def __init__(self, parent=None, ListImageWrongSize=None, ReferencePicture=None):
        super(MessageWindowEndOfProgram, self).__init__(parent)
        self.ListImageWrongSize=ListImageWrongSize
        self.ReferencePicture=ReferencePicture
        self.init_MessageWindowEndOfProgram()
        
    def init_MessageWindowEndOfProgram(self):
        self.setWindowTitle('')

        self.label_Text = QLabel('')
        self.button_close = QPushButton('OK') 
        self.List=QLabel('')
        self.scrollArea=QScrollArea(self)
        
        if self.ListImageWrongSize==[]:
            self.label_Text.setText('Done ! \n All the outputs have been properly saved')
        else:
            ReferencePicture=cv2.imread(self.ReferencePicture) 
            sizefirstImage=np.shape(ReferencePicture)
            self.label_Text.setText("Done ! \n The following haven't been processed because their size is different from the image used for choosing the region of interest"+str(sizefirstImage)+"\n or because they were taken between 16:00 and 8:00: \n")
            for i in self.ListImageWrongSize:
                TestImageBGR=cv2.imread(i) 
                size=np.shape(TestImageBGR)
                self.List.setText(self.List.text()+str(i)+str(size)+'\n')
        
        self.scrollArea.setWidget(self.List)
                
        self.v_box=QVBoxLayout()
        self.v_box.addWidget(self.label_Text)
        self.v_box.addWidget(self.scrollArea)
        self.v_box.addWidget(self.button_close)

        self.setLayout(self.v_box)
        
        self.show()
        
        self.button_close.clicked.connect(self.close)
    
    def close(self):
        self.hide()
            
    
        
    
app = QApplication(sys.argv)
a_window = MainWindow()
app.exec_()


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################



