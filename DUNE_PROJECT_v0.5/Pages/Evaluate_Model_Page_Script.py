from Imports.common_imports import *

class Evaluate_Model_Page(tk.Frame):
    """A Tkinter frame for evaluating trained models on selected datasets.
        Probably need to add more description here later.
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Title and Navigation Frame
        title_frame = tk.Frame(self)
        title_frame.pack(fill='x', pady=(10, 5), padx=10)

        back_button = tk.Button( title_frame ,  text="Back" , command=lambda:( controller.attributes("-fullscreen", False )  , controller.show_frame("Training_And_Eval_Options_Page")) )
        back_button.pack(side='left')

        title_label = tk.Label(  title_frame  ,  text="Evaluate Model"  ,  font=("Helvetica", 16) )
        title_label.pack(side='left', padx=(20, 0))

        # Directory Selection Frame
        dir_frame_input_1 = tk.Frame(self)
        dir_frame_input_1.pack(fill='x', pady=(5, 10), padx=10)

        self.selected_dir_1 = tk.StringVar()
        dir_entry_1 = tk.Entry( dir_frame_input_1  ,  textvariable=self.selected_dir_1  ,   bg='black' ,  fg='white',  font=("Arial", 12) ,  width=90,  state='disabled' )
        dir_entry_1.pack(side='left', fill='x')
        self.selected_dir_1.trace_add('write', lambda *args: self.Add_Dir_Classes( self.selected_dir_1 ) )

        select_button = tk.Button( dir_frame_input_1  ,  text="Select Directory (1)" ,  command=lambda: self.controller.Frame_Manager.select_directory_window(self , dir_entry_1) )
        select_button.pack(side='left', padx=(5, 0))


        dir_frame_input_2 = tk.Frame(self)
        dir_frame_input_2.pack(fill='x', pady=(5, 10), padx=10)

        self.selected_dir_2 = tk.StringVar()
        dir_entry_2 = tk.Entry( dir_frame_input_2  ,  textvariable=self.selected_dir_2  ,   bg='black' ,  fg='white',  font=("Arial", 12) ,  width=90,  state='disabled' )
        dir_entry_2.pack(side='left', fill='x')

        select_button = tk.Button( dir_frame_input_2  ,  text="Select Directory (2)" ,  command=lambda: self.controller.Frame_Manager.select_directory_window(self , dir_entry_2) )
        select_button.pack(side='left', padx=(5, 0))

        dir_frame_input_3 = tk.Frame(self)
        dir_frame_input_3.pack(fill='x', pady=(5, 10), padx=10)

        self.selected_dir_3 = tk.StringVar()
        dir_entry_3 = tk.Entry( dir_frame_input_3  ,  textvariable=self.selected_dir_3  ,   bg='black' ,  fg='white',  font=("Arial", 12) ,  width=90,  state='disabled' )
        dir_entry_3.pack(side='left', fill='x')

        select_button = tk.Button( dir_frame_input_3  ,  text="Select Directory (3)" ,  command=lambda: self.controller.Frame_Manager.select_directory_window(self , dir_entry_3) )
        select_button.pack(side='left', padx=(5, 0))

        dropdown_frame = tk.Frame( self )
        dropdown_frame.pack(fill='x', pady=(10, 0), padx=10)

        self.Class_dropdown_var = tk.StringVar()
        self.Image_dropdown_var = tk.StringVar()

        tk.Label(dropdown_frame , text= 'Class : ' ).pack(side = tk.LEFT , anchor= 'w')
        self.Class_dropdown = ttk.Combobox( dropdown_frame  , textvariable= self.Class_dropdown_var , state='disabled', width= 30 )
        self.Class_dropdown.pack(side=tk.LEFT , anchor='w')
        self.Class_dropdown_var.trace_add('write', lambda *args: self.Add_Image_Options( self.selected_dir_1 ) )



        tk.Label(dropdown_frame , text= 'Image : ' ).pack(side = tk.LEFT , anchor= 'w')
        self.Image_dropdown = ttk.Combobox( dropdown_frame, textvariable = self.Image_dropdown_var , state='disabled' , width= 15 )
        self.Image_dropdown.pack(side=tk.LEFT , anchor='w')


        # Evaluation Options Frame
        eval_options_frame = tk.Frame(self)
        eval_options_frame.pack(fill='x', pady=(10, 0), padx=10)

        eval_options_frame_2 = tk.Frame(self)
        eval_options_frame_2.pack(fill='x', pady=(10, 0), padx=10)

        heatmap_button = tk.Button( eval_options_frame , state='normal' ,  text='Heatmap' , command= lambda : self.Heatmap_Pressed( path = [ os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ) ,  self.selected_dir_2.get() ,  self.selected_dir_3.get() ]  )  )
        heatmap_button.pack(side='left', padx=(0, 5))


        heatmap_button = tk.Button( eval_options_frame , state='disabled', text='Heatmap (Random 100)' , command= lambda : self.Heatmap_Pressed( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() ) , random_100 = True) )
        heatmap_button.pack(side='left', padx=(0, 5))

        correlation_button = tk.Button( eval_options_frame  ,state='normal' ,  text='Correlation' , command= lambda : self.Correlation_Pressed( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        correlation_button.pack(side='left')


        correlation_button = tk.Button( eval_options_frame  ,  text='Metrics' , command= lambda : self.Mertrics_Pressed( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        correlation_button.pack(side='left')

        Single_Pred_Button = tk.Button( eval_options_frame ,state='disabled' ,  text='Single Prediction' , command= lambda : self.Prob_image_func( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() , self.Image_dropdown_var.get() ))  )
        Single_Pred_Button.pack(side='left')


        correlation_button = tk.Button( eval_options_frame  ,  text='Input Usefulness' , command= lambda : self.Usefulness_Pressed( path = [self.selected_dir_1.get() ,
                                                                                                                                            self.selected_dir_2.get() ,
                                                                                                                                            self.selected_dir_3.get()  ]  , 
                                                                                                                                            image_branch =  os.path.join( self.Class_dropdown_var.get() , self.Image_dropdown_var.get()) )    )
        correlation_button.pack(side='left')

        Big_Pred_Button = tk.Button( eval_options_frame  ,state='disabled',  text='Class Probs' , command= lambda : self.Whole_Class_prediction( path = os.path.join( self.selected_dir_1.get() , self.Class_dropdown_var.get() ))  )
        Big_Pred_Button.pack(side='left')

        Deposite_button = tk.Button( eval_options_frame  ,  text='Image_to_Array' , command= lambda : self.Important_Regions( path = os.path.join( self.selected_dir_1.get()  )  ) ) 
        Deposite_button.pack(side='left')
        
        Energy_Hist_button = tk.Button( eval_options_frame  ,state='disabled',  text='Energy_Hist' , command= lambda : self.Energy_Prediction_Hist( path = os.path.join( self.selected_dir_1.get()  ))  )
        Energy_Hist_button.pack(side='left')

        Pixel_Hist_button = tk.Button( eval_options_frame_2  ,state='disabled',  text='Pixel_Count_Hist' , command= lambda : self.Pixel_Count_Prediction_Hist( path = os.path.join( self.selected_dir_1.get()  ))  )
        Pixel_Hist_button.pack(side='left')

        Pixel_PDF = tk.Button( eval_options_frame_2  ,state='disabled',  text='Download_Pixel_PDF' , command= lambda : self.Pixel_Predictions_PDF( path = os.path.join( self.selected_dir_1.get()  ))  )
        Pixel_PDF.pack(side='left')

        FP_PDF = tk.Button( eval_options_frame_2  ,state='disabled',  text='False_Positve_NES' , command= lambda : self.Scattering_Softmax_PDF( path = os.path.join( self.selected_dir_1.get()  ))  )
        FP_PDF.pack(side='left')

        FP_2_Class_PDF = tk.Button( eval_options_frame_2 ,state='disabled' ,  text='Two_Class_Threshold' , command= lambda : self.Two_Class_Binary_Threshold_PDF( path = os.path.join( self.selected_dir_1.get()  ))  )
        FP_2_Class_PDF.pack(side='left')

        Lepton_Angle_PDF = tk.Button( eval_options_frame_2 ,state='disabled' ,  text='Lepton_Angle' , command= lambda : self.Two_Class_Lepton_Angle( path = os.path.join( self.selected_dir_1.get()  ))  )
        Lepton_Angle_PDF.pack(side='left')

        FP_Hist = tk.Button( eval_options_frame_2  ,state='disabled',  text='FP Hist' , command= lambda : self.False_Positive_Hist( path = os.path.join( self.selected_dir_1.get()  ))  )
        FP_Hist.pack(side='left')

        FP_PDF = tk.Button( eval_options_frame_2  ,state='disabled',  text='FP PDF' , command= lambda : self.False_Positive_PDF( path = os.path.join( self.selected_dir_1.get()  ))  )
        FP_PDF.pack(side='left')


        self.Figure_Canvas_Frame = tk.Frame( self )
        self.Figure_Canvas_Frame.pack( anchor='w' , side= tk.TOP , pady= 10 )


    def Add_Dir_Classes(self , Entry_Box):
        possible_classes = sorted ( os.listdir( Entry_Box.get() ) ) 
        self.Class_dropdown_var.set('')
        self.Class_dropdown.config(state='readonly' , values= possible_classes )

        return

    def Add_Image_Options(self, Entry_Box):
        # When processing a selected directory, this function populates the image options based on the selected class.
        path = os.path.join( Entry_Box.get()  , self.Class_dropdown_var.get() )
        self.Image_dropdown_var.set('')
        possible_images = sorted ( os.listdir( path) ) 

        self.Image_dropdown.config(state='readonly' , values= possible_images)

    def Heatmap_Pressed(self , path  , random_100 = False):
        # Given a list of directory paths, this function generates heatmaps for the images in those directories.
        
        if random_100 == False:
            print( path )     
            self.controller.Heat_Map_Class.Heatmap_func( self , paths = path)

        else:
            print( path )
            self.controller.Heat_Map_Class.Heatmap_func( self , paths = path , random_100 = True)

        return  

    def Prob_image_func(self, path ):
        # Given a path to a single image, this function evaluates the model's probability predictions and visualizes them.
        print(path)
        self.controller.Evaluating_Model.Plot_Prob_Single_Image_func( self , path = path)


    def Usefulness_Pressed(self, path , image_branch ):
        # For multi-input models - Given a list of directory paths and an image branch, this function evaluates the input usefulness across different inputs.
        print(path)
        updated_paths = []
        for p in path:
            old_base = os.path.basename( os.path.normpath(  path[0] ) )
            new_base = os.path.basename( os.path.normpath(     p    ) )
            print( image_branch  )
            updated_paths.append( os.path.join( p , image_branch.replace( old_base , new_base ) ) ) 

        print( '\n', updated_paths)
        self.controller.Evaluating_Model.Plot_input_usefulness( self , image_paths = updated_paths )



    def Correlation_Pressed(self , path ):
        self.controller.Evaluating_Model.Correlation_func( self)

        return  
    

    def Mertrics_Pressed(self , path ):
        self.controller.Evaluating_Model.Plot_metrics_func( self)

        return  

    def Whole_Class_prediction(self , path):
        self.controller.Evaluating_Model.Whole_Class_Predictinos( self , path = path)

        return

    def Important_Regions(self, path):
        print( path )
        self.controller.Evaluating_Model.Detector_Important_Regions( self , path = path)

        return


    def Energy_Prediction_Hist(self, path):
        self.controller.Evaluating_Model.energy_prediction_distribution( self , path = path)

        return


    def Pixel_Count_Prediction_Hist(self, path):
        # THis function generates a histogram analyzing the distribution of active pixel counts in model predictions, whether ther are correct or incorrect
        self.controller.Evaluating_Model.Active_Pixel_Count( self , path = path)

        return


    def Pixel_Predictions_PDF(self, path):
        # This function generates and downloads a PDF report analyzing pixel-level model predictions.
        self.controller.Evaluating_Model.Download_Pixel_Eval_Plots( self , path = path)
        return


    def Scattering_Softmax_PDF(self, path):
        # This function generates and downloads a PDF report analyzing scattering model performance based on false positive rates.
        self.controller.Evaluating_Model.Scattering_False_Positve_Analysis( self , path = path)
        return


    def Two_Class_Binary_Threshold_PDF(self, path):
        # This function generates and downloads a PDF report analyzing two-class model performance based on varying binary classification thresholds.
        self.controller.Evaluating_Model.Two_Class_Scattering_False_Positve_Analysis( self , path = path)
        return

    def Two_Class_Lepton_Angle(self, path):
        # This function generates and downloads a PDF report analyzing lepton angle distributions for two-class models againts the neutrino energy spectrum.
        self.controller.Evaluating_Model.two_class_lepton_angel_PDF( self , path = path)

        return

    def False_Positive_Hist(self, path):
        # This function generates a histogram analyzing false positive predictions.
        self.controller.Evaluating_Model.FP_Positive_Prob_Hist( self , path = path)
        return

    def False_Positive_PDF(self, path):
        # This function generates and downloads a PDF report analyzing false positive predictions.
        self.controller.Evaluating_Model.New_Comparison_Func( self , path = path)

        return
