from Imports.common_imports import *
from Helpers.Frame_Manager_Script import Frame_Manager

class View_Segments_Page(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller  # Reference to the main app
        self.h5_data_name = 'segments'


        self.Event_ID_selection = 0
        self.Event_IDs = []


        title_frame = tk.Frame(self)
        title_frame.pack(anchor='w')

        # "Back" button to return to the Dataset_View_Page
        tk.Button(title_frame, text="Back", command=lambda: controller.show_frame("Dataset_View_Page")).pack(anchor='w', side= tk.LEFT, padx=10 , pady=(0, 10))
        # Page title label
        tk.Label(title_frame, text="View Segments", font=("Helvetica", 16)).pack(anchor='w', side= tk.LEFT ,pady=(10, 10))

        # Create file selection dropdown frame.
        file_selection_frame = tk.Frame(self)
        file_selection_frame.pack(anchor='w', pady=(0, 10))


        tk.Label(file_selection_frame, text="Select file:").pack(side=tk.LEFT, padx=(0, 5))

        # Dropdown menu to select files to view segments data from.
        self.selected_file = tk.StringVar()
        self.files_drop_down = tk.OptionMenu(file_selection_frame, self.selected_file, "")
        self.files_drop_down.pack(side=tk.LEFT)


        self.display_frame = tk.Frame(self)
        self.display_frame.pack(anchor='w', pady=(10, 10))


        # create one manager instance for this page
        self.manager = Frame_Manager(self)

        # now populate the dropdown
        self.manager.update_dropdown()

        navigation_buttons_frame = tk.Frame(self)
        navigation_buttons_frame.pack(anchor='w', pady=(5, 10))


        # navigation buttons "Back" and "Next" to show the previous/next event.
        self.back_button = tk.Button( navigation_buttons_frame, text="Back",  command=lambda: self.manager.go_back(self) )
        self.back_button.pack(side=tk.LEFT, padx=5)


        self.next_button = tk.Button( navigation_buttons_frame, text="Next",  command=lambda: self.manager.go_next(self)  )
        self.next_button.pack(side=tk.LEFT, padx=5)

        # label to show the current event number and total events.
        self.event_counter_label = tk.Label(self, text="Event 0 of 0", font=("Helvetica", 10))
        self.event_counter_label.pack(anchor='w', padx=5)

        # link the selection variable to the handler. In this case, h5py.File(File_path)['segments'] would be the location for the required data.
        self.selected_file.trace_add('write', lambda *args: self.manager.on_file_selected( 'segments'))
