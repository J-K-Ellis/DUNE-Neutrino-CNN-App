from Imports.common_imports import *


# Define a class Dataset_View_Page, which holds buttons to navigate to different dataset viewing pages.
class Dataset_View_Page(tk.Frame):
    "'A Tkinter frame for viewing different datasets."
    def __init__(self, parent, controller):
        """Initialize the Dataset View Page frame.
        Args:       
            parent: The parent Tkinter widget.
            controller: The main application controller for managing frames and data.
        Returns:
            None
        """
        super().__init__(parent)

        # Create a frame for the title and back button
        title_frame = tk.Frame(self)
        title_frame.pack(anchor='w', pady=5 )

        # "Back" button to return to the StartPage
        tk.Button(title_frame, text="Back", command=lambda: controller.show_frame("StartPage")).pack(anchor='w', side=tk.LEFT, pady=2)

        # Page title label
        tk.Label(title_frame, text="View Datasets", font=("Helvetica", 16)).pack(  anchor='w' , side=tk.LEFT, pady=10, padx=10,)

        # Button to navigate to the "View Segments" page
        tk.Button(self, text="View segments", command=lambda: controller.show_frame("View_Segments_Page")).pack( anchor='w')

        # Button to navigate to the "View mc_hdr" page
        tk.Button(self, text="View mc_hdr", command=lambda: controller.show_frame("View_mc_hdr_Page") ).pack( anchor='w')

        # Button to navigate to the "View Trajectories" page
        tk.Button(self, text="View Trajectories", command=lambda: controller.show_frame("View_traj_Page")  ).pack( anchor='w' )
