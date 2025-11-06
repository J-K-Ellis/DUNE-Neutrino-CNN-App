from Imports.common_imports import *

class Model_Tuning_Page(tk.Frame):
    """A Tkinter frame for setting model tuning conditions."""
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.prev_different_classes = None
        self.conditions = []

        page_title_frame = tk.Frame(self)
        page_title_frame.pack(anchor='w', pady=10)

        Extra_Frame = tk.Frame(self)
        Extra_Frame.pack(anchor='w', pady=(100, 40))

        self.Multiple_Initialisations = tk.BooleanVar(value=False)
        self.K_Fold_Cross_Validation_Option = tk.BooleanVar(value=False)

        Advance_Page = self.controller.frames.get("Advance_Class_Selection_Page")

        tk.Button( page_title_frame, text="Back", command=lambda: (self.controller.attributes("-fullscreen", False), controller.show_frame("Model_Training_Page")) ).pack(anchor='w', padx=10, side=tk.LEFT)

        tk.Button( page_title_frame, text="Update Page", command=lambda: self.Update_Tuning_Page(Advance_Page_var=Advance_Page) ).pack(anchor='w', padx=10, side=tk.LEFT)

        tk.Label(page_title_frame, text="Model Tuning ", font=("Helvetica", 16)).pack( padx=50, anchor='w', side=tk.LEFT )

        tk.Button(page_title_frame, text="Show Conditions", command= lambda: ( self.test_function() ) ).pack( anchor='w', padx=10, side=tk.LEFT )


        if Advance_Page is not None:
            tk.Button( page_title_frame, text="Go to Advance Class Selection", command=lambda: (self.controller.attributes("-fullscreen", False), controller.show_frame("Advance_Class_Selection_Page"))  ).pack(anchor='w', padx=10, side=tk.LEFT)

        tk.Label(Extra_Frame, text="Condition Variable:").pack(anchor='w', side=tk.LEFT)
        self.Condition_Varible_Dropdown = ttk.Combobox(Extra_Frame, values=[], state="readonly")
        self.Condition_Varible_Dropdown.pack(anchor='w', side=tk.LEFT, padx=10)
        self.Condition_Varible_Dropdown.bind("<<ComboboxSelected>>", self._on_condition_var_selected)

        # Scrollable container for conditions
        container = tk.Frame(self)
        container.pack(fill='both', expand=True, pady=10)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind( "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")) )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Add/remove condition buttons
        button_frame = tk.Frame(self)
        button_frame.pack(anchor='w', pady=5)
        tk.Button(button_frame, text="Add Condition", command=self._add_condition_row).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Remove Last Condition", command=self._remove_condition_row).pack(side=tk.LEFT, padx=5)

    
    
    def Update_Tuning_Page(self, Advance_Page_var):
        """ Function to update the Model Tuning Page based on the Advance Class Selection Page.
        Args:
            Advance_Page_var: The instance of the Advance_Class_Selection_Page frame.
        Returns:
            None
        """
        # Update the condition variable dropdown based on selected classes from Advance_Class_Selection_Page
        if not hasattr(Advance_Page_var, "Page_Activated"):
            print("Advance page not initialized.")
            return

        if Advance_Page_var.Page_Activated:
            current_classes = getattr(Advance_Page_var, "different_classes", [])
            print(current_classes)

            # Reset all conditions if different_classes changed
            if current_classes != self.prev_different_classes:
                self._clear_all_conditions()
                self.prev_different_classes = list(current_classes)

            self.Condition_Varible_Dropdown["values"] = current_classes
        else:
            print("No Training Set Directory has been selected")


    def test_function(self):
        """ Function to print the current conditions for debugging purposes."""
        for idx, (row, cond_var, entry_var) in enumerate(self.conditions):
            print(f"Condition {idx + 1}: Variable = {cond_var.get()}, Entry = {entry_var.get()}")


    def _on_condition_var_selected(self, event=None):
        """Callback when a condition variable is selected from the dropdown.
        Args:
            event: The Tkinter event object (not used).
        Returns:
            None"""
        # Ensure a variable is selected before adding a condition row
        if not self.Condition_Varible_Dropdown.get():
            return
        self._add_condition_row()

    def _add_condition_row(self):
        """Add a new condition row to the scrollable frame."""
        # Create a new row for conditions in the scrollable frame
        row = tk.Frame(self.scrollable_frame)
        row.pack(anchor='w', pady=2, fill='x')

        var_label = tk.Label(row, text=self.Condition_Varible_Dropdown.get() , width=25, anchor='w')
        var_label.pack(side=tk.LEFT, padx=5)

        cond_var = tk.StringVar()
        cond_dropdown = ttk.Combobox(row, values=["Incorect Prediction" , "Correct Prediction","Overall Confidence Cut", "Correct Confidence Cut", "Incorrect Confidence Cut"], textvariable=cond_var, state="readonly", width=20)
        cond_dropdown.pack(side=tk.LEFT, padx=5)

        entry_var = tk.StringVar()
        cond_entry = tk.Entry(row, textvariable=entry_var, width=10)
        cond_entry.pack(side=tk.LEFT, padx=5)
        cond_entry.pack_forget()  # hide until selection made

        def on_dropdown_select(event=None):
            """Callback when a condition type is selected from the dropdown."""
            cond_entry.pack(side=tk.LEFT, padx=5)

        cond_dropdown.bind("<<ComboboxSelected>>", on_dropdown_select)

        self.conditions.append((row, cond_var, entry_var))

    def _remove_condition_row(self):
        """Remove the last condition row from the scrollable frame."""
        if not self.conditions:
            return
        row, _, _ = self.conditions.pop()
        row.destroy()

    def _clear_all_conditions(self):
        """Clear all condition rows from the scrollable frame."""
        for row, _, _ in self.conditions:
            row.destroy()
        self.conditions.clear()

