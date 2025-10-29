from Imports.common_imports import *

class Advance_Class_Selection_Page(tk.Frame):


    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller  = controller
        self.style       = ttk.Style()
        self.update_flag = tk.BooleanVar(value=False)
        self.test_set_dir = None 

        self.Enable_Value = tk.BooleanVar( value = True)
        self.Epochs_Before_Refresh = tk.IntVar( value= 0)
        self.Enable_Class_Weights = tk.BooleanVar(value=True)
        self.Balance_Batches = tk.BooleanVar(value=True)

        self.test_set_status = tk.StringVar(value="Test sets: 0/3 selected")


        self.different_classes = []
        self.leaf_class_counts = {}

        self.custom_groups =  {}

        self.slider_sections = {}
        self.alloc_vars      = {}

        self.clear_page()   



    def Update_Page_With_Class(self, dir_path = None ):
        # Load classes from dir_path or previously selected directory

        dir_path = dir_path or self.controller.selected_directory
        if not dir_path or not os.path.isdir(dir_path):
            tk.messagebox.showerror("Error", "Please pick a valid dataset directory.")
            return

        self.clear_page()

        self.different_classes = sorted( d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) )
        self.leaf_class_counts = {c: len(os.listdir(os.path.join(dir_path, c))) for c in self.different_classes}


        if not self.custom_groups:
            self.custom_groups = {c: [c] for c in self.different_classes}

        # ─ header row ─
        hdr = tk.Frame(self); hdr.pack(fill='x', pady=10, padx=10)
        tk.Button(hdr, text='Back', command=lambda: [self.controller.attributes('-fullscreen', False), self.controller.show_frame("Model_Training_Page")] ).pack(side='left')
        tk.Label(hdr, text='Advanced Training Configuration', font=('Helvetica', 16)).pack(side='left', padx=40)
        tk.Button(hdr, text='Full', command=lambda: self.controller.attributes( '-fullscreen', not self.controller.attributes('-fullscreen')) ).pack(side='left')

        tk.Button(hdr, text='Select Test Sets', command=self.select_test_directory).pack(side='left')
        tk.Label(hdr, textvariable=self.test_set_status).pack(side='left', padx=8)

        # ─ allocation 
        self.build_allocation_row(parent=self)

        # split windows
        main = tk.Frame(self); main.pack(fill='both', expand=True)
        self.extra_column  = tk.Frame(main, relief='groove', bd=1)
        self.slider_column = tk.Frame(main)
        self.extra_column.pack(side='left', fill='both', expand=True, padx=(0, 5))
        self.slider_column.pack(side='left', fill='y')

        self.build_aggregate_ui()
        # initial sliders use only leaves
        self.create_slider_panes(classes=self.different_classes, counts=self.leaf_class_counts)


    # ──────────────────────────────────────────────────────────────────────────
    def build_aggregate_ui(self):
        # Everything inside self.extra_column.

        tk.Label(self.extra_column, text="Create / Edit Class", font=("Helvetica", 12, 'bold')).pack(anchor='w', pady=(8,4), padx=8)


        name_row = tk.Frame(self.extra_column); name_row.pack(anchor='w', pady=4, padx=10)
        tk.Label(name_row, text="Class name:").pack(side='left')
        self.new_group_name = tk.StringVar()
        tk.Entry(name_row, textvariable=self.new_group_name, width=24  ).pack(side='left', padx=(4,0))


        list_frame = self.controller.ScrollableFrame_2(self.extra_column)
        list_frame.pack(fill='both', expand=True, padx=10, pady=(2,8))
        self.leaf_vars: dict[str, tk.BooleanVar] = {}
        for cls in self.different_classes:
            var = tk.BooleanVar(value=False)
            self.leaf_vars[cls] = var
            ttk.Checkbutton(list_frame.scrollable_frame, text=cls, variable=var  ).pack(anchor='w', pady=1)


        btn_row = tk.Frame(self.extra_column); btn_row.pack(anchor='w', pady=4, padx=10)
        tk.Button(btn_row, text="Add / Update", command=self.add_or_update_group ).pack(side='left')
        tk.Button(btn_row, text="Remove Selected", command=self.remove_group ).pack(side='left', padx=6)
        tk.Button(btn_row, text="Remove All", command=self.remove_all ).pack(side='left', padx=6)

        # —— listbox of existing aggregates
        tk.Label(self.extra_column, text="Current Classes:").pack(anchor='w', padx=8)
        self.group_listbox = tk.Listbox(self.extra_column, height=5)
        self.group_listbox.pack(fill='x', padx=10, pady=(0,10))
        self.refresh_group_listbox()

    def build_allocation_row(self, parent):
        self.alloc_vars = {n: tk.DoubleVar(value=40 if n != 'Test' else 20) for n in ('Train', 'Validate', 'Test')}

        row = ttk.Frame(parent); row.pack(anchor='w', padx=10, pady=(4, 8))

        for i, name in enumerate(('Train', 'Validate', 'Test')):
            ttk.Label(row, text=f'{name}:').grid(row=0, column=2 * i, sticky='e')
            ttk.Entry(row, textvariable=self.alloc_vars[name], width=6,  validate='key', validatecommand=(self.register(self.validate_digit), '%P')  ).grid(row=0, column=2 * i + 1, padx=(0, 10))

        # “Load Classes” button sits after the Test entry
        ttk.Button(row, text='Load Classes', command=self.reload_slider_panes ).grid(row=0, column=6, padx=(10, 0))

        tk.Label( row , text= "Enable:" ).grid( row=0, column=7, padx=(10, 0))
        self.Enable_Value_Button = ttk.Checkbutton( row  , variable= self.Enable_Value , onvalue= True, offvalue= False)
        self.Enable_Value_Button.grid(row=0, column=8 )

        tk.Label( row , text= "Class Weights:" ).grid( row=0, column=11, padx=(10, 0))
        self.Enable_Value_Button = ttk.Checkbutton( row  , variable= self.Enable_Class_Weights,  onvalue= True, offvalue= False ,command= lambda: print( self.Enable_Class_Weights.get() ) )
        self.Enable_Value_Button.grid(row=0, column=13 )

        tk.Label( row , text = "Balance Batches:" ).grid( row=0, column=14, padx=(10, 0))
        self.Balance_Batches_Button = ttk.Checkbutton( row , variable= self.Balance_Batches ,  onvalue= True, offvalue= False )
        self.Balance_Batches_Button.grid(row=0, column=15 )


        ttk.Label(row, text='Epochs Before Refresh:').grid( row=0, column=17, sticky='e', padx=(5, 1))
        ttk.Entry(row, textvariable=self.Epochs_Before_Refresh, width=6).grid(row=0, column=18, padx=2)

        

        self.error_label = ttk.Label(row, text="", foreground="red")
        self.error_label.grid( row = 0 , column= 10 )

    def create_slider_panes(self, *, classes: list[str], counts: dict[str,int]):
        # destroy old
        for child in self.slider_column.winfo_children():
            child.destroy()
        self.slider_sections.clear()

        pct = {n: v.get()/100 for n,v in self.alloc_vars.items()}
        for sec in ("Train","Validate","Test"):
            self.slider_sections[sec] = self.controller.SliderSection(
                self.slider_column, sec, tk.DoubleVar(value=100), self.update_flag)

        for cls in classes:
            cnt = counts[cls]
            for sec in ("Train","Validate","Test"):
                self.slider_sections[sec].add_slider(  slider_label=cls, max_value=cnt*pct[sec],  initial_value=cnt*pct[sec], original_count=cnt )

        self.active_classes = classes[:]
        self.check_entries_sum()      # reuse existing validation logic


    def add_or_update_group(self):
        name  = self.new_group_name.get().strip()
        leaves= [cls for cls,var in self.leaf_vars.items() if var.get()]

        # —— validation 
        if not name:
            messagebox.showwarning("Validation", "Class name cannot be empty.")
            return
        if not leaves:
            messagebox.showwarning("Validation", "You must select at least one underlying class.")
            return
        if  (name in self.custom_groups and set(leaves)==set(self.custom_groups[name])):
            messagebox.showwarning("Validation", "Duplicate class name.")
            return

        # store
        self.custom_groups[name] = leaves
        self.refresh_group_listbox()
        self.clear_leaf_checks()

    def remove_group(self):
        sel = self.group_listbox.curselection()
        if not sel:
            return
        name = self.group_listbox.get(sel[0])
        if name in self.custom_groups:
            del self.custom_groups[name]
            self.refresh_group_listbox()

    def remove_all(self):

        # print(self.slider_sections)
        print(self.leaf_class_counts)
        print(self.custom_groups)
        listy = list(self.group_listbox.get(0 , tk.END))
        # print( listy )
        for name in listy:
            del self.custom_groups[name]

        self.refresh_group_listbox()
        


    def refresh_group_listbox(self):
        self.group_listbox.delete(0, tk.END)
        for name in sorted(self.custom_groups):
            self.group_listbox.insert(tk.END, name)

    def clear_leaf_checks(self):
        for var in self.leaf_vars.values():
            var.set(False)

    def reload_slider_panes(self):
        names  = sorted(self.custom_groups.keys())
        counts = {n: sum(self.leaf_class_counts[l] for l in leaves) for n, leaves in self.custom_groups.items() }
        self.create_slider_panes(classes=names, counts=counts)

        print(names)



    def select_test_directory(self):
        # Opens dialog to select up to 3 test set directories

        self.first_dir = None
        # Use previous selection if any, else 3x None
        prev = self.test_set_dir if isinstance(self.test_set_dir, list) else [None, None, None]

        top = tk.Toplevel(self)
        top.title("Select up to 3 Test Set Directories")
        top.transient(self.winfo_toplevel())
        top.grab_set()
        top.resizable(False, False)

        ttk.Label(top, text="Pick up to three directories. "
                            "Directory 1 is used for per-class image counts.").pack(anchor='w', padx=10, pady=(10, 6))

        path_vars = [tk.StringVar(value=p or "") for p in prev]

        def choose(i: int):
            d = tk.filedialog.askdirectory(title=f"Select Test Set Directory #{i+1}", initialdir=os.getcwd())
            if not d:
                return
            # Verify required class subfolders exist
            missing = [c for c in self.different_classes if not os.path.isdir(os.path.join(d, c))]
            if missing:
                messagebox.showerror("Error", f"Missing classes in directory #{i+1}: {missing}")
                return
            path_vars[i].set(d)

        for i in range(3):
            row = ttk.Frame(top); row.pack(fill='x', padx=10, pady=(6 if i == 0 else 4, 0))
            ttk.Label(row, text=f"Directory {i+1}:").pack(side='left')
            ttk.Entry(row, textvariable=path_vars[i], width=54, state='readonly').pack(side='left', padx=6)
            ttk.Button(row, text="Browse…", command=lambda i=i: choose(i)).pack(side='left')

        btns = ttk.Frame(top); btns.pack(fill='x', padx=10, pady=10)

        def clear_all():
            for v in path_vars:
                v.set("")
        ttk.Button(btns, text="Clear", command=clear_all).pack(side='left')

        def on_ok():

            paths = [v.get() or None for v in path_vars]

            # Must have Directory 1 if anything is selected; counts come from it
            if not paths[0]:
                if any(paths[1:]):
                    messagebox.showwarning("Validation", "Please select Directory 1 (used for class counts).")
                    return
                top.destroy()
                return

            for idx, p in enumerate(paths):
                if not p:
                    continue
                missing = [c for c in self.different_classes if not os.path.isdir(os.path.join(p, c))]
                if missing:
                    messagebox.showerror("Error", f"Missing classes in directory #{idx+1}: {missing}")
                    return

            self.test_set_dir = paths  # e.g., ['path/dir1', 'path/dir2', None]

            self.style.configure("TestSet.TLabelframe.Label", foreground="red")
            if "Test" in self.slider_sections:
                self.slider_sections["Test"].frame.configure(style="TestSet.TLabelframe")

                self.first_dir = paths[0]
                sec = self.slider_sections["Test"]
                for i, cls in enumerate(self.different_classes):
                    cnt = len(os.listdir(os.path.join(self.first_dir, cls)))
                    sec.slider_max_values[i] = cnt
                    sec.slider_widgets[i].config(to=cnt)
                    sec.slider_vars[i].set(cnt)
                    # keep the percent labels in sync if your SliderSection exposes this helper
                    if hasattr(sec, "_update_percentage"):
                        sec._update_percentage(sec.slider_vars[i], i)


            selected_count = sum(1 for p in paths if p)
            self.test_set_status.set(f"Test sets: {selected_count}/3 selected")

            # run validation
            self.check_entries_sum()
            top.destroy()
            print( self.test_set_dir  )

        def on_cancel():
            top.destroy()

        ttk.Button(btns, text="OK", command=on_ok).pack(side='right')
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side='right', padx=(0, 6))


        self.style.configure("TestSet.TLabelframe.Label", foreground="red")
        self.slider_sections["Test"].frame.configure(style="TestSet.TLabelframe")

        for i, cls in enumerate(self.different_classes):
            cnt = len(os.listdir(os.path.join(self.first_dir, cls)))
            sec = self.slider_sections["Test"]
            sec.slider_max_values[i] = cnt
            sec.slider_widgets[i].config(to=cnt)
            sec.slider_vars[i].set(cnt)



    def train_slider_changed(self, idx, *args):
        if self.slider_sections["Train"].slider_vars[idx].get()==0:
            self.slider_sections["Validate"].slider_vars[idx].set(0)
            self.slider_sections["Test"].slider_vars[idx].set(0)

    def check_entries_sum(self):
        if self.update_flag.get(): return
        total = sum(v.get() for v in self.alloc_vars.values())
        if abs(total-100)>1e-6:
            self.error_label.config(text="Train + Validate + Test must sum to 100!")
            for sec in self.slider_sections.values():
                sec.allocated_var.set(0)
                for row in sec.sliders if hasattr(sec,'sliders') else []:
                    for ch in row.winfo_children(): ch.configure(state='disabled')
        else:
            self.error_label.config(text="")
            for name,sec in self.slider_sections.items():
                sec.allocated_var.set(100)
                for row in sec.sliders if hasattr(sec,'sliders') else []:
                    for ch in row.winfo_children(): ch.configure(state='normal')
                pct = self.alloc_vars[name].get()/100
                for i,orig in enumerate(sec.slider_original_counts):
                    if name=='Test' and self.test_set_dir:
                        new_max = sec.slider_max_values[i]
                    else:
                        new_max = orig*pct
                    sec.slider_max_values[i]=new_max
                    if sec.slider_vars[i].get()>new_max: sec.slider_vars[i].set(new_max)
                    sec.slider_widgets[i].config(to=new_max)
                    sec._update_percentage(sec.slider_vars[i], i)

    def validate_digit(self, nv):
        if nv=="": return True
        try: float(nv); return True
        except ValueError: return False

    def clear_page(self):
        for w in self.winfo_children():
            w.destroy()


        
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#