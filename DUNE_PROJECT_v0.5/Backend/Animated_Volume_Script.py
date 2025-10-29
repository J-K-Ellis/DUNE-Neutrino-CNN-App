from Imports.common_imports import *

class Animated_Volume:

    # Class to run a 3D animation on a Tkinter Frame.

    def __init__(self,parent_frame : tk.Frame, h5_file_path:str, event_id:int, vertex_id:int ,energy_cut = 1.5, playback_speed = 1.0, fade_duration = 2.0,interval_ms= 100 , limit_dict = {},  animation_page = None , controller = None , extra_frame = None) :

        self.parent_frame = parent_frame
        self.extra_frame = extra_frame

        self.h5_path = h5_file_path
        self.event_id = event_id
        self.vertex_id = vertex_id
        self.energy_cut = energy_cut
        self.playback_speed = playback_speed
        self.fade_duration = fade_duration
        self.interval_ms = interval_ms
        self.animation_page = animation_page
        self.controller = controller

        self.min_x_for_plot = limit_dict['min_x']
        self.max_x_for_plot = limit_dict['max_x']
        self.min_y_for_plot = limit_dict['min_y']
        self.max_y_for_plot = limit_dict['max_y']
        self.min_z_for_plot = limit_dict['min_z']
        self.max_z_for_plot = limit_dict['max_z']


        self.dE_min = 0.0
        self.dE_max = 75.0

        self.current_rotation_angle = 0

        self.logged_predictions = {}
        self.logged_heatmaps = {}
        self.predict_fn = None 


        self.df = None
        self.df_mc_hdr = None
        
        self.params = None
        self._visible_indices = []

        # These will be set during setup
        self.fig = None
        self.ax = None
        self.canvas_widget = None
        self.ani = None

        self.extra_fig = None
        self.extra_ax = None
        self.extra_canvas_widget = None



    def _load_dataframe(self) -> pd.DataFrame:

        with h5py.File(self.h5_path, "r") as hf:
            segments = hf["segments"][()]

        df = pd.DataFrame(segments)
        # Filter on event_id and vertex_id
        # df = df[(df["event_id"] == self.event_id) & (df["vertex_id"] == self.vertex_id)].copy()

        # Apply energy cut
        df = df[df["dE"] > self.energy_cut].copy()

        # Ensure t0_end >= t0_start
        df["t0_end"] = np.maximum(df["t0_start"], df["t0_end"])

        if df.empty:
            raise ValueError(  f"No hits found for event {self.event_id}, vertex {self.vertex_id} with dE > {self.energy_cut}." )

        return df.reset_index(drop=True)
    


    def _load_mc_hdr(self):
        df_mc_hdr = pd.DataFrame.from_records( h5py.File(self.h5_path)['mc_hdr'], columns=np.dtype(  h5py.File(self.h5_path)['mc_hdr'] ).names )

        return df_mc_hdr

    def _make_cuboid_faces(self, x0: float, y0: float, z0: float, dx: float, dy: float, dz: float):

        # Build one cuboid (6 faces) in plot-coordinates (Z, X, Y).

        # eight corners in original coords (x_orig, y_orig, z_orig):
        v000 = (x0, y0, z0)
        v100 = (x0 + dx, y0, z0)
        v110 = (x0 + dx, y0, z0 + dz)
        v010 = (x0, y0, z0 + dz)
        v001 = (x0, y0 + dy, z0)
        v101 = (x0 + dx, y0 + dy, z0)
        v111 = (x0 + dx, y0 + dy, z0 + dz)
        v011 = (x0, y0 + dy, z0 + dz)

        original_faces = [
            [v000, v100, v110, v010],  # bottom (y = y0)
            [v001, v011, v111, v101],  # top    (y = y0 + dy)
            [v000, v001, v101, v100],  # front  (z = z0)
            [v010, v110, v111, v011],  # back   (z = z0 + dz)
            [v100, v101, v111, v110],  # right  (x = x0 + dx)
            [v000, v010, v011, v001],  # left   (x = x0)
        ]

        faces_plot = []
        for face in original_faces:
            face_plot = []
            for (xo, yo, zo) in face:
                # Map original coords to plotting coords:
                x_plot = zo        
                y_plot = xo      
                z_plot = yo       
                face_plot.append((x_plot, y_plot, z_plot))
            faces_plot.append(face_plot)

        return faces_plot

    def _initialize_plot(self, df: pd.DataFrame):

        # Create figure & 3D axes
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")



        NX, NZ = 5, 7                         # number of modules along X and Z
        DX = (self.max_x_for_plot - self.min_x_for_plot) / NX      # 700 / 5  = 140.0
        DY = (self.max_y_for_plot - self.min_y_for_plot )           # ≈300
        DZ = (self.max_z_for_plot - self.min_z_for_plot) / NZ      # ≈71.77

        # dE_min = 0.0
        # dE_max = 75.0

        self.ax.set_xlim(self.min_z_for_plot, self.max_z_for_plot)
        self.ax.set_ylim(self.min_x_for_plot, self.max_x_for_plot)
        self.ax.set_zlim(self.min_y_for_plot, self.max_y_for_plot)

        self.ax.set_xlabel("Z")
        self.ax.set_ylabel("X")
        self.ax.set_zlabel("Y")

        # Build the grid of 35 cuboids (5 × 7 modules)
        all_faces = []
        x0_base = self.min_x_for_plot
        y0_base = self.min_y_for_plot
        z0_base = self.min_z_for_plot


        for i in range(NX):
            for j in range(NZ):
                x0 = x0_base + i * DX
                y0 = y0_base
                z0 = z0_base + j * DZ
                faces = self._make_cuboid_faces(x0, y0, z0, DX, DY, DZ)
                all_faces.extend(faces)

        cuboid_collection = Poly3DCollection( all_faces, facecolors=(0.8, 0.8, 0.8, 0.2),edgecolors="red",linewidths=0.8 ) 
        self.ax.add_collection3d(cuboid_collection)

        # Prepare empty scatter
        self.scatter = self.ax.scatter([], [], [], c=[], marker="o", s=7)

        # Title placeholder
        self.title = self.ax.set_title("")

        # Embed figure into Tk frame
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)



    def _compute_animation_params(self, df: pd.DataFrame):

        # Given df sorted by t0_start, compute arrays and frame count for animation.

        t_start_arr = df["t0_start"].to_numpy()
        t_end_arr = df["t0_end"].to_numpy()
        x_arr = df["x"].to_numpy()
        y_arr = df["y"].to_numpy()
        z_arr = df["z"].to_numpy()
        dE_arr = df["dE"].to_numpy()

        norm = mcolors.Normalize(vmin= self.dE_min, vmax= self.dE_max)
        cmap = cm.plasma

        t_min = t_start_arr.min()
        t_max = t_end_arr.max()
        t_final = t_max + self.fade_duration

        dt_real = self.interval_ms / 1000.0
        dt_data = self.playback_speed * dt_real
        total_duration = t_final - t_min
        num_frames = int(np.ceil(total_duration / dt_data)) + 1

        return {"t_start_arr": t_start_arr, "t_end_arr": t_end_arr, "x_arr": x_arr, "y_arr": y_arr, "z_arr": z_arr, "dE_arr": dE_arr, "norm": norm, "cmap": cmap, "t_min": t_min, "num_frames": num_frames, "dt_data": dt_data}



    def _animate_update(self, frame_idx: int, params: dict):

        t_current = params["t_min"] + frame_idx * params["dt_data"]

        t_start_arr = params["t_start_arr"]
        t_end_arr = params["t_end_arr"]
        x_arr = params["x_arr"]
        y_arr = params["y_arr"]
        z_arr = params["z_arr"]
        dE_arr = params["dE_arr"]
        norm = params["norm"]
        cmap = params["cmap"]

        if self.fade_duration > 0:
            mask_alive = ( (t_current >= t_start_arr) & (t_current <= (t_end_arr + self.fade_duration)) )
        else:
            mask_alive = ( (t_current >= t_start_arr) & (t_current <= t_end_arr) )

        if not mask_alive.any():
            # No visible points
            self._visible_indices = []

            self.scatter._offsets3d = ([], [], [])
            self.scatter.set_facecolors([])
            self.scatter.set_edgecolors([])
            self.title.set_text(f"t = {t_current:.3f}")
            return (self.scatter, self.title)

        idxs = np.where(mask_alive)[0]
        alphas = np.zeros_like(idxs, dtype=float)

        for out_i, i in enumerate(idxs):
            ts = t_start_arr[i]
            te = t_end_arr[i]
            if t_current < ts:
                alpha = 0.0
            elif ts <= t_current <= te:
                alpha = 1.0
            else:  # t_current > te
                if self.fade_duration > 0:
                    alpha = 1.0 - (t_current - te) / self.fade_duration
                    alpha = np.clip(alpha, 0.0, 1.0)
                else:
                    alpha = 0.0
            alphas[out_i] = alpha

        keep = (alphas > 0.0)
        if keep.any():
            idxs = idxs[keep]
            alphas = alphas[keep]

            idxs_visible = idxs
            self._visible_indices = idxs_visible

            xs_orig = x_arr[idxs]
            ys_orig = y_arr[idxs]
            zs_orig = z_arr[idxs]

            xs_plot = zs_orig
            ys_plot = xs_orig
            zs_plot = ys_orig

            dE_vals = dE_arr[idxs]
            normed = norm(dE_vals)
            rgba = cmap(normed)[:] 
            rgba[:, 3] = alphas

            self.scatter._offsets3d = (xs_plot, ys_plot, zs_plot)
            self.scatter.set_facecolors(rgba)
            self.scatter.set_edgecolors(rgba)
        else:

            self._visible_indices = [] 
            self.scatter._offsets3d = ([], [], [])
            self.scatter.set_facecolors([])
            self.scatter.set_edgecolors([])

        self.title.set_text(f"t = {t_current:.3f}")
        return (self.scatter, self.title)


    def _animate_update_roations(self, frame_idx: int, params: dict):

        t_current = frame_idx * params["dt_data"]

        x_arr = params["x_arr"]
        y_arr = params["y_arr"]
        z_arr = params["z_arr"]
        dE_arr = params["dE_arr"]

        self._visible_indices = np.arange(len(x_arr))

        angle = self.current_rotation_angle
        rad = np.deg2rad(angle)


        dx = x_arr - self.pivot_x
        dy = y_arr - self.pivot_y
        new_x = dx * np.cos(rad) - dy * np.sin(rad) + self.pivot_x
        new_y = dx * np.sin(rad) + dy * np.cos(rad) + self.pivot_y
        new_z = z_arr 

        xs_plot = new_z
        ys_plot = new_x
        zs_plot = new_y


        normed = params["norm"](dE_arr)
        colors = params["cmap"](normed)
        colors[:, 3] = 1.0


        self.scatter._offsets3d = (xs_plot, ys_plot, zs_plot)
        self.scatter.set_facecolors(colors)
        self.scatter.set_edgecolors(colors)


        self.current_rotation_angle = (angle + 10 * params["dt_data"]) % 360
        self.title.set_text(f"t = {t_current:.3f}s   rot = {self.current_rotation_angle:.1f}°")

        return (self.scatter, self.title)



    def _render_volume_at_angle(self, angle_deg: float, params: dict):
        """Render the full volume (all hits) at a fixed z-axis rotation angle."""
        x_arr = params["x_arr"]; y_arr = params["y_arr"]; z_arr = params["z_arr"]; dE_arr = params["dE_arr"]
        self._visible_indices = np.arange(len(x_arr))

        rad = np.deg2rad(angle_deg)
        dx  = x_arr - self.pivot_x
        dy  = y_arr - self.pivot_y
        new_x = dx * np.cos(rad) - dy * np.sin(rad) + self.pivot_x
        new_y = dx * np.sin(rad) + dy * np.cos(rad) + self.pivot_y
        new_z = z_arr

        xs_plot = new_z
        ys_plot = new_x
        zs_plot = new_y

        normed = params["norm"](dE_arr)
        colors = params["cmap"](normed)
        colors[:, 3] = 1.0

        self.scatter._offsets3d = (xs_plot, ys_plot, zs_plot)
        self.scatter.set_facecolors(colors)
        self.scatter.set_edgecolors(colors)

        self.current_rotation_angle = angle_deg % 360
        self.title.set_text(f"rot = {self.current_rotation_angle:.0f}°")
        if self.canvas_widget is not None:
            self.canvas_widget.draw()


    def _rotated_df_for_angle(self, angle_deg: float) -> pd.DataFrame:
        """Return a copy of self.df with x,y rotated about pivot by angle_deg (z unchanged)."""
        df = self.df.copy()
        rad = np.deg2rad(angle_deg)
        dx  = df['x'] - self.pivot_x
        dy  = df['y'] - self.pivot_y
        df['x'] = dx * np.cos(rad) - dy * np.sin(rad) + self.pivot_x
        df['y'] = dx * np.sin(rad) + dy * np.cos(rad) + self.pivot_y
        # z unchanged
        return df


    def _log_prediction(self, angle_deg: int, df_at_angle: pd.DataFrame):
        """Call an optional prediction hook and store results under the angle."""
        pred = None
        try:
            if callable(self.predict_fn):
                pred = self.predict_fn(angle_deg, df_at_angle)
            elif hasattr(self.controller, "predict_at_angle") and callable(self.controller.predict_at_angle):
                pred = self.controller.predict_at_angle(angle=angle_deg, df=df_at_angle, context=self)
            # else: leave as None
        except Exception as e:
            pred = {"error": str(e)}
        self.logged_predictions[int(angle_deg)] = pred


    def _start_360_predict_sequence(self, params: dict):

        # Angles 0..360 inclusive so we end exactly one full turn
        self._angles_360 = list(range(0, 361, 10))  
        self._angle_idx  = 0
        self._predict_params = params

        # Initial render at 0°
        self._render_volume_at_angle(self._angles_360[0], params)

        # schedule after 0.5s
        self.parent_frame.after(100, self._advance_360_predict_step)


    # --- add inside class ---
    def build_heat_volume_async(self, angles=None, weights=(1.0,1.0,0.2),
                                blur_sigma=0.8, max_grid=(32,96,96), batch=3):
        if not self.logged_heatmaps:
            print("No logged_heatmaps"); return
        if angles is None:
            angles = sorted(self.logged_heatmaps.keys())
        else:
            angles = [int(a)%360 for a in angles if int(a)%360 in self.logged_heatmaps]
            if not angles:
                print("No matching angles"); return

        sample = self.logged_heatmaps[angles[0]]
        Nz, Ny, Nx = self._default_grid_from_heatmaps(sample, max_grid=max_grid)

        X = np.linspace(self.min_x_for_plot, self.max_x_for_plot, Nx, dtype=np.float32)
        Y = np.linspace(self.min_y_for_plot, self.max_y_for_plot, Ny, dtype=np.float32)
        Z = np.linspace(self.min_z_for_plot, self.max_z_for_plot, Nz, dtype=np.float32)
        Xg, Yg = np.meshgrid(X, Y, indexing="xy")

        self._bv_state = {
            "angles": angles, "i": 0, "w": weights, "blur": blur_sigma,
            "Nx": Nx, "Ny": Ny, "Nz": Nz,
            "X": X, "Y": Y, "Z": Z, "Xg": Xg, "Yg": Yg,
            "V": np.zeros((Nz,Ny,Nx), np.float32),
            "C": np.zeros((Nz,Ny,Nx), np.float32),
        }
        self.parent_frame.after(0, self._bv_step, batch)

    def _bv_step(self, batch):
        st = getattr(self, "_bv_state", None)
        if not st: return
        angles = st["angles"]; i = st["i"]; end = min(i+batch, len(angles))
        w_zy, w_zx, w_xy = st["w"]; eps = 1e-6
        Nx, Ny, Nz = st["Nx"], st["Ny"], st["Nz"]; Xg, Yg = st["Xg"], st["Yg"]

        for k in range(i, end):
            ang = angles[k]
            hms = self.logged_heatmaps.get(ang, {})
            Hzy = self._resize2d(hms.get("ZY"), (Nz, Ny)) if hms.get("ZY") is not None else None
            Hzx = self._resize2d(hms.get("ZX"), (Nz, Nx)) if hms.get("ZX") is not None else None
            Hxy = self._resize2d(hms.get("XY"), (Ny, Nx)) if hms.get("XY") is not None else None

            theta = np.deg2rad(float(ang))
            c, s = np.cos(theta), np.sin(theta)
            Xc, Yc = Xg - self.pivot_x, Yg - self.pivot_y
            Xp = Xc*c + Yc*s + self.pivot_x   # for ZX/XY
            Yp = -Xc*s + Yc*c + self.pivot_y  # for ZY/XY

            if Hxy is not None and w_xy:
                ixp = self._lin_to_idx(Xp, self.min_x_for_plot, self.max_x_for_plot, Nx)
                iyp = self._lin_to_idx(Yp, self.min_y_for_plot, self.max_y_for_plot, Ny)
                slab = Hxy[iyp, ixp]
                st["V"] += w_xy * slab[None, :, :]
                st["C"] += w_xy * (slab[None, :, :] > eps)

            if Hzy is not None and w_zy:
                iyp = self._lin_to_idx(Yp, self.min_y_for_plot, self.max_y_for_plot, Ny)
                for iz in range(Nz):
                    add = Hzy[iz][iyp]
                    st["V"][iz] += w_zy * add
                    st["C"][iz] += w_zy * (add > eps)

            if Hzx is not None and w_zx:
                ixp = self._lin_to_idx(Xp, self.min_x_for_plot, self.max_x_for_plot, Nx)
                for iz in range(Nz):
                    add = Hzx[iz][ixp]
                    st["V"][iz] += w_zx * add
                    st["C"][iz] += w_zx * (add > eps)

        st["i"] = end
        if end < len(angles):
            self.parent_frame.after(1, self._bv_step, batch)
        else:
            V = np.divide(st["V"], np.maximum(st["C"], 1.0),
                        out=np.zeros_like(st["V"]), where=(st["C"]>0))
            if st["blur"] > 0:
                try:
                    from scipy.ndimage import gaussian_filter
                    V = gaussian_filter(V, sigma=st["blur"])
                except Exception:
                    pass
            vmax = float(V.max())
            if vmax > 0: V /= vmax
            self.render_heat_volume(V, axes=(st["Z"], st["Y"], st["X"]), threshold=0.15)
            print("V stats:", V.min(), V.max(), "frac>0.15:", (V >= 0.15).mean())
            self._bv_state = None
    # --- end add ---



    def _advance_360_predict_step(self):

        if self._angle_idx >= len(self._angles_360) - 1:
            try:

                # V, axes = self.build_heat_volume_from_logged(
                #     weights=(1.0, 1.0, 0.2), blur_sigma=0.8, max_grid=(32, 96, 96)
                # )
                # print("V stats:", V.min(), V.max(), "frac>0.15:", (V >= 0.15).mean())
                # self.render_heat_volume(V, axes=axes, threshold=0.15)
                pass 
            
                # This is not working........ 
                # self.build_heat_volume_async(max_grid=(32,96,96), batch=3)

                # print("V stats:", V.min(), V.max(), "frac>0.15:", (V >= 0.15).mean())

            except Exception as e:
                print("Heat-volume build failed:", e)
            return

        self._angle_idx += 1
        angle = self._angles_360[self._angle_idx]


        self._render_volume_at_angle(angle, self._predict_params)

        self.pause_1()

        df_at_angle = self._rotated_df_for_angle(angle)
        self._log_prediction(angle, df_at_angle)

        # wait 0.5s
        self.parent_frame.after(100, self._advance_360_predict_step)


    # ===================== 3D HEAT-VOLUME RECONSTRUCTION =====================

    def _as_heat2d(self, arr):
        """Return a 2-D float32 array in [0,1] from any image-like input."""
        a = np.array(arr, dtype=np.float32)
        if a.ndim == 3 and a.shape[-1] in (3, 4):   # RGB/RGBA -> luminance
            a = a[..., :3].mean(axis=-1)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        amin, amax = float(a.min()), float(a.max())
        if amax > amin:
            a = (a - amin) / (amax - amin)
        else:
            a[:] = 0.0
        return a
    

    def _resize2d(self, a, out_hw):
        """Resize 2-D array to (H,W) in [0,1]. Prefers cv2, falls back to PIL, then stride."""
        if a is None: return None
        a = self._as_heat2d(a)  # ensure 2-D [0,1]
        H, W = map(int, out_hw)
        try:
            import cv2
            return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        except Exception:
            try:
                from PIL import Image
                x = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
                x = Image.fromarray(x).resize((W, H), resample=Image.BILINEAR)
                return (np.asarray(x).astype(np.float32) / 255.0)
            except Exception:
                # very rough fallback (stride slice)
                ys = max(int(np.floor(a.shape[0] / H)), 1)
                xs = max(int(np.floor(a.shape[1] / W)), 1)
                return a[::ys, ::xs][:H, :W].astype(np.float32)

    def _default_grid_from_heatmaps(self, sample_hms, max_grid=(48, 96, 96)):
        """Return Nz,Ny,Nx, clamped to max_grid."""
        zy = sample_hms.get("ZY")
        zx = sample_hms.get("ZX")
        xy = sample_hms.get("XY")
        Nz = int(zy.shape[0]) if zy is not None and zy.ndim == 2 else 64
        Ny = int(zy.shape[1]) if zy is not None and zy.ndim == 2 else (int(xy.shape[0]) if xy is not None else 64)
        Nx = int(zx.shape[1]) if zx is not None and zx.ndim == 2 else (int(xy.shape[1]) if xy is not None else 64)
        # clamp
        Nz = min(Nz, max_grid[0]); Ny = min(Ny, max_grid[1]); Nx = min(Nx, max_grid[2])
        # keep a minimum so axes aren’t degenerate
        Nz = max(Nz, 16); Ny = max(Ny, 32); Nx = max(Nx, 32)
        return Nz, Ny, Nx



    def _lin_to_idx(self, vals, vmin, vmax, n):
        """Map real coords to integer indices [0, n-1] with clipping."""
        t = (vals - vmin) / max(vmax - vmin, 1e-9)
        return np.clip((t * (n - 1)).astype(np.int32), 0, n - 1)

  
    def build_heat_volume_from_logged(self, angles=None, weights=(1.0, 1.0, 0.7), blur_sigma=0.0, max_grid=(48,96,96)):
        if not self.logged_heatmaps:
            raise RuntimeError("No logged heatmaps. Run 'Rotations' or '360-Predict' first.")
        if angles is None:
            angles = sorted(self.logged_heatmaps.keys())
        else:
            angles = [int(a) % 360 for a in angles if int(a) % 360 in self.logged_heatmaps]
            if not angles:
                raise RuntimeError("Provided angles not found in logged_heatmaps.")

        sample = self.logged_heatmaps[angles[0]]
        Nz, Ny, Nx = self._default_grid_from_heatmaps(sample, max_grid=max_grid)

        # world axes at the chosen (downsampled) resolution
        X = np.linspace(self.min_x_for_plot, self.max_x_for_plot, Nx, dtype=np.float32)
        Y = np.linspace(self.min_y_for_plot, self.max_y_for_plot, Ny, dtype=np.float32)
        Z = np.linspace(self.min_z_for_plot, self.max_z_for_plot, Nz, dtype=np.float32)
        Xg, Yg = np.meshgrid(X, Y, indexing="xy")

        V = np.zeros((Nz, Ny, Nx), dtype=np.float32)
        C = np.zeros((Nz, Ny, Nx), dtype=np.float32)
        w_zy, w_zx, w_xy = weights
        eps = 1e-6

        for ang in angles:
            hms = self.logged_heatmaps.get(ang)
            if not hms: 
                continue
            Hzy = self._resize2d(hms.get("ZY"), (Nz, Ny)) if hms.get("ZY") is not None else None   # (Nz,Ny)
            Hzx = self._resize2d(hms.get("ZX"), (Nz, Nx)) if hms.get("ZX") is not None else None   # (Nz,Nx)
            Hxy = self._resize2d(hms.get("XY"), (Ny, Nx)) if hms.get("XY") is not None else None   # (Ny,Nx)

            theta = np.deg2rad(float(ang))
            c, s = np.cos(theta), np.sin(theta)
            Xc, Yc = Xg - self.pivot_x, Yg - self.pivot_y
            Xp = Xc * c + Yc * s + self.pivot_x   # for ZX/XY
            Yp = -Xc * s + Yc * c + self.pivot_y  # for ZY/XY

            if Hxy is not None and w_xy:
                ixp = self._lin_to_idx(Xp, self.min_x_for_plot, self.max_x_for_plot, Nx)
                iyp = self._lin_to_idx(Yp, self.min_y_for_plot, self.max_y_for_plot, Ny)
                slab = Hxy[iyp, ixp]                    # (Ny,Nx)
                V += w_xy * slab[None, :, :]            # smear along Z
                C += w_xy * (slab[None, :, :] > eps)

            if Hzy is not None and w_zy:
                iyp = self._lin_to_idx(Yp, self.min_y_for_plot, self.max_y_for_plot, Ny)
                for iz in range(Nz):
                    row = Hzy[iz]                        # (Ny,)
                    add = row[iyp]                       # (Ny,Nx)
                    V[iz] += w_zy * add
                    C[iz] += w_zy * (add > eps)

            if Hzx is not None and w_zx:
                ixp = self._lin_to_idx(Xp, self.min_x_for_plot, self.max_x_for_plot, Nx)
                for iz in range(Nz):
                    row = Hzx[iz]                        # (Nx,)
                    add = row[ixp]                       # (Ny,Nx)
                    V[iz] += w_zx * add
                    C[iz] += w_zx * (add > eps)

        # normalize where we had contributions
        mask = C > 0
        V = np.divide(V, np.maximum(C, 1.0), out=np.zeros_like(V), where=mask)

        if blur_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                V = gaussian_filter(V, sigma=blur_sigma)
            except Exception:
                pass

        vmax = float(V.max())
        if vmax > 0:
            V /= vmax
        V = np.clip(V, 0.0, 1.0)
        return V, (Z, Y, X)



    # render_heat_volume: replace the whole function with this
    def render_heat_volume(self, V, axes=None, threshold=0.35):
        """
        Show an isovolume via voxels on the EXTRA canvas (top-left).
        Reuses a single FigureCanvasTkAgg so we don't create another widget.
        """
        Z, Y, X = axes if axes is not None else (
            np.linspace(self.min_z_for_plot, self.max_z_for_plot, V.shape[0]),
            np.linspace(self.min_y_for_plot, self.max_y_for_plot, V.shape[1]),
            np.linspace(self.min_x_for_plot, self.max_x_for_plot, V.shape[2]),
        )

        # 1) Create the extra canvas ONCE; otherwise just clear it
        if self.extra_canvas_widget is None:
            # optional: ensure the frame is empty
            for w in self.extra_frame.winfo_children():
                w.destroy()

            self.extra_fig = plt.Figure(figsize=(8, 6), dpi=100)
            self.extra_ax = self.extra_fig.add_subplot(111, projection="3d")
            self.extra_canvas_widget = FigureCanvasTkAgg(self.extra_fig, master=self.extra_frame)
            self.extra_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.extra_ax.clear()

        # 2) Draw the heat volume into self.extra_ax
        occ = V >= float(threshold)
        if not np.any(occ):
            self.extra_ax.text2D(0.1, 0.5, f"No voxels above threshold={threshold}",
                                transform=self.extra_ax.transAxes)
        else:
            norm = (V - V.min()) / (V.max() - V.min() + 1e-9)
            cmap = cm.plasma
            facecolors = np.zeros(occ.shape + (4,), dtype=np.float32)
            rgba = cmap(norm)
            facecolors[occ] = rgba[occ]
            self.extra_ax.voxels(occ, facecolors=facecolors, edgecolor="none")

        self.extra_ax.set_xlim(0, V.shape[2])
        self.extra_ax.set_ylim(0, V.shape[1])
        self.extra_ax.set_zlim(0, V.shape[0])
        self.extra_ax.set_xlabel("X")
        self.extra_ax.set_ylabel("Y")
        self.extra_ax.set_zlabel("Z")
        self.extra_ax.set_title("3D Heat-Volume")

        # 3) Draw on the existing extra canvas (no new widgets created)
        self.extra_canvas_widget.draw()


    def start(self):

        # Main entrypoint: loads data, initializes plot, and launches the animation.
        
        # if self.controller.Advanced_Evaluation_Page.Type_of_Animation_Dropdown_selected.get() == 'Full Run':
        df = self._load_dataframe()
        df = df.sort_values("t0_start").reset_index(drop=True)

        df_mc_hdr =  self._load_mc_hdr()

        self.df = df



        if self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() == 'Full Run':

            self._initialize_plot(df)

            params = self._compute_animation_params(df)

            self.ani = FuncAnimation(self.fig ,lambda idx: self._animate_update(idx, params),frames=params["num_frames"],interval=self.interval_ms,blit=False,repeat=False, )

        elif self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() == 'Rotations':
            current_event_id = self.controller.Setup_Animation_Page.event_id_selected.get()
            current_vertex_id = self.controller.Setup_Animation_Page.vertex_id_selected.get()

            clean_df = df[ (df['event_id'] == int(current_event_id)) & (df['vertex_id'] == int(current_vertex_id)) ]
            clean_df_mc_hdr = df_mc_hdr[ (df_mc_hdr['event_id'] == int(current_event_id)) & (df_mc_hdr['vertex_id'] == int(current_vertex_id)) ]


            if clean_df.empty:
                print(  "No Data",
                        f"No hits found for event {current_event_id}, "
                        f"vertex {current_vertex_id} above dE > {self.energy_cut}"
                )
                return


            # remember to replace the buggy self_df_mc_hdr line with:
            self.df_mc_hdr = clean_df_mc_hdr

            # print( clean_df['t0'] )
            # grab the first (and only) pivot for this event/vertex:
            pivot = self.df_mc_hdr.iloc[0]
            self.pivot_x = float(pivot['x_vert'])
            self.pivot_y = float(pivot['y_vert'])
            self.pivot_z = float(pivot['z_vert'])
            self.df = clean_df

            self._initialize_plot(self.df)
            params = self._compute_animation_params(self.df)



            # self.ani = FuncAnimation(self.fig ,lambda idx: self._animate_update_roations(idx, params),frames=params["num_frames"],interval=self.interval_ms,blit=False,repeat=False, )
            self.ani = FuncAnimation(self.fig ,lambda idx: self._animate_update_roations(idx, params),frames=itertools.count(),interval=self.interval_ms,blit=False,repeat=False, )

        # elif self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() == '360-Predict':
        #     pass

        elif self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() == '360-Predict':
            current_event_id = self.controller.Setup_Animation_Page.event_id_selected.get()
            current_vertex_id = self.controller.Setup_Animation_Page.vertex_id_selected.get()

            clean_df = df[ (df['event_id'] == int(current_event_id)) & (df['vertex_id'] == int(current_vertex_id)) ]
            clean_df_mc_hdr = df_mc_hdr[ (df_mc_hdr['event_id'] == int(current_event_id)) & (df_mc_hdr['vertex_id'] == int(current_vertex_id)) ]

            if clean_df.empty:
                print("No Data",
                    f"No hits found for event {current_event_id}, "
                    f"vertex {current_vertex_id} above dE > {self.energy_cut}")
                return

            self.df_mc_hdr = clean_df_mc_hdr
            pivot = self.df_mc_hdr.iloc[0]
            self.pivot_x = float(pivot['x_vert'])
            self.pivot_y = float(pivot['y_vert'])
            self.pivot_z = float(pivot['z_vert'])
            self.df = clean_df

            self._initialize_plot(self.df)
            params = self._compute_animation_params(self.df)


            self.ani = None
            self._start_360_predict_sequence(params)


        # self.ani.event_source.start()
        if self.ani is not None:
            self.ani.event_source.start()
    
        # else:
        #     print(self.controller.Advanced_Evaluation_Page.Type_of_Animation_Dropdown_selected.get() )



    def play(self):
        if self.ani is not None:
            self.ani.event_source.start()






    def pause_1(self):

        if self.ani is not None:
            self.ani.event_source.stop()

        temp_df = self.df.iloc[self._visible_indices].copy()

        if self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() in ("Rotations", "360-Predict"):
            angle = self.current_rotation_angle
            rad   = np.deg2rad(angle)
            dx    = temp_df['x'] - self.pivot_x
            dy    = temp_df['y'] - self.pivot_y
            temp_df['x'] = dx * np.cos(rad) - dy * np.sin(rad) + self.pivot_x
            temp_df['y'] = dx * np.sin(rad) + dy * np.cos(rad) + self.pivot_y


        # compute base projections once
        base_zy = self.controller.Use_Pixel_Array.plot_testing(self=self, DF=temp_df, plot_canvas=None, projection="ZY", return_image=True)
        base_zx = self.controller.Use_Pixel_Array.plot_testing(self=self, DF=temp_df, plot_canvas=None, projection="ZX", return_image=True)
        base_xy = self.controller.Use_Pixel_Array.plot_testing(self=self, DF=temp_df, plot_canvas=None, projection="XY", return_image=True)

        base_zy = base_zy[:, :, :3] / 255.0
        base_zx = base_zx[:, :, :3] / 255.0
        base_xy = base_xy[:, :, :3] / 255.0
        
        if self.controller.model != None:

            figs = self.controller.Heat_Map_Class.get_heatmaps_from_list(self=self, image_list=[base_zy, base_zx, base_xy])

            hm_arrays = []
            for _fig in figs:
                _ax = _fig.axes[0]
                _raw = _ax.images[0].get_array()
                _arr = self._as_heat2d(_raw)
                hm_arrays.append(_arr)
                plt.close(_fig)

            angle_key = int(round(self.current_rotation_angle)) % 360
            self.logged_heatmaps[angle_key] = {"ZY": hm_arrays[0], "ZX": hm_arrays[1], "XY": hm_arrays[2]}
        
        else:
            print( "There is not Model Loaded for heatmaps" )
    


        for i in range(4):
            dropdown = getattr(self.animation_page, f"dropdown_{i+1}")
            proj     = dropdown.get()
            frame    = getattr(self.animation_page, f"plot_frame_{i+1}")

            if proj == '':
                for w in frame.winfo_children():
                    w.destroy()


            elif proj in ("ZY", "ZX", "XY"):
                self.controller.Use_Pixel_Array.plot_testing( self=self, DF=temp_df, plot_canvas=frame, projection=proj  )
                continue
            
            elif proj in ("ZY (heatmap)", "ZX (heatmap)", "XY (heatmap)"):

                for w in frame.winfo_children():
                    w.destroy()


                if self.controller.model != None: 
                    figs = self.controller.Heat_Map_Class.get_heatmaps_from_list( self=self, image_list=[base_zy, base_zx,  base_xy] )
                    if proj == "ZY (heatmap)":
                        fig_i = figs[0]
                    elif proj == "ZX (heatmap)":
                        fig_i = figs[1]
                    else:
                        fig_i = figs[2]

                    ax_src = fig_i.axes[0]
                    img_np = ax_src.images[0].get_array()


                    hm_arrays = []
                    for _fig in figs:
                        _ax = _fig.axes[0]
                        _arr = np.array(_ax.images[0].get_array(), dtype=np.float32)
                        # defensive normalization to [0,1]
                        if _arr.size and np.nanmax(_arr) > 0:
                            _arr = (_arr - np.nanmin(_arr)) / (np.nanmax(_arr) - np.nanmin(_arr) + 1e-12)
                        hm_arrays.append(_arr)
                    # hm_arrays[0] -> ZY (z,y), hm_arrays[1] -> ZX (z,x), hm_arrays[2] -> XY (y,x)

                    # log per-angle heatmaps if we’re in a rotation-based mode
                    if self.controller.Setup_Animation_Page.Type_of_Animation_Dropdown_selected.get() in ("Rotations", "360-Predict"):
                        angle_key = int(round(self.current_rotation_angle)) % 360
                        self.logged_heatmaps[angle_key] = {"ZY": hm_arrays[0], "ZX": hm_arrays[1], "XY": hm_arrays[2]}



                    h, w = img_np.shape[:2]
                    dpi = 96
                    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
                    ax  = fig.add_axes([0, 0, 1, 1])   # no margins
                    ax.imshow(img_np, aspect='auto')
                    ax.axis('off')
                    plt.close(fig_i)  # close original

                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    canvas.draw()
                    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
                    toolbar.update(); toolbar.pack(side=tk.LEFT, fill=tk.X)




            elif proj == 'Prediction Probalility':
                if self.controller.model != None:
                    for w in frame.winfo_children():
                        w.destroy()


                    input_shapes = self.controller.model.input_shape
                    if isinstance(input_shapes, tuple):  # single-input safety
                        input_shapes = [input_shapes]

                    def _prep_for_model(x, expected_shape):
                        """  expected_shape is typically (None, H, W, C) for TF 'channels_last'. If H/W are None, we skip resizing and just add a batch axis. """
                        # If grayscale came back, make it (H,W,1)
                        if x.ndim == 2:
                            x = np.expand_dims(x, -1)

                        # Determine expected H, W, C if provided
                        H = expected_shape[1] if len(expected_shape) > 1 else None
                        W = expected_shape[2] if len(expected_shape) > 2 else None
                        C = expected_shape[3] if len(expected_shape) > 3 else None

                        # Resize only if model requires fixed H/W
                        if H is not None and W is not None and (x.shape[0] != H or x.shape[1] != W):
                            try:
                                # go through uint8 for PIL then back to float32 [0,1]
                                x_uint8 = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
                                x = np.array(Image.fromarray(x_uint8).resize((W, H), resample=Image.BILINEAR)).astype(np.float32) / 255.0
                            except Exception:
                                # fallback to cv2
                                x = cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
                                if x.max() > 1.0:  # if cv2 returned 0..255
                                    x = x / 255.0

                        # If model expects 1 channel but we have 3, average to luma
                        if C == 1 and x.shape[-1] == 3:
                            x = np.mean(x, axis=-1, keepdims=True)

                        # Add batch axis => (1, H, W, C)
                        x = np.expand_dims(x, axis=0).astype(np.float32)
                        return x

                    x_zy = _prep_for_model(base_zy, input_shapes[0])
                    x_zx = _prep_for_model(base_zx, input_shapes[1])
                    x_xy = _prep_for_model(base_xy, input_shapes[2])

                    # Predict with a consistent batch size (1 for each input)
                    predict_probs = self.controller.model.predict([x_zy, x_zx, x_xy], verbose=0)[0]
                    # print(predict_probs)

                    try:
                        class_name_map = self.controller.model.class_name_dict
                        print( class_name_map )

                    except:
                        class_name_map = { i  : i for i in range(len( predict_probs ))}

                    # print(class_name_map)n
                    fig = plt.figure()
                    # ax  = fig.add_axes([0, 0, 1, 1])   # no margins
                    ax  = fig.add_axes( [0.2, 0.2, 0.6, 0.6] )   # no margins

                    x_labels    = list(class_name_map.values())
                    int_label   = np.arange( len(x_labels))
                    
                    ax.bar( x_labels , predict_probs )
                    ax.set_xticks( x_labels )
                    ax.set_xticklabels(int_label, rotation=90, fontsize=8)
                    ax.set_ylabel("Probability", fontsize=10)
                    ax.set_ylim(0,1)

                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    canvas.draw()
                    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
                    toolbar.update(); toolbar.pack(side=tk.LEFT, fill=tk.X)
                    
                    plt.close()

                    
                    pass

            elif proj == "Input Usefulness":
                if self.controller.model != None: 
                    self.controller.model
                else:
                    print("No model found")
                    return

                for w in frame.winfo_children():
                    w.destroy()
                try:
                    self.controller.Evaluating_Model.Plot_input_usefulness( self = self ,  image_paths = None , raw_images=[ base_zy , base_zx , base_xy], plot_frame = frame )
                except:
                    print("ERROR : Input Usefulness")

            

    def pause(self):
        if self.ani is not None:
            self.ani.event_source.stop()
            print( self.ani )

            
 
    def replay(self):
        if self.ani is not None:
            self.ani.event_source.stop()

            self.ani.frame_seq = self.ani.new_frame_seq()

            self._animate_update(0, self._compute_animation_params(self._load_dataframe()))
            self.canvas_widget.draw()
            self.ani.event_source.start()

