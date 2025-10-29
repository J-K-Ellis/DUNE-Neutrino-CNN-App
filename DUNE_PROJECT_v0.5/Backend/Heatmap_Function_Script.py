from Imports.common_imports import *


class Heat_Map_Class:

    def Heatmap_func(self, paths, random_100=False):
        evaluate_page = self.controller.frames.get("Evaluate_Model_Page")
        if evaluate_page is None:
            print("Evaluate_Model_Page not found.")
            return
        for w in evaluate_page.Figure_Canvas_Frame.winfo_children():
            w.destroy()
        if isinstance(paths, str):
            paths = [paths]
        if random_100:
            base_dir = paths[0]
            pdf_name = f"Heatmap_predictions_{os.path.basename(base_dir)}.pdf"
            files = np.random.choice(os.listdir(base_dir), size=100, replace=False)
            with PdfPages(pdf_name) as pp:
                for fn in files:
                    file_paths = [os.path.join(d, fn) if os.path.isdir(d) else d for d in paths]
                    if os.path.isdir(paths[0]):
                        file_paths[0] = os.path.join(paths[0], fn)
                    self._heatmap_multi_step(file_paths, evaluate_page, os.path.basename(os.path.dirname(file_paths[0])))
                    pp.savefig(dpi=600)
                    plt.close()
            print(f"Saved to {pdf_name}")
            return
        if len(paths) == 1:
            note = os.path.basename(os.path.dirname(paths[0]))
            self._heatmap_single_step(paths[0], evaluate_page, note)
        else:
            main_file = paths[0]
            others = paths[1:]
            subdir = os.path.basename(os.path.dirname(main_file))
            fname = os.path.basename(main_file)
            ch0 = os.path.basename(os.path.dirname(os.path.dirname(main_file)))
            file_paths = [main_file]
            for o in others:
                if os.path.isdir(o):
                    ch = os.path.basename(o.rstrip("/"))
                    new_fname = fname.replace(f"_{ch0}_", f"_{ch}_")
                    file_paths.append(os.path.join(o, subdir, new_fname))
                else:
                    file_paths.append(o)
            note = os.path.basename(os.path.dirname(main_file))
            Heat_Map_Class._heatmap_multi_step(self, file_paths, evaluate_page, note)

    def _heatmap_single_step(self, path, evaluate_page, Note):
        shapes = self.controller.model.input_shape
        if isinstance(shapes, tuple):
            shapes = [shapes]
        h, w = shapes[0][1:3]
        pil_img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
        img_arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        batch = img_arr[None, ...]
        inp = tf.keras.Input(shape=self.controller.model.input_shape[1:])
        x = inp
        last_conv = None
        for layer in self.controller.model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = x
        if last_conv is None:
            print("No Conv2D layer found.")
            return
        preds = x
        grad_model = tf.keras.Model(inputs=inp, outputs=[last_conv, preds])
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(batch)
            loss = tf.reduce_max(predictions, axis=1)
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(tf.maximum(conv_out[0] @ pooled[..., None], 0))
        heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
        heatmap = np.uint8(255 * heatmap.numpy())
        heatmap = Image.fromarray(heatmap).resize((w, h))
        cmap = cm.jet(np.array(heatmap))[:, :, :3]
        overlay = (0.5 * img_arr + 0.5 * cmap).clip(0, 1)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
        ax0.imshow(pil_img); ax0.axis("off"); ax0.set_title("Original")
        ax1.imshow(overlay); ax1.axis("off"); ax1.set_title("Heatmap Overlay")
        pred_pos = np.argmax(predictions.numpy().flatten())
        pred_lbl = self.controller.model.class_name_dict.get(pred_pos, str(pred_pos))
        plt.suptitle(f"True Label : {Note} | Pred Label : {pred_lbl}")
        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, evaluate_page.Figure_Canvas_Frame, pack_toolbar=False)
        toolbar.update(); toolbar.pack(side=tk.LEFT, fill=tk.X)
        plt.close(fig)

    def _heatmap_multi_step(self, paths, evaluate_page, Note = ''):
        model = self.controller.model
        shapes = model.input_shape
        if isinstance(shapes, tuple):
            shapes = [shapes]
        pil_imgs = []
        img_arrs = []
        for i, p in enumerate(paths):
            h, w = shapes[i][1:3]
            img = Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            pil_imgs.append(img)
            img_arrs.append(arr)
        concat_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                concat_layer = layer
                break
        if concat_layer is None:
            print("Concatenate layer not found.")
            return
        branch_inputs = concat_layer.input if isinstance(concat_layer.input, list) else [concat_layer.input]
        last_convs = []
        for t in branch_inputs:
            layer = t._keras_history[0]
            visited = set()
            stack = [layer]
            found = None
            while stack and found is None:
                L = stack.pop()
                if id(L) in visited:
                    continue
                visited.add(id(L))
                if isinstance(L, tf.keras.layers.Conv2D):
                    found = L
                    break
                ins = L.input if isinstance(L.input, list) else [L.input]
                for tin in ins:
                    stack.append(tin._keras_history[0])
            last_convs.append(found)
        grad_models = [tf.keras.Model(inputs=model.inputs, outputs=[lc.output, model.output]) for lc in last_convs]
        batches = [arr[None, ...] for arr in img_arrs]
        overlays = []
        predictions_ref = None
        for i, gm in enumerate(grad_models):
            with tf.GradientTape() as tape:
                conv_out, preds = gm(batches)
                loss = tf.reduce_max(preds, axis=1)
            grads = tape.gradient(loss, conv_out)
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.squeeze(tf.maximum(conv_out[0] @ pooled[..., None], 0))
            cam /= tf.reduce_max(cam) + tf.keras.backend.epsilon()
            cam = np.uint8(255 * cam.numpy())
            h, w = shapes[i][1:3]
            hm = Image.fromarray(cam).resize((w, h))
            cmap = cm.jet(np.array(hm))[:, :, :3]
            ovl = (0.5 * img_arrs[i] + 0.5 * cmap).clip(0, 1)
            overlays.append(ovl)
            predictions_ref = preds
        N = len(paths)
        fig, axes = plt.subplots(2, N, figsize=(5 * N, 9))
        if N == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        chan_names = [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths]
        for i in range(N):
            axes[0, i].imshow(pil_imgs[i]); axes[0, i].axis("off"); axes[0, i].set_title(f"Original {chan_names[i]}")
            axes[1, i].imshow(overlays[i]); axes[1, i].axis("off"); axes[1, i].set_title(f"Heatmap {chan_names[i]}")
        pred_pos = np.argmax(predictions_ref.numpy().flatten())
        try:
            pred_lbl = model.class_name_dict.get(pred_pos, str(pred_pos))
        except:
            pred_lbl = pred_pos

        plt.suptitle(f"True Label : {Note} | Pred Label : {pred_lbl}")
        canvas = FigureCanvasTkAgg(fig, master=evaluate_page.Figure_Canvas_Frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, evaluate_page.Figure_Canvas_Frame, pack_toolbar=False)
        toolbar.update(); toolbar.pack(side=tk.LEFT, fill=tk.X)
        plt.close(fig)

    def get_heatmaps_from_list(self, image_list):
        model = self.controller.model
        shapes = model.input_shape
        if isinstance(shapes, tuple):
            shapes = [shapes]

        pil_imgs = []
        img_arrs = []
        for i, img in enumerate(image_list):
            h, w = shapes[i][1:3]

            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")

            img_resized = img.resize((w, h), Image.BILINEAR)
            # img_resized = img.resize((h , w), Image.BILINEAR)
            # img_resized = img
            arr = np.asarray(img_resized, dtype=np.float32) / 255.0
            pil_imgs.append(img_resized)
            img_arrs.append(arr)

        if len(shapes) == 1:
            return [Heat_Map_Class._generate_heatmap_single_branch(self ,img_arrs[0], pil_imgs[0], shapes[0])]
        else:
            return Heat_Map_Class._generate_heatmap_multi_branch(self,img_arrs, pil_imgs, shapes)


    def _generate_heatmap_single_branch(self, img_arr, pil_img, shape):
        h, w = shape[1:3]
        batch = img_arr[None, ...]
        inp = tf.keras.Input(shape=self.controller.model.input_shape[1:])
        x = inp
        last_conv = None
        for layer in self.controller.model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = x
        grad_model = tf.keras.Model(inputs=inp, outputs=[last_conv, x])
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(batch)
            loss = tf.reduce_max(predictions, axis=1)
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(tf.maximum(conv_out[0] @ pooled[..., None], 0))
        heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
        heatmap = np.uint8(255 * heatmap.numpy())
        heatmap = Image.fromarray(heatmap).resize((w, h))
        cmap = cm.jet(np.array(heatmap))[:, :, :3]
        overlay = (0.5 * img_arr + 0.5 * cmap).clip(0, 1)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
        ax0.imshow(pil_img); ax0.axis("off"); ax0.set_title("Original")
        ax1.imshow(overlay); ax1.axis("off"); ax1.set_title("Heatmap Overlay")
        return fig

    def _generate_heatmap_multi_branch(self, img_arrs, pil_imgs, shapes,
                                    alpha=0.5,          # 50/50 blend
                                    perc=99.5,          # robust max for normalization
                                    gamma=0.8):         # <1 boosts midtones; >1 compresses

        model = self.controller.model

        # Find the Concatenate that merges branches
        concat_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                concat_layer = layer
                break
        if concat_layer is None:
            print("Concatenate layer not found.")
            return []

        # Find last Conv2D per branch
        branch_inputs = concat_layer.input if isinstance(concat_layer.input, list) else [concat_layer.input]
        last_convs = []
        for t in branch_inputs:
            layer = t._keras_history[0]
            visited, stack, found = set(), [layer], None
            while stack and found is None:
                L = stack.pop()
                if id(L) in visited:
                    continue
                visited.add(id(L))
                if isinstance(L, tf.keras.layers.Conv2D):
                    found = L
                    break
                ins = L.input if isinstance(L.input, list) else [L.input]
                for tin in ins:
                    stack.append(tin._keras_history[0])
            last_convs.append(found)

        grad_models = [
            tf.keras.Model(inputs=model.inputs, outputs=[lc.output, model.output]) for lc in last_convs
        ]

        # Use ALL real inputs (don’t zero the other branches)
        batches = [arr[None, ...] for arr in img_arrs]

        figs = []
        for i, gm in enumerate(grad_models):
            with tf.GradientTape() as tape:
                conv_out, preds = gm(batches)
                loss = tf.reduce_max(preds, axis=1)

            grads   = tape.gradient(loss, conv_out)        # (1, Hc, Wc, C)
            weights = tf.reduce_mean(grads, axis=(0,1,2))  # (C,)

            cam = tf.squeeze(tf.maximum(conv_out[0] @ weights[..., None], 0)).numpy()  # (Hc, Wc)

            # Resize CAM to branch input size
            h, w = shapes[i][1:3]
            cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR), dtype=np.float32)

            # Robust normalization + gamma for contrast
            cam_resized -= cam_resized.min()
            denom = np.percentile(cam_resized, perc)
            if denom <= 1e-6:
                denom = cam_resized.max() + 1e-6
            cam_resized = np.clip(cam_resized / denom, 0, 1)
            if gamma != 1.0:
                cam_resized = cam_resized ** gamma

            # Colorize and blend (exactly 50/50)
            cmap_rgb = cm.get_cmap("jet")(cam_resized)[:, :, :3]   # [0,1]
            overlay  = 0.5 * img_arrs[i] + 0.5 * cmap_rgb
            overlay  = np.clip(overlay, 0, 1)

            # Single image (overlay only)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay)
            ax.axis("off")
            figs.append(fig)

        return figs
