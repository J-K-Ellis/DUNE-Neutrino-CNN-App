from Imports.common_imports import *

class Model_Training:

    def __init__(self):

        self.optimizer = tf.keras.optimizers.Adam( )
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()



    @tf.function
    def Model_training_Step(self, model, x_batch, y_batch, class_weights_vec , Conditions = []):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)

            per_example = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=y_batch, logits=logits )
            weights = tf.gather(class_weights_vec, y_batch)

            loss = tf.reduce_mean(per_example * weights)
            if Conditions !=[]:    
                loss = self.controller.Tuning_Condition_Handler.evaluate_conditions(batch_data={ 'logits': logits, 'y_batch': y_batch, 'per_example': per_example, 'weights': weights, 'loss': loss }, conditions=Conditions ) 
                pass



        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_batch), tf.float32))
        return loss, acc

    @tf.function
    def Model_val_Step(self, model, x_batch, y_batch, class_weights_vec):
        logits = model(x_batch, training=False)
        per_example = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=y_batch, logits=logits )
        weights = tf.gather(class_weights_vec, y_batch)
        loss = tf.reduce_mean(per_example * weights)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return loss, preds

# ----------------------------------------------------------------------
    def Advance_Image_Set_Selection(self, replace_train=False, free_File_Names_Dict=None, train_File_value_Dict=None):
        # Advanced image set selection with custom train/val/test splits and optional custom test directory.

        seed = int(getattr(self, "split_seed", 12345))
        self.split_seed = seed                      # ensure it sticks around
        rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)                    # TF ops that use RNG


        n_inputs = len(self.controller.model.inputs)
        base_dirs = [self.selected_dir_got]
        if n_inputs >= 2: base_dirs.append(self.selected_dir_got_2)
        if n_inputs == 3: base_dirs.append(self.selected_dir_got_3)
        for d in base_dirs:
            if n_inputs != len(base_dirs):
                raise ValueError(f'Mismatch between number of paths ( {len(base_dirs)} ) and the number of inputs required')
            if not os.path.isdir(d):
                raise ValueError(f"Input directory not found: {d}")
        base_proj_tokens = [os.path.basename(d) for d in base_dirs]

        # Access Advance Page and all of its variables.
        Advance_Page = self.controller.frames.get("Advance_Class_Selection_Page")

        # If Balanced_Batches is enabled (True), we will balance the batches during training for each of the classes.
        Balanced_Batches = Advance_Page.Balance_Batches.get()

        custom_test_dirs = None
        if Advance_Page and isinstance(Advance_Page.test_set_dir, list) and any(Advance_Page.test_set_dir):
            custom_test_dirs = [p for p in Advance_Page.test_set_dir if p]
            if len(custom_test_dirs) != n_inputs:
                print(f"[ERROR] Selected custom test dirs = {len(custom_test_dirs)} but model expects {n_inputs} inputs. Aborting.")
                return None

            missing_dirs = [p for p in custom_test_dirs if not os.path.isdir(p)]
            if missing_dirs:
                print(f"[ERROR] Custom test dirs not found: {missing_dirs}. Aborting.")
                return None



        slider_sections = {split: Advance_Page.slider_sections[split] for split in ("Train","Validate","Test")}
        requested_counts = {
            split: {
                sec.labels[i]: int(sec.entry_vars[i].get() or 0)
                for i in range(len(sec.labels))
                if int(sec.entry_vars[i].get() or 0) > 0
            }
            for split, sec in slider_sections.items()
        }
        custom_groups = Advance_Page.custom_groups
        leaves_of = lambda c: custom_groups.get(c, [c])
        all_slider_classes = sorted({c for d in requested_counts.values() for c in d})
        needed_leaves      = sorted({leaf for cls in all_slider_classes for leaf in leaves_of(cls)})


        def build_grouped_map(dirs, proj_tokens):
            # """Map key -> {proj_token: path} for keys present in ALL dirs."""

            proj_atoms = set(itertools.chain.from_iterable(proj.split('_') for proj in proj_tokens))
            n_inputs = len(proj_tokens)

            def scan_one(d, proj):
                for root, dirnames, files in os.walk(d):
                    dirnames.sort()                                    # <- stable tree walk
                    files = sorted(f for f in files if f.lower().endswith('.png'))  # <- stable file order
                    for f in files:
                        stem = os.path.splitext(f)[0]
                        parts = [p for p in stem.split('_') if p and p not in proj_atoms]
                        if not parts:
                            continue
                        key = "_".join(parts)
                        yield key, os.path.join(root, f), proj

            tmp = {}
            with ThreadPoolExecutor(max_workers=min(len(dirs), 8)) as ex:
                for key, path, proj in itertools.chain.from_iterable(
                        ex.map(lambda dp: scan_one(*dp), zip(dirs, proj_tokens))):
                    tmp.setdefault(key, {})[proj] = path

            # Return a dict with deterministic key order
            return dict(sorted((k, v) for k, v in tmp.items() if len(v) == n_inputs))
        

        
        def bucket_by_leaf(grouped_map, needed_leaves):
            # """Bucket keys by leaf, first match wins, stable order."""
            buckets = {leaf: [] for leaf in needed_leaves}
            for key in sorted(grouped_map):        
                key_w = f"_{key}_"
                for leaf in needed_leaves:
                    if f"_{leaf}_" in key_w:
                        buckets[leaf].append(key)
                        break
            return buckets
        

        def select_for_split(leaf_to_keys_map, split_name):
            # """Select keys for a given split from the leaf_to_keys_map."""
            out = {}
            for cls in all_slider_classes:
                pool = []
                for leaf in leaves_of(cls):
                    pool += leaf_to_keys_map.get(leaf, [])
                pool = sorted(set(pool))               
                rng.shuffle(pool)                
                req = requested_counts.get(split_name, {}).get(cls, 0)
                out[cls] = pool[:min(req, len(pool))]
            return out

        def key_to_paths_from(grouped_map, proj_tokens, key):
            return tuple(grouped_map[key][proj] for proj in proj_tokens)


        grouped_base = build_grouped_map(base_dirs, base_proj_tokens)
        leaf_to_keys_base = bucket_by_leaf(grouped_base , needed_leaves)


        split_order = ("Train", "Validate", "Test")
        selected = {s: {cls: [] for cls in all_slider_classes} for s in split_order}
        free_keys = {cls: [] for cls in all_slider_classes}


        for cls in all_slider_classes:
            pool = []
            for leaf in leaves_of(cls):
                pool += leaf_to_keys_base.get(leaf, [])
            rng.shuffle(pool)
            rem = pool.copy()
            for split in split_order:
                req = requested_counts.get(split, {}).get(cls, 0)
                take = min(req, len(rem))
                if take < req:
                    print(f"[WARN] Requested {req} of '{cls}' in {split} but got {take}.")
                selected[split][cls] = rem[:take]
                rem = rem[take:]
            free_keys[cls] = rem


        def build_paths_dict(selected_map, grouped_map, proj_tokens):
            # """Build dict of class -> list of path tuples from selected keys."""
            out = {}
            for cls, keys in selected_map.items():
                out[cls] = [key_to_paths_from(grouped_map, proj_tokens, k) for k in keys]
            return out

        if custom_test_dirs:
            test_proj_tokens = [os.path.basename(d) for d in custom_test_dirs]
            grouped_test = build_grouped_map(custom_test_dirs, test_proj_tokens)
            leaf_to_keys_test = bucket_by_leaf(grouped_test , needed_leaves)

            selected_test = select_for_split(leaf_to_keys_test, "Test")
            selected["Test"] = selected_test  # override Test keys only


            path_dicts_train = build_paths_dict(selected["Train"], grouped_base, base_proj_tokens)
            path_dicts_val   = build_paths_dict(selected["Validate"], grouped_base, base_proj_tokens)
            path_dicts_test  = build_paths_dict(selected["Test"], grouped_test, test_proj_tokens)

        else:
            # No custom test: use BASE for all splits

            path_dicts_train = build_paths_dict(selected["Train"], grouped_base, base_proj_tokens)
            path_dicts_val   = build_paths_dict(selected["Validate"], grouped_base, base_proj_tokens)
            path_dicts_test  = build_paths_dict(selected["Test"], grouped_base, base_proj_tokens)


        def load_png(path, H, W):
            data = tf.io.read_file(path)
            img  = tf.image.decode_png(data, channels=3)
            return tf.image.resize(img, (H, W)) / 255.0

        def make_single_ds(pd, batch):
            paths, labels = [], []
            for cls, plist in pd.items():
                paths += [p[0] for p in plist]
                labels += [label_map[cls]] * len(plist)

            # If Balanced_Batches is enabled, we will balance the dataset batches here. If not, we proceed as normal.
            if Balanced_Batches:
                class_counts = { cls: len(plist) for cls, plist in pd.items() }
                min_count = min(class_counts.values()) if class_counts else 0

                balanced_paths, balanced_labels = [], [] 
                for cls , plist in pd.items():
                    balanced_paths  += [p[0] for p in plist[:min_count]]
                    balanced_labels += [label_map[cls]] * min_count
                paths, labels = balanced_paths, balanced_labels
                print(f"Balanced batches enabled — using {min_count} samples per class.")

            H, W = self.controller.model.input_shape[1:3]
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=False)   
            ds = ds.map(lambda p, y: (load_png(p, H, W), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
            opts = tf.data.Options()
            opts.experimental_deterministic = True
            return ds.with_options(opts).batch(batch).prefetch(tf.data.AUTOTUNE)

        def make_multi_ds(pd, batch):
            groups, labels = [], []
            for cls, plist in pd.items():
                for tpl in plist:
                    groups.append(tpl)
                    labels.append(label_map[cls])

            if Balanced_Batches:
                class_counts = {cls: len(plist) for cls, plist in pd.items()}
                min_count = min(class_counts.values())

                balanced_groups, balanced_labels = [], []
                for cls, plist in pd.items():
                    balanced_groups += plist[:min_count]
                    balanced_labels += [label_map[cls]] * min_count

                groups, labels = balanced_groups, balanced_labels
                print(f"Balanced batches enabled — using {min_count} samples per class.")


            view_lists = list(zip(*groups))

            ds = tf.data.Dataset.from_tensor_slices( (tuple(tf.constant(v) for v in view_lists), tf.constant(labels)) )

            ds = ds.shuffle(len(groups), seed=seed, reshuffle_each_iteration=False) 

            HsWs = [self.controller.model.input_shape[i][1:3] for i in range(n_inputs)]

            def map_fn(paths, y):
                imgs = [load_png(paths[i], *HsWs[i]) for i in range(n_inputs)]
                return tuple(imgs), y
            
            ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            opts = tf.data.Options()
            opts.experimental_deterministic = True

            return ds.with_options(opts).batch(batch).prefetch(tf.data.AUTOTUNE)



        batch     = self.Batch_size_text_selected.get()
        label_map = {c: i for i, c in enumerate(all_slider_classes)}


        if replace_train:

            new_keys = {c: free_File_Names_Dict.get(c, [])[:train_File_value_Dict.get(c, 0)] for c in all_slider_classes}
            mini_path_dict = {cls: [tuple(grouped_base[key][proj] for proj in base_proj_tokens) for key in keys] for cls, keys in new_keys.items()}


            builder = make_multi_ds if n_inputs > 1 else make_single_ds
            return builder(mini_path_dict, batch)


        builder = make_multi_ds if n_inputs > 1 else make_single_ds
        train_ds = builder(path_dicts_train, batch)
        val_ds   = builder(path_dicts_val,   batch)
        test_ds  = builder(path_dicts_test,  batch)

        if sum(len(v) for v in path_dicts_train.values()) == 0:
            raise ValueError("No training images found; check slider or directory.")


        Raw_Data = train_ds.concatenate(val_ds).concatenate(test_ds)
        Raw_Data.class_names = all_slider_classes

        lbls = [label_map[c] for c, plist in path_dicts_train.items() for _ in plist]
        if lbls:
            if Advance_Page.Enable_Class_Weights.get():
                cw = class_weight.compute_class_weight("balanced", classes=np.unique(lbls), y=lbls)
                cw_dict = {int(c): float(w) for c, w in zip(np.unique(lbls), cw)}
            else:
                cw_dict = {i: 1.0 for i in label_map.values()}
        else:
            cw_dict = {i: 1.0 for i in label_map.values()}

        counts = {  "Train":    {cls: len(v) for cls, v in path_dicts_train.items()},
                    "Validate": {cls: len(v) for cls, v in path_dicts_val.items()},
                    "Test":     {cls: len(v) for cls, v in path_dicts_test.items()}, }

        return (Raw_Data, train_ds, val_ds, test_ds, counts["Train"], counts["Validate"], counts["Test"], free_keys, cw_dict)




    def Basic_Image_Set_Selection(self):

        Model_input_height, Model_input_width = ( self.controller.model.input_shape[1], self.controller.model.input_shape[2]  )


        Raw_Data = tf.keras.utils.image_dataset_from_directory( self.selected_dir_got , shuffle=True , batch_size=self.Batch_size_text_selected.get() , label_mode='int',  image_size=(Model_input_height, Model_input_width) )

        total_size = len(Raw_Data.file_paths) 

        train_percent = self.train_size_text_selected.get()
        val_percent = self.val_size_text_selected.get()
        test_percent = self.test_size_text_selected.get()


        if (train_percent + val_percent + test_percent) != 100:
            print("Train, validation, and test sizes do not add up to 100%.")
            return

        train_size = int(total_size * (train_percent / 100))
        val_size = int(total_size * (val_percent / 100))
        test_size = int(total_size * (test_percent / 100))

        train = Raw_Data.take(train_size)
        val = Raw_Data.skip(train_size).take(val_size)
        test = Raw_Data.skip(train_size + val_size).take(test_size)


        return Raw_Data , train , val , test



    def Train_Model(self):

        os.system('cls||clear')
        n_inputs = len(self.controller.model.inputs)

        if n_inputs == 1:
            self.selected_dir_got   = self.selected_dir.get()
            print(self.selected_dir_got)
        elif n_inputs == 2: 
            self.selected_dir_got   = self.selected_dir.get()
            self.selected_dir_got_2 = self.selected_dir_2.get()
            print(self.selected_dir_got)
            print(self.selected_dir_got_2)

        else:
            self.selected_dir_got   = self.selected_dir.get()
            self.selected_dir_got_2 = self.selected_dir_2.get()
            self.selected_dir_got_3 = self.selected_dir_3.get()
            print(self.selected_dir_got)
            print(self.selected_dir_got_2)
            print(self.selected_dir_got_3)
        
        trainer = Model_Training()
        trainer.optimizer.learning_rate = self.controller.model_learning_rate



        def print_dataset_counts(dataset, class_names, name):

            labels_ds = dataset.unbatch().map(lambda x, y: y)
            all_labels = np.array(list(labels_ds.as_numpy_iterator()), dtype=int)
            
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            

            print(f"\n{name} set:")
            for lbl, cnt in zip(unique_labels, counts):
                print(f"  Class {class_names[lbl]}: {cnt} images")


        def find_first_image(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.png','.jpg','jpeg','bmp','gif')):
                        return os.path.join(root, file)
            return None

        first_image = find_first_image(self.selected_dir_got)
        if not first_image:
            print("No images found.")
            return
        try:
            with Image.open(first_image): pass
        except Exception as e:
            print(f"Bad image: {e}")
            return


        # Advance_Class_Page = next(f for cls,f in self.controller.frames.items() if cls.__name__=="Advance_Class_Selection_Page")
        Advance_Class_Page = self.controller.frames.get("Advance_Class_Selection_Page")
        Model_Tuning_Page = self.controller.frames.get("Model_Tuning_Page")

        use_advanced = Advance_Class_Page.Enable_Value.get()

        custom_test_active = Advance_Class_Page.Enable_Class_Weights.get()

        Model_Tuning_Conditions = Model_Tuning_Page.conditions

        # build datasets and class_weight_dict 
        if use_advanced:
            output = Model_Training.Advance_Image_Set_Selection(self)

            if output is None:
                print("[ERROR] Advance_Image_Set_Selection aborted due to custom test dir/input mismatch.")
                return
            
            (Raw_Data, train, val, test, train_File_value_Dict, val_File_value_Dict, test_File_value_Dict, free_File_Names_Dict, class_weight_dict) = output
            print("✔ Dataset split complete.")


            if custom_test_active:
                test_labels_ds = test.unbatch().map(lambda x, y: y)
                all_test = np.array(list(test_labels_ds.as_numpy_iterator()), dtype=int)
                n_classes = len(Raw_Data.class_names)
                classes = np.arange(n_classes)
                if all_test.size > 0:
                    weights = class_weight.compute_class_weight('balanced', classes=classes, y=all_test)
                    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
                else:
                    class_weight_dict = {i: 1.0 for i in range(n_classes)}

                # class_weight_dict = {i: 1.0 for i in range(n_classes)}
                print("✔ Recomputed weights from TEST set:", class_weight_dict)

        else:
            Raw_Data, train, val, test = Model_Training.Basic_Image_Set_Selection(self)
            train_labels_ds = train.unbatch().map(lambda x, y: y)
            all_train = np.array(list(train_labels_ds.as_numpy_iterator()), dtype=int)
            n_classes = len(Raw_Data.class_names)
            classes = np.arange(n_classes)
            if all_train.size > 0:
                weights = class_weight.compute_class_weight( 'balanced', classes=classes, y=all_train  )
                class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
            else:
                class_weight_dict = {i: 1.0 for i in range(n_classes)}
            print("✔ Computed weights from TRAIN set:", class_weight_dict)

        # ——— print summary info ———
        class_names = Raw_Data.class_names
        num_classes = len(class_names)
        print("\n\n\nTraining :")
        print("Class Names:", class_names)
        print("Number of classes =", num_classes)

        print_dataset_counts(train, class_names, "Training")
        print_dataset_counts(val,   class_names, "Validation")
        print_dataset_counts(test,  class_names, "Test")

        print(self.controller.model.summary())

        num_epochs = int(self.Epoch_size_text_selected.get())
        history = { 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'per_class': {} }

        refresh_train_cnter = 0

        if num_epochs > 0:
            for epoch in range(num_epochs):
                if not self.controller.running:
                    break

                # ——— refresh train set if requested ———
                if use_advanced:
                    ebfr = int(Advance_Class_Page.Epochs_Before_Refresh.get())
                    if ebfr != 0 and refresh_train_cnter >= ebfr:
                        train = Model_Training.Advance_Image_Set_Selection( self, replace_train=True, free_File_Names_Dict=free_File_Names_Dict, train_File_value_Dict=train_File_value_Dict )
                        refresh_train_cnter = 0
                        print("✔ Training set replaced")
                refresh_train_cnter += 1

                class_weights_vec = tf.constant( [class_weight_dict[j] for j in range(num_classes)], dtype=tf.float32 )

                # # ——— training loop ———

                train_loss_total = 0.0
                train_acc_total = 0.0
                train_steps = 0

                len_train= len(train)
                for i, (x_batch, y_batch) in enumerate(train):
                    if not self.controller.running:
                        break
                    batch_loss, batch_acc = trainer.Model_training_Step(  self.controller.model, x_batch, y_batch, class_weights_vec  , Conditions = Model_Tuning_Conditions)
                    train_loss_total += batch_loss
                    train_acc_total += batch_acc
                    train_steps += 1

                    # throttle printing
                    if i % 10 == 0:
                        tf.print(i, "/",len_train  , " - loss:", batch_loss, " - acc:", batch_acc)

                avg_train_loss = train_loss_total / max(train_steps, 1)
                avg_train_acc  = train_acc_total  / max(train_steps, 1)

                # VALIDATION (single pass; no predict/evaluate)
                val_loss_total = 0.0
                val_steps = 0
                all_true = []
                all_pred = []

                for x_val, y_val in val:
                    batch_loss, preds = trainer.Model_val_Step( self.controller.model, x_val, y_val, class_weights_vec  )
                    val_loss_total += batch_loss
                    val_steps += 1
                    all_true.extend(y_val.numpy())
                    all_pred.extend(preds.numpy())

                avg_val_loss = val_loss_total / max(val_steps, 1)
                all_true = np.array(all_true)
                all_pred = np.array(all_pred)
                avg_val_acc = sklearn.metrics.accuracy_score(all_true, all_pred)
                avg_val_pre = sklearn.metrics.precision_score(all_true, all_pred, average='macro', zero_division=0)
                avg_val_rec = sklearn.metrics.recall_score(all_true, all_pred, average='macro', zero_division=0)

                # per‐class metrics
                for idx, cname in enumerate(class_names):
                    if cname not in history['per_class']:
                        history['per_class'][cname] = { 'val_accuracy': [], 'val_precision': [], 'val_recall': [] }
                    TP = np.sum((all_true == idx) & (all_pred == idx))
                    FP = np.sum((all_true != idx) & (all_pred == idx))
                    FN = np.sum((all_true == idx) & (all_pred != idx))
                    sup = np.sum(all_true == idx)

                    history['per_class'][cname]['val_accuracy'].append( TP / sup if sup else 0.0 )
                    history['per_class'][cname]['val_precision'].append( TP / (TP + FP) if (TP + FP) else 0.0 )
                    history['per_class'][cname]['val_recall'].append( TP / (TP + FN) if (TP + FN) else 0.0 )

                # record
                history['loss'].append(float(avg_train_loss))
                history['accuracy'].append(float(avg_train_acc))
                # history['loss'].append(float(avg_train_loss.numpy()))
                # history['accuracy'].append(float(avg_train_acc.numpy()))
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(avg_val_acc)
                history['val_precision'].append(avg_val_pre)
                history['val_recall'].append(avg_val_rec)

                print(
                    f"Epoch {epoch+1}/{num_epochs}"
                    f" - loss: {avg_train_loss:.4f}"
                    f" - val_loss: {avg_val_loss:.4f}"
                    f" - accuracy: {avg_train_acc:.2f}"
                    f" - val_accuracy: {avg_val_acc:.2f}"
                    f" - val_precision: {avg_val_pre:.2f}"
                    f" - val_recall: {avg_val_rec:.2f}"
                )

                # # --- Confusion Matrix on Validation Set each epoch ---
                # Confusion_Page = next( (f for cls, f in self.controller.frames.items() if cls.__name__ == "Show_Confusion_Page"), None )
                # if Confusion_Page:
                #     # clear old widgets
                #     for w in Confusion_Page.Confusion_Canvas_Frame.winfo_children():
                #         w.destroy()

                #     # compute and display
                #     cm = sklearn.metrics.confusion_matrix( all_true_labels, all_pred_labels , labels = list(range(len(class_names))) )
                #     labels = [Model_Training.translate_label(self, c) for c in class_names]
                #     fig_cm, ax_cm = plt.subplots(figsize=(6, 6), dpi=120)
                #     disp = sklearn.metrics.ConfusionMatrixDisplay( confusion_matrix=cm , display_labels=labels )
                #     disp.plot(ax=ax_cm, cmap='Blues', values_format='.2f')
                #     ax_cm.set_title(f"Validation Confusion Matrix\nEpoch {epoch+1}")
                #     ax_cm.tick_params(axis='both', which='major', labelsize=6)
                #     fig_cm.tight_layout()

                #     canvas_cm = FigureCanvasTkAgg(
                #         fig_cm, master=Confusion_Page.Confusion_Canvas_Frame
                #     )
                #     canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                #     canvas_cm.draw()
                #     NavigationToolbar2Tk(canvas_cm, Confusion_Page.Confusion_Canvas_Frame, pack_toolbar=False).pack(side=tk.LEFT, fill=tk.X)
                #     plt.close(fig_cm)
    
                if not hasattr(self, 'live_plots_initialized'):
                    # monitor_page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Monitor_Training_Page") , None )
                    monitor_page = self.controller.frames.get("Monitor_Training_Page")
                    if monitor_page is None:
                        print("Monitor_Training_Page not found in frames.")
                        return
                    
                    self.live_plots_initialized = True
                    self.monitor_page = monitor_page
                    self.train_fig_acc, self.train_ax_acc = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_acc = FigureCanvasTkAgg(self.train_fig_acc, master=monitor_page.First_Canvas_Row_Accuracy)
                    self.canvas_acc.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_fig_pre, self.train_ax_pre = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_pre = FigureCanvasTkAgg(self.train_fig_pre, master=monitor_page.First_Canvas_Row_Precision)
                    self.canvas_pre.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_fig_recall, self.train_ax_recall = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_recall = FigureCanvasTkAgg(self.train_fig_recall, master=monitor_page.First_Canvas_Row_Recall)
                    self.canvas_recall.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_fig_loss, self.train_ax_loss = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_loss = FigureCanvasTkAgg(self.train_fig_loss, master=monitor_page.First_Canvas_Row_Loss)
                    self.canvas_loss.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_delta_loss_fig, self.train_delta_loss_ax = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_delta_loss = FigureCanvasTkAgg(self.train_delta_loss_fig, master=monitor_page.Second_Canvas_delta_Loss)
                    self.canvas_delta_loss.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                if not hasattr(self, 'live_class_plots_initialized'):
                    self.live_class_plots_initialized = True
                    self.train_class_fig_acc, self.train_class_ax_acc = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_class_acc = FigureCanvasTkAgg(self.train_class_fig_acc, master=monitor_page.Second_Canvas_Row_Class_Accuracy)
                    self.canvas_class_acc.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_class_fig_pre, self.train_class_ax_pre = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_class_pre = FigureCanvasTkAgg(self.train_class_fig_pre, master=monitor_page.Second_Canvas_Row_Class_Precision)
                    self.canvas_class_pre.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                    self.train_class_fig_rec, self.train_class_ax_rec = plt.subplots(figsize=(4, 4), dpi=100)
                    self.canvas_class_rec = FigureCanvasTkAgg(self.train_class_fig_rec, master=monitor_page.Second_Canvas_Row_Class_Recall)
                    self.canvas_class_rec.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                self.train_ax_acc.cla()
                self.train_ax_acc.plot(history['accuracy'], color='teal', label='accuracy')
                self.train_ax_acc.plot(history['val_accuracy'], color='orange', label='val_accuracy')
                self.train_ax_acc.set_xlabel("Epoch")
                self.train_ax_acc.set_ylabel("Accuracy")
                self.train_ax_acc.set_title("Accuracy", fontsize=17)
                self.train_ax_acc.set_ylim(0, 1.1)
                self.train_ax_acc.legend(loc="upper left")
                self.canvas_acc.draw()
                self.train_ax_pre.cla()
                self.train_ax_pre.plot(history['val_precision'], color='orange', label='val_precision')
                self.train_ax_pre.set_xlabel("Epoch")
                self.train_ax_pre.set_ylabel("Precision")
                self.train_ax_pre.set_title("Precision", fontsize=14)
                self.train_ax_pre.set_ylim(0, 1.1)
                self.train_ax_pre.legend(loc="upper left", prop={'size': 7})
                self.canvas_pre.draw()
                self.train_ax_recall.cla()
                self.train_ax_recall.plot(history['val_recall'], color='orange', label='val_recall')
                self.train_ax_recall.set_xlabel("Epoch")
                self.train_ax_recall.set_ylabel("Recall")
                self.train_ax_recall.set_title("Recall", fontsize=14)
                self.train_ax_recall.set_ylim(0, 1.1)
                self.train_ax_recall.legend(loc="upper left", prop={'size': 7})
                self.canvas_recall.draw()
                self.train_ax_loss.cla()
                self.train_ax_loss.plot(history['loss'], color='blue', label='loss')
                self.train_ax_loss.plot(history['val_loss'], color='orange', label='val_loss')
                self.train_ax_loss.set_xlabel("Epoch")
                self.train_ax_loss.set_ylabel("Loss")
                self.train_ax_loss.set_title("Loss", fontsize=14)
                self.train_ax_loss.legend(loc="upper left")
                self.canvas_loss.draw()
                self.train_delta_loss_ax.cla()
                self.train_delta_loss_ax.plot(np.array(history['val_loss']) - np.array(history['loss']))
                self.train_delta_loss_ax.hlines(y=0, xmin=0, xmax=len(history['val_loss']), colors='black', linestyles='--')
                self.train_delta_loss_ax.set_xlabel("Epoch")
                self.train_delta_loss_ax.set_ylabel(r"$\Delta$ Loss")
                self.train_delta_loss_ax.set_title(r"$\Delta$ Loss", fontsize=14)
                self.canvas_delta_loss.draw()
                default_line_styles = ['-', '--', ':', '-.']
                possible_line_styles = default_line_styles.copy()
                for _ in range(len(class_names) - len(default_line_styles)):
                    possible_line_styles.append((np.random.randint(0,10),(np.random.randint(0,10),np.random.randint(0,10),np.random.randint(0,10),np.random.randint(0,10))))
                self.train_class_ax_acc.cla()
                Model_Training.per_class_plot(self, self.train_class_ax_acc, class_names, history['per_class'], 'val_accuracy', possible_line_styles)
                self.train_class_ax_acc.set_xlabel("Epoch")
                self.train_class_ax_acc.set_ylabel("val_Accuracy")
                self.train_class_ax_acc.legend(loc="upper left", prop={'size': 7})
                self.train_class_ax_acc.set_title("per class accuracy", fontsize=14)
                self.train_class_ax_acc.set_ylim(0, 1.1)
                self.canvas_class_acc.draw()
                self.train_class_ax_pre.cla()
                Model_Training.per_class_plot(self, self.train_class_ax_pre, class_names, history['per_class'], 'val_precision', possible_line_styles)
                self.train_class_ax_pre.set_xlabel("Epoch")
                self.train_class_ax_pre.set_ylabel("val_Precision")
                self.train_class_ax_pre.legend(loc="upper left", prop={'size': 7})
                self.train_class_ax_pre.set_title("per class precision", fontsize=14)
                self.train_class_ax_pre.set_ylim(0, 1.1)
                self.canvas_class_pre.draw()
                self.train_class_ax_rec.cla()
                Model_Training.per_class_plot(self, self.train_class_ax_rec, class_names, history['per_class'], 'val_recall', possible_line_styles)
                self.train_class_ax_rec.set_xlabel("Epoch")
                self.train_class_ax_rec.set_ylabel("val_Recall")
                self.train_class_ax_rec.legend(loc="upper left", prop={'size': 7})
                self.train_class_ax_rec.set_title("per class recall", fontsize=14)
                self.train_class_ax_rec.set_ylim(0, 1.1)
                self.canvas_class_rec.draw()
                # self.controller.root.update_idletasks()
                # self.controller.root.update()
            # -------------------- PLOTTING ------------------------


            # Dynamically find the Monitor_Training_Page frame by class name
            if not hasattr(self, 'live_plots_initialized'):
                # monitor_page = next( (frame for cls, frame in self.controller.frames.items() if cls.__name__ == "Monitor_Training_Page") , None )
                monitor_page = self.controller.frames.get("Monitor_Training_Page")
                if monitor_page is None:
                    print("Monitor_Training_Page not found in frames.")
                    return


            #Destroy all children in each sub-frame to clear old plots
            for child in monitor_page.First_Canvas_Row_Accuracy.winfo_children():
                child.destroy()
            for child in monitor_page.First_Canvas_Row_Precision.winfo_children():
                child.destroy()
            for child in monitor_page.First_Canvas_Row_Recall.winfo_children():
                child.destroy()
            for child in monitor_page.First_Canvas_Row_Loss.winfo_children():
                child.destroy()

            for child in monitor_page.Second_Canvas_Row_Class_Accuracy.winfo_children():
                child.destroy()
            for child in monitor_page.Second_Canvas_Row_Class_Precision.winfo_children():
                child.destroy()
            for child in monitor_page.Second_Canvas_Row_Class_Recall.winfo_children():
                child.destroy()
            for child in monitor_page.Second_Canvas_delta_Loss.winfo_children():
                child.destroy()



            self.train_fig_acc, self.train_ax_acc = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_2 = FigureCanvasTkAgg(self.train_fig_acc, master=monitor_page.First_Canvas_Row_Accuracy)
            canvas_2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_ax_acc.plot(history['accuracy'], color='teal', label='accuracy')
            self.train_ax_acc.plot(history['val_accuracy'], color='orange', label='val_accuracy')
            self.train_ax_acc.set_xlabel("Epoch")
            self.train_ax_acc.set_ylabel("Accuracy")
            self.train_ax_acc.set_title("Accuracy", fontsize=17)
            self.train_ax_acc.set_ylim(0, 1.1)
            self.train_ax_acc.legend(loc="upper left" )

            canvas_2.draw()

            toolbar_2 = NavigationToolbar2Tk(canvas_2, monitor_page.First_Canvas_Row_Accuracy, pack_toolbar=False)
            toolbar_2.update()
            toolbar_2.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_fig_acc)


            self.train_fig_pre , self.train_ax_pre = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_3 = FigureCanvasTkAgg(self.train_fig_pre, master=monitor_page.First_Canvas_Row_Precision)
            canvas_3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_ax_pre.plot(history['val_precision'], color='orange', label='val_precision')
            self.train_ax_pre.set_xlabel("Epoch")
            self.train_ax_pre.set_ylabel("Precision")
            self.train_ax_pre.set_title("Precision", fontsize=14)
            self.train_ax_pre.set_ylim(0, 1.1)
            self.train_ax_pre.legend(loc="upper left"   , prop={'size': 7})

            canvas_3.draw()

            toolbar_3 = NavigationToolbar2Tk(canvas_3, monitor_page.First_Canvas_Row_Precision, pack_toolbar=False)
            toolbar_3.update()
            toolbar_3.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_fig_pre)


            self.train_fig_recall , self.train_ax_recall = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_4 = FigureCanvasTkAgg(self.train_fig_recall, master=monitor_page.First_Canvas_Row_Recall)
            canvas_4.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_ax_recall.plot(history['val_recall'], color='orange', label='val_recall')
            self.train_ax_recall.set_xlabel("Epoch")
            self.train_ax_recall.set_ylabel("Recall")
            self.train_ax_recall.set_title("Recall", fontsize=14)
            self.train_ax_recall.set_ylim(0, 1.1)
            self.train_ax_recall.legend(loc="upper left" ,  prop={'size': 7})

            canvas_4.draw()

            toolbar_4 = NavigationToolbar2Tk(canvas_4, monitor_page.First_Canvas_Row_Recall, pack_toolbar=False)
            toolbar_4.update()
            toolbar_4.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_fig_recall)


            self.train_fig_loss , self.train_ax_loss = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_5 = FigureCanvasTkAgg(self.train_fig_loss, master=monitor_page.First_Canvas_Row_Loss)
            canvas_5.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_ax_loss.plot(history['loss'], color='blue', label='loss')
            self.train_ax_loss.plot(history['val_loss'], color='orange', label='val_loss')
            self.train_ax_loss.set_xlabel("Epoch")
            self.train_ax_loss.set_ylabel("Loss")
            self.train_ax_loss.set_title("Loss", fontsize=14)
            self.train_ax_loss.legend(loc="upper left")

            monitor_page.loss_fig_ax = self.train_ax_loss
            monitor_page.loss_fig_canvas = canvas_5

            canvas_5.draw()

            toolbar_5 = NavigationToolbar2Tk(canvas_5, monitor_page.First_Canvas_Row_Loss, pack_toolbar=False)
            toolbar_5.update()
            toolbar_5.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_fig_loss)



            self.train_class_fig_acc , self.train_class_ax_acc = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_6 = FigureCanvasTkAgg(self.train_class_fig_acc, master=monitor_page.Second_Canvas_Row_Class_Accuracy)
            canvas_6.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


            default_line_styles = ['-', '--' , ':' , '-.'] 
            possible_line_styles = [ i for i in default_line_styles ]
            

            for _ in range( len(class_names) - len(default_line_styles) ):
                possible_line_styles.append( (  np.random.randint(0,10) , ( np.random.randint(0,10) ,  np.random.randint(0,10) ,  np.random.randint(0,10) ,  np.random.randint(0,10)) )  )



            Model_Training.per_class_plot( self = self , ax_i =  self.train_class_ax_acc , class_names = class_names , plot_data = history['per_class'] , plot_data_key = 'val_accuracy' , possible_line_styles = possible_line_styles )


            self.train_class_ax_acc.set_xlabel("Epoch")
            self.train_class_ax_acc.set_ylabel("val_Accuracy")
            self.train_class_ax_acc.legend(loc="upper left" , prop={'size': 7})
            self.train_class_ax_acc.set_title("per class accuracy", fontsize=14)
            self.train_class_ax_acc.set_ylim(0, 1.1)

            monitor_page.class_accuracy_ax = self.train_class_ax_acc
            monitor_page.class_accuracy_canvas = canvas_6

            canvas_6.draw()

            toolbar_6 = NavigationToolbar2Tk(canvas_6, monitor_page.Second_Canvas_Row_Class_Accuracy, pack_toolbar=False)
            toolbar_6.update()
            toolbar_6.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_class_fig_acc)



            self.train_class_fig_pre , self.train_class_ax_pre = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_7 = FigureCanvasTkAgg(self.train_class_fig_pre, master=monitor_page.Second_Canvas_Row_Class_Precision)
            canvas_7.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)




            Model_Training.per_class_plot( self = self , ax_i =  self.train_class_ax_pre , class_names = class_names , plot_data = history['per_class'] , plot_data_key = 'val_precision' , possible_line_styles = possible_line_styles )

            self.train_class_ax_pre.set_xlabel("Epoch")
            self.train_class_ax_pre.set_ylabel("val_Precision")
            self.train_class_ax_pre.legend( loc="upper left" ,prop=dict(size=7) ) 
            self.train_class_ax_pre.set_title("per class precision", fontsize=14)
            self.train_class_ax_pre.set_ylim(0, 1.1)

            monitor_page.class_precision_ax = self.train_class_ax_pre
            monitor_page.class_precision_canvas = canvas_7
            canvas_7.draw()

            toolbar_7 = NavigationToolbar2Tk(canvas_7, monitor_page.Second_Canvas_Row_Class_Precision, pack_toolbar=False)
            toolbar_7.update()
            toolbar_7.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_class_fig_pre)



            self.train_class_fig_rec , self.train_class_ax_rec = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_8 = FigureCanvasTkAgg(self.train_class_fig_rec, master=monitor_page.Second_Canvas_Row_Class_Recall)
            canvas_8.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            Model_Training.per_class_plot( self = self , ax_i =  self.train_class_ax_rec , class_names = class_names , plot_data = history['per_class'] , plot_data_key = 'val_recall', possible_line_styles = possible_line_styles )


            self.train_class_ax_rec.set_xlabel("Epoch")
            self.train_class_ax_rec.set_ylabel("val_Recall")
            self.train_class_ax_rec.legend(loc="upper left" ,  prop=dict(size=7) ) 
            self.train_class_ax_rec.set_title("per class recall", fontsize=14)
            self.train_class_ax_rec.set_ylim(0, 1.1)

            monitor_page.class_recall_ax = self.train_class_ax_rec
            monitor_page.class_recall_canvas = canvas_8
            canvas_8.draw()

            toolbar_8 = NavigationToolbar2Tk(canvas_8, monitor_page.Second_Canvas_Row_Class_Recall, pack_toolbar=False)
            toolbar_8.update()
            toolbar_8.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_class_fig_rec)



            self.train_delta_loss_fig , self.train_delta_loss_ax = plt.subplots(figsize=(4, 4), dpi=100)

            canvas_9 = FigureCanvasTkAgg(self.train_delta_loss_fig, master=monitor_page.Second_Canvas_delta_Loss)
            canvas_9.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.train_delta_loss_ax.plot( (np.array( history['val_loss'] ) - np.array( history['loss'] )) ,  )
            self.train_delta_loss_ax.hlines( y = 0 , xmin = 0 , xmax= len(  history['val_loss'] )  , colors='black' , linestyles='--')

            self.train_delta_loss_ax.set_xlabel("Epoch")
            self.train_delta_loss_ax.set_ylabel(r"$\Delta$ Loss")
            self.train_delta_loss_ax.set_title(r"$\Delta$ Loss", fontsize=14)

            canvas_9.draw()

            toolbar_9 = NavigationToolbar2Tk(canvas_9, monitor_page.Second_Canvas_delta_Loss, pack_toolbar=False)
            toolbar_9.update()
            toolbar_9.pack(side=tk.LEFT, fill=tk.X)
            plt.close(self.train_delta_loss_fig)

        
        if int(num_epochs) != 0 :
            translated = [ Model_Training.translate_label(self, cls) for cls in class_names ]
            monitor_page.setup_class_filter( self, translated)



        all_test_true_labels = []
        all_test_pred_labels = []

        for x_val_batch, y_val_batch in test:
            predictions = self.controller.model.predict(x_val_batch)
            rounded_preds = np.argmax(predictions, axis=-1)

            all_test_true_labels.extend(y_val_batch.numpy())
            all_test_pred_labels.extend(rounded_preds)

        self.confusion_fig, self.confusion_ax = plt.subplots(figsize=(6, 6), dpi=160)
        plt.subplots_adjust(left=0.216, right=1, top=0.96, bottom=0.212, wspace=0.4, hspace=0.3)


        Confusion_Page = self.controller.frames.get("Show_Confusion_Page")


        if Confusion_Page is None:
            print("Show_Confusion_Page not found in frames.")
            return

        for child in Confusion_Page.Confusion_Canvas_Frame.winfo_children():
            child.destroy()

        Confusion_Canvas = FigureCanvasTkAgg(self.confusion_fig, master=Confusion_Page.Confusion_Canvas_Frame)
        Confusion_Canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        all_test_true_labels_cm = list(map( lambda i : Model_Training.translate_label( self = self, label = class_names[i] )  , all_test_true_labels))
        all_test_pred_labels_cm = list(map( lambda i : Model_Training.translate_label( self = self, label = class_names[i] )  , all_test_pred_labels))

        Confusion_dict_data = pd.DataFrame( data = {'Actual Interaction' : all_test_true_labels_cm , 'Predicted Interaction' : all_test_pred_labels_cm } )

        confusion_matrix = pd.crosstab( Confusion_dict_data['Actual Interaction'] , Confusion_dict_data['Predicted Interaction'] ,
                                        rownames= ['Actual Interaction'] , colnames= ['Predicted Interaction'] , 
                                        normalize= 'index')

        sn.heatmap( confusion_matrix, annot=True , ax=self.confusion_ax  , xticklabels = True, yticklabels = True , fmt=".2f" , annot_kws={'size': 7} , cbar_kws={"shrink": 0.7})

        self.confusion_ax.tick_params( axis='both', which='major', labelsize= 6 )
        self.confusion_ax.set_xlabel('Predicted Interaction')
        self.confusion_ax.set_ylabel('Actual Interaction')


        Confusion_Canvas.draw()

        toolbar_confusion = NavigationToolbar2Tk(Confusion_Canvas, Confusion_Page.Confusion_Canvas_Frame, pack_toolbar=False)
        toolbar_confusion.update()
        toolbar_confusion.pack(side=tk.LEFT, fill=tk.X)

        plt.close(self.confusion_fig)

        Confusion_Page.confusion_fig_ax = self.confusion_ax
        Confusion_Page.confusion_fig_canvas = Confusion_Canvas


        # Main_Run_File.Show_Model_Filters_Page.Model_Filters_Ready(self)
        self.controller.Show_Model_Filters_Page.Model_Filters_Ready()


        # Main_Run_File.Frame_Manager.cancel_process(self)
        self.controller.Frame_Manager.cancel_process(self)

        self.controller.model.class_name_dict = { i : Model_Training.translate_label(self ,j)  for i , j in enumerate(class_names)} 


        self.controller.model.history_dict = history

        self.controller.test_images = test

        


        return




    def per_class_plot(self, ax_i , class_names , plot_data , plot_data_key , possible_line_styles ):

        cnter  = 0 
        default_line_styles = ['-', '--' , ':' , '-.'] 
        for c_name  in  class_names :

            label = Model_Training.translate_label( self= self , label = c_name)

            ax_i.plot( plot_data[c_name][ plot_data_key ] , label = label , linestyle = possible_line_styles[ cnter ] )
            
            cnter +=1
        return
    

    def translate_label(self,label):

        if label == 'Neutrino_Electron_Scattering':
            translated_label = r'$\nu - e$ (scattering)'
        
        elif label == 'Neutrino_Electron_Scattering_NC':
            translated_label = r'$\nu - e$ (scattering) (NC)'

        elif label == 'Neutrino_Electron_Scattering_CC':
            translated_label = r'$\nu - e$ (scattering) (CC)'
        
        elif label == 'Anti_Electron_Neutrino_CC':
            translated_label = r'$\bar{\nu_e} - e$ (CC)'

        elif label == 'Electron_Neutrino':
            translated_label = r'$\nu_{e}$ - (NC/CC)'
        
        elif label == 'Electron_Neutrino_NC':
            translated_label = r'$\nu_{e}$ - NC'

        elif label == 'Electron_Neutrino_CC':
            translated_label = r'$\nu_{e}$ - CC'

        elif label == 'Neutrino_NC':
            translated_label = r'$\nu$ - NC'

        elif label == 'Neutrino_CC_Other':
            translated_label = r'$\nu$ - CC (Other)'

        elif label == 'Neutrino_NC_Other':
            translated_label = r'$\nu$ - NC (Other)'

        elif label == 'Other':
            translated_label = 'Other'

        else:
            translated_label = label


        return translated_label
