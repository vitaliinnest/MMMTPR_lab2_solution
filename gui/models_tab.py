import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
from database import DatabaseManager


class ModelsTab:
    """Tab for managing embedding models and constraints."""
    
    def __init__(self, parent: ttk.Frame, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.selected_model_id: Optional[int] = None
        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets for the models tab."""
        self._create_models_section()
        self._create_constraints_section()

    def _create_models_section(self):
        """Create the models table and edit controls."""
        upper = ttk.Frame(self.parent)
        upper.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("id", "name", "var", "cost", "time", "precision", "recall")
        self.tree = ttk.Treeview(upper, columns=columns, show="headings", height=8)
        
        for col, txt in zip(columns, ["ID", "Модель", "Змінна", "Вартість", "Час", "Точність", "Повнота"]):
            self.tree.heading(col, text=txt)
        
        self.tree.column("id", width=40, anchor="center")
        self.tree.column("name", width=170)
        self.tree.column("var", width=60, anchor="center")
        self.tree.column("cost", width=80, anchor="center")
        self.tree.column("time", width=80, anchor="center")
        self.tree.column("precision", width=80, anchor="center")
        self.tree.column("recall", width=80, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(upper, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Edit controls
        lower = ttk.Frame(self.parent)
        lower.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(lower, text="Назва:").grid(row=0, column=0, sticky="e")
        ttk.Label(lower, text="Змінна (x_i):").grid(row=0, column=2, sticky="e")
        ttk.Label(lower, text="Вартість c:").grid(row=0, column=4, sticky="e")
        ttk.Label(lower, text="Час t:").grid(row=0, column=6, sticky="e")
        ttk.Label(lower, text="Точність p:").grid(row=1, column=0, sticky="e")
        ttk.Label(lower, text="Повнота r:").grid(row=1, column=2, sticky="e")

        self.entry_name = ttk.Entry(lower, width=20)
        self.entry_var = ttk.Entry(lower, width=8)
        self.entry_cost = ttk.Entry(lower, width=10)
        self.entry_time = ttk.Entry(lower, width=10)
        self.entry_precision = ttk.Entry(lower, width=10)
        self.entry_recall = ttk.Entry(lower, width=10)

        self.entry_name.grid(row=0, column=1, padx=3, pady=2)
        self.entry_var.grid(row=0, column=3, padx=3, pady=2)
        self.entry_cost.grid(row=0, column=5, padx=3, pady=2)
        self.entry_time.grid(row=0, column=7, padx=3, pady=2)
        self.entry_precision.grid(row=1, column=1, padx=3, pady=2)
        self.entry_recall.grid(row=1, column=3, padx=3, pady=2)

        btn_update = ttk.Button(lower, text="Зберегти вибрану модель", command=self._save_selected_model)
        btn_reload = ttk.Button(lower, text="Оновити з БД", command=self.reload_models)
        btn_add = ttk.Button(lower, text="Додати нову модель", command=self._add_new_model)

        btn_update.grid(row=2, column=0, columnspan=2, pady=4)
        btn_reload.grid(row=2, column=2, columnspan=2, pady=4)
        btn_add.grid(row=2, column=4, columnspan=2, pady=4)

        self.refresh_tree()

    def _create_constraints_section(self):
        """Create the constraints edit section."""
        sep = ttk.Separator(self.parent, orient="horizontal")
        sep.pack(fill=tk.X, padx=5, pady=5)

        constr_frame = ttk.Frame(self.parent)
        constr_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(constr_frame, text="Pmin:").grid(row=0, column=0, sticky="e")
        ttk.Label(constr_frame, text="Rmin:").grid(row=0, column=2, sticky="e")
        ttk.Label(constr_frame, text="Tmax:").grid(row=0, column=4, sticky="e")

        self.entry_pmin = ttk.Entry(constr_frame, width=10)
        self.entry_rmin = ttk.Entry(constr_frame, width=10)
        self.entry_tmax = ttk.Entry(constr_frame, width=10)

        self.entry_pmin.grid(row=0, column=1, padx=3, pady=2)
        self.entry_rmin.grid(row=0, column=3, padx=3, pady=2)
        self.entry_tmax.grid(row=0, column=5, padx=3, pady=2)

        btn_save_constr = ttk.Button(constr_frame, text="Зберегти обмеження", command=self._save_constraints)
        btn_save_constr.grid(row=1, column=0, columnspan=2, pady=4)

        btn_reload_constr = ttk.Button(constr_frame, text="Оновити обмеження", command=self._reload_constraints)
        btn_reload_constr.grid(row=1, column=2, columnspan=2, pady=4)

        self._fill_constraints_entries()

    def refresh_tree(self):
        """Refresh the models table."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        models = self.db.load_models()
        for m in models:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    m.id,
                    m.name,
                    m.var_code,
                    f"{m.cost_per_query:.2f}",
                    f"{m.response_time:.2f}",
                    f"{m.precision:.2f}",
                    f"{m.recall:.2f}",
                ),
            )

    def reload_models(self):
        """Reload models from database."""
        self.refresh_tree()

    def _fill_constraints_entries(self):
        """Fill constraint entry fields with current values."""
        constraints = self.db.load_constraints()
        self.entry_pmin.delete(0, tk.END)
        self.entry_rmin.delete(0, tk.END)
        self.entry_tmax.delete(0, tk.END)
        self.entry_pmin.insert(0, str(constraints.p_min))
        self.entry_rmin.insert(0, str(constraints.r_min))
        self.entry_tmax.insert(0, str(constraints.t_max))

    def _on_tree_select(self, event):
        """Handle model selection in the tree."""
        selected = self.tree.selection()
        if not selected:
            self.selected_model_id = None
            return
        
        item = self.tree.item(selected[0])
        vals = item["values"]
        self.selected_model_id = int(vals[0])
        
        self.entry_name.delete(0, tk.END)
        self.entry_var.delete(0, tk.END)
        self.entry_cost.delete(0, tk.END)
        self.entry_time.delete(0, tk.END)
        self.entry_precision.delete(0, tk.END)
        self.entry_recall.delete(0, tk.END)
        
        self.entry_name.insert(0, vals[1])
        self.entry_var.insert(0, vals[2])
        self.entry_cost.insert(0, vals[3])
        self.entry_time.insert(0, vals[4])
        self.entry_precision.insert(0, vals[5])
        self.entry_recall.insert(0, vals[6])

    def _save_selected_model(self):
        """Save changes to the selected model."""
        if self.selected_model_id is None:
            messagebox.showwarning("Увага", "Оберіть модель у таблиці")
            return
        
        name = self.entry_name.get().strip()
        var_code = self.entry_var.get().strip()
        
        if not var_code:
            messagebox.showerror("Помилка", "Поле змінної не може бути порожнім")
            return
        
        try:
            cost = float(self.entry_cost.get().strip())
            time_val = float(self.entry_time.get().strip())
            precision = float(self.entry_precision.get().strip())
            recall = float(self.entry_recall.get().strip())
        except ValueError:
            messagebox.showerror("Помилка", "Невірні числові значення")
            return
        
        cur = self.db.conn.cursor()
        cur.execute(
            """
            UPDATE embedding_models
            SET name = ?, var_code = ?, cost_per_query = ?, response_time = ?, precision_value = ?, recall_value = ?
            WHERE id = ?
            """,
            (name, var_code, cost, time_val, precision, recall, self.selected_model_id),
        )
        self.db.conn.commit()
        self.reload_models()
        messagebox.showinfo("Готово", "Модель оновлено")

    def _add_new_model(self):
        """Add a new model to the database."""
        name = self.entry_name.get().strip()
        var_code = self.entry_var.get().strip()
        
        if not name:
            messagebox.showerror("Помилка", "Вкажіть назву моделі")
            return
        
        models = self.db.load_models()
        if not var_code:
            existing_vars = {m.var_code for m in models}
            idx = 1
            while True:
                candidate = f"x{idx}"
                if candidate not in existing_vars:
                    var_code = candidate
                    break
                idx += 1
        
        try:
            cost = float(self.entry_cost.get().strip())
            time_val = float(self.entry_time.get().strip())
            precision = float(self.entry_precision.get().strip())
            recall = float(self.entry_recall.get().strip())
        except ValueError:
            messagebox.showerror("Помилка", "Невірні числові значення")
            return
        
        cur = self.db.conn.cursor()
        cur.execute(
            """
            INSERT INTO embedding_models
            (name, var_code, cost_per_query, response_time, precision_value, recall_value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, var_code, cost, time_val, precision, recall),
        )
        self.db.conn.commit()
        self.reload_models()
        messagebox.showinfo("Готово", f"Додано нову модель з змінною {var_code}")

    def _save_constraints(self):
        """Save constraint values to the database."""
        try:
            pmin = float(self.entry_pmin.get().strip())
            rmin = float(self.entry_rmin.get().strip())
            tmax = float(self.entry_tmax.get().strip())
        except ValueError:
            messagebox.showerror("Помилка", "Невірні числові значення")
            return
        
        from models import LPConstraints
        constraints = LPConstraints(p_min=pmin, r_min=rmin, t_max=tmax)
        self.db.save_constraints(constraints)
        messagebox.showinfo("Готово", "Обмеження оновлено")

    def _reload_constraints(self):
        """Reload constraints from the database."""
        self._fill_constraints_entries()
