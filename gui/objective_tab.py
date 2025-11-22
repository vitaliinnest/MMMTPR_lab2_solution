import tkinter as tk
from tkinter import ttk, messagebox
from database import DatabaseManager


class ObjectiveTab:
    """Tab for computing objective function manually."""
    
    def __init__(self, parent: ttk.Frame, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets for the objective tab."""
        frm = ttk.Frame(self.parent)
        frm.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frm, text="x1 (fastText):").grid(row=0, column=0, sticky="e")
        ttk.Label(frm, text="x2 (SBERT):").grid(row=0, column=2, sticky="e")
        ttk.Label(frm, text="x3 (хмарна):").grid(row=0, column=4, sticky="e")

        self.entry_x1 = ttk.Entry(frm, width=10)
        self.entry_x2 = ttk.Entry(frm, width=10)
        self.entry_x3 = ttk.Entry(frm, width=10)

        self.entry_x1.grid(row=0, column=1, padx=3, pady=2)
        self.entry_x2.grid(row=0, column=3, padx=3, pady=2)
        self.entry_x3.grid(row=0, column=5, padx=3, pady=2)

        btn_calc = ttk.Button(frm, text="Обчислити", command=self._compute_objective)
        btn_calc.grid(row=1, column=0, columnspan=2, pady=4)

        self.text_obj = tk.Text(self.parent, width=100, height=20)
        self.text_obj.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _compute_objective(self):
        """Compute and display objective function value."""
        models = self.db.load_models()
        
        if len(models) < 3:
            messagebox.showerror("Помилка", "Функція розрахована щонайменше на 3 моделі")
            return
        
        try:
            x1 = float(self.entry_x1.get().strip())
            x2 = float(self.entry_x2.get().strip())
            x3 = float(self.entry_x3.get().strip())
        except ValueError:
            messagebox.showerror("Помилка", "Невірні значення x1, x2, x3")
            return
        
        x = [x1, x2, x3]
        s = sum(x)
        objective = sum(models[i].cost_per_query * x[i] for i in range(3))
        p_val = sum(models[i].precision * x[i] for i in range(3))
        r_val = sum(models[i].recall * x[i] for i in range(3))
        t_val = sum(models[i].response_time * x[i] for i in range(3))
        
        self.text_obj.delete("1.0", tk.END)
        self.text_obj.insert(tk.END, f"Сума часток x1+x2+x3 = {s:.4f}\n")
        self.text_obj.insert(tk.END, f"Цільова функція Z = {objective:.6f}\n")
        self.text_obj.insert(tk.END, f"Середня точність = {p_val:.4f}\n")
        self.text_obj.insert(tk.END, f"Середня повнота = {r_val:.4f}\n")
        self.text_obj.insert(tk.END, f"Середній час відповіді = {t_val:.4f}\n")
