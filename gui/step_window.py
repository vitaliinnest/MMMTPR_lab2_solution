import tkinter as tk
import time
from tkinter import messagebox
from typing import List
from models import EmbeddingModel, LPConstraints
from algorithms import greedy_solve_with_trace


class StepWindow(tk.Toplevel):
    """Window for step-by-step demonstration of the greedy algorithm."""
    
    def __init__(self, master, models: List[EmbeddingModel], constraints: LPConstraints, step: float):
        super().__init__(master)
        self.title("Покрокова демонстрація")
        self.models = models
        self.constraints = constraints
        self.step = step
        
        try:
            start = time.perf_counter()
            self.solution, self.trace = greedy_solve_with_trace(self.models, self.constraints, self.step)
            self.elapsed = time.perf_counter() - start
        except RuntimeError as e:
            messagebox.showerror("Помилка", str(e), parent=self)
            self.destroy()
            return
        
        self.index = 0
        self._create_widgets()
        self.show_next()

    def _create_widgets(self):
        """Create UI widgets for the step window."""
        self.text = tk.Text(self, width=90, height=25)
        self.text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text.insert(tk.END, f"Час виконання жадібного алгоритму = {self.elapsed:.6f} с\n\n")
        
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_next = tk.Button(btn_frame, text="Наступний крок", command=self.show_next)
        self.btn_next.pack(side=tk.LEFT, padx=5)
        
        self.btn_final = tk.Button(btn_frame, text="Підсумок", command=self.show_final)
        self.btn_final.pack(side=tk.LEFT, padx=5)

    def show_next(self):
        """Show the next step in the algorithm trace."""
        if self.index >= len(self.trace):
            self.text.insert(tk.END, "Кроки завершено\n")
            return
        
        r = self.trace[self.index]
        line = (
            f"Крок {r['step']}: модель={r['model_name']}, "
            f"додана частка={r['added_share']:.2f}, "
            f"x={[f'{v:.2f}' for v in r['x']]}, "
            f"P={r['precision']:.4f}, R={r['recall']:.4f}, "
            f"T={r['time']:.4f}, Z={r['objective']:.6f}\n"
        )
        self.text.insert(tk.END, line)
        self.text.see(tk.END)
        self.index += 1

    def show_final(self):
        """Show the final solution summary."""
        self.text.insert(tk.END, "\nПідсумковий розв'язок:\n")
        for i, m in enumerate(self.models):
            self.text.insert(tk.END, f"{m.name}: x{i+1} = {self.solution.x[i]:.2f}\n")
        self.text.insert(tk.END, f"Цільова функція Z = {self.solution.objective:.6f}\n")
        self.text.see(tk.END)
