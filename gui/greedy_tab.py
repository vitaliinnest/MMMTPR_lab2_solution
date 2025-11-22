import tkinter as tk
from tkinter import ttk, messagebox
import time
from database import DatabaseManager
from algorithms import greedy_solve
from .step_window import StepWindow


class GreedyTab:
    """Tab for running the greedy algorithm."""
    
    def __init__(self, parent: ttk.Frame, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets for the greedy tab."""
        top = ttk.Frame(self.parent)
        top.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(top, text="Крок step:").pack(side=tk.LEFT)
        self.entry_step = ttk.Entry(top, width=10)
        self.entry_step.insert(0, "0.01")
        self.entry_step.pack(side=tk.LEFT, padx=5)

        btn_run = ttk.Button(top, text="Запустити жадібний алгоритм", command=self._run_greedy)
        btn_run.pack(side=tk.LEFT, padx=5)

        btn_trace = ttk.Button(top, text="Покрокова демонстрація", command=self._show_trace_window)
        btn_trace.pack(side=tk.LEFT, padx=5)

        self.text_result = tk.Text(self.parent, width=100, height=25)
        self.text_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _run_greedy(self):
        """Run the greedy algorithm and display results."""
        try:
            step = float(self.entry_step.get().strip())
            if step <= 0 or step > 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Помилка", "Невірне значення кроку step")
            return
        
        models = self.db.load_models()
        constraints = self.db.load_constraints()
        self.text_result.delete("1.0", tk.END)
        
        try:
            start = time.perf_counter()
            solution = greedy_solve(models, constraints, step=step)
            elapsed = time.perf_counter() - start
        except RuntimeError as e:
            messagebox.showerror("Помилка", str(e))
            return
        
        self.db.save_solution(solution, algorithm_name="greedy")
        
        self.text_result.insert(tk.END, f"Час виконання жадібного алгоритму = {elapsed:.6f} с\n\n")
        for i, m in enumerate(models):
            self.text_result.insert(tk.END, f"{m.name}: x{i+1} = {solution.x[i]:.4f}\n")
        self.text_result.insert(tk.END, f"\nЦільова функція Z = {solution.objective:.6f}\n")
        
        p_final = sum(m.precision * solution.x[i] for i, m in enumerate(models))
        r_final = sum(m.recall * solution.x[i] for i, m in enumerate(models))
        t_final = sum(m.response_time * solution.x[i] for i, m in enumerate(models))
        
        self.text_result.insert(tk.END, f"Середня точність = {p_final:.4f}\n")
        self.text_result.insert(tk.END, f"Середня повнота = {r_final:.4f}\n")
        self.text_result.insert(tk.END, f"Середній час відповіді = {t_final:.4f}\n")

    def _show_trace_window(self):
        """Show step-by-step trace window."""
        try:
            step = float(self.entry_step.get().strip())
            if step <= 0 or step > 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Помилка", "Невірне значення кроку step")
            return
        
        models = self.db.load_models()
        constraints = self.db.load_constraints()
        StepWindow(self.parent, models, constraints, step)
