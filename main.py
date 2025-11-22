import sqlite3
from dataclasses import dataclass
from typing import List
from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk, messagebox


@dataclass
class EmbeddingModel:
    id: int
    name: str
    var_code: str
    cost_per_query: float
    response_time: float
    precision: float
    recall: float


@dataclass
class LPConstraints:
    p_min: float
    r_min: float
    t_max: float


@dataclass
class LPSolution:
    x: List[float]
    objective: float


def get_default_models() -> List[EmbeddingModel]:
    return [
        EmbeddingModel(
            id=1,
            name="fastText",
            var_code="x1",
            cost_per_query=0.20,
            response_time=0.15,
            precision=0.72,
            recall=0.68,
        ),
        EmbeddingModel(
            id=2,
            name="SBERT",
            var_code="x2",
            cost_per_query=0.40,
            response_time=0.40,
            precision=0.82,
            recall=0.80,
        ),
        EmbeddingModel(
            id=3,
            name="Хмарна модель",
            var_code="x3",
            cost_per_query=0.80,
            response_time=0.70,
            precision=0.90,
            recall=0.88,
        ),
    ]


def get_default_constraints() -> LPConstraints:
    return LPConstraints(
        p_min=0.80,
        r_min=0.78,
        t_max=0.50,
    )


def create_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS embedding_models (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            var_code TEXT NOT NULL,
            cost_per_query REAL NOT NULL,
            response_time REAL NOT NULL,
            precision_value REAL NOT NULL,
            recall_value REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS constraints (
            id INTEGER PRIMARY KEY,
            p_min REAL NOT NULL,
            r_min REAL NOT NULL,
            t_max REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS solutions (
            id INTEGER PRIMARY KEY,
            x1 REAL NOT NULL,
            x2 REAL NOT NULL,
            x3 REAL NOT NULL,
            objective REAL NOT NULL,
            algorithm_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def save_models(conn: sqlite3.Connection, models: List[EmbeddingModel]) -> None:
    cur = conn.cursor()
    for m in models:
        cur.execute(
            """
            INSERT OR REPLACE INTO embedding_models
            (id, name, var_code, cost_per_query, response_time, precision_value, recall_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.id,
                m.name,
                m.var_code,
                m.cost_per_query,
                m.response_time,
                m.precision,
                m.recall,
            ),
        )
    conn.commit()


def save_constraints(conn: sqlite3.Connection, constraints: LPConstraints) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO constraints (id, p_min, r_min, t_max)
        VALUES (1, ?, ?, ?)
        """,
        (constraints.p_min, constraints.r_min, constraints.t_max),
    )
    conn.commit()


def load_models(conn: sqlite3.Connection) -> List[EmbeddingModel]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, name, var_code, cost_per_query, response_time, precision_value, recall_value
        FROM embedding_models
        ORDER BY id
        """
    )
    rows = cur.fetchall()
    models = []
    for row in rows:
        models.append(
            EmbeddingModel(
                id=row[0],
                name=row[1],
                var_code=row[2],
                cost_per_query=row[3],
                response_time=row[4],
                precision=row[5],
                recall=row[6],
            )
        )
    return models


def load_constraints(conn: sqlite3.Connection) -> LPConstraints:
    cur = conn.cursor()
    cur.execute("SELECT p_min, r_min, t_max FROM constraints WHERE id = 1")
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Constraints not found")
    return LPConstraints(p_min=row[0], r_min=row[1], t_max=row[2])


def greedy_solve(models: List[EmbeddingModel], constraints: LPConstraints, step: float = 0.01) -> LPSolution:
    n = len(models)
    x = [0.0] * n
    s = 0.0
    p_cur = 0.0
    r_cur = 0.0
    t_cur = 0.0
    max_p = max(m.precision for m in models)
    max_r = max(m.recall for m in models)
    min_t = min(m.response_time for m in models)
    eps = 1e-9
    while s < 1.0 - eps:
        candidates = []
        for j, m in enumerate(models):
            if s + step > 1.0 + eps:
                continue
            s_new = s + step
            p_new = p_cur + step * m.precision
            r_new = r_cur + step * m.recall
            t_new = t_cur + step * m.response_time
            remaining = 1.0 - s_new
            p_max_possible = p_new + remaining * max_p
            r_max_possible = r_new + remaining * max_r
            t_min_possible = t_new + remaining * min_t
            if p_max_possible + eps < constraints.p_min:
                continue
            if r_max_possible + eps < constraints.r_min:
                continue
            if t_min_possible - eps > constraints.t_max:
                continue
            candidates.append((m.cost_per_query, m.response_time, j, p_new, r_new, t_new, s_new))
        if not candidates:
            break
        candidates.sort(key=lambda v: (v[0], v[1]))
        _, _, j_best, p_cur, r_cur, t_cur, s = candidates[0]
        x[j_best] += step
    if abs(sum(x) - 1.0) > 1e-6:
        raise RuntimeError("No feasible solution found by greedy algorithm")
    p_final = sum(m.precision * x[i] for i, m in enumerate(models))
    r_final = sum(m.recall * x[i] for i, m in enumerate(models))
    t_final = sum(m.response_time * x[i] for i, m in enumerate(models))
    if p_final + eps < constraints.p_min or r_final + eps < constraints.r_min or t_final - eps > constraints.t_max:
        raise RuntimeError("Greedy solution does not satisfy constraints")
    objective = sum(m.cost_per_query * x[i] for i, m in enumerate(models))
    return LPSolution(x=x, objective=objective)


def greedy_solve_with_trace(models: List[EmbeddingModel], constraints: LPConstraints, step: float = 0.01):
    n = len(models)
    x = [0.0] * n
    s = 0.0
    p_cur = 0.0
    r_cur = 0.0
    t_cur = 0.0
    max_p = max(m.precision for m in models)
    max_r = max(m.recall for m in models)
    min_t = min(m.response_time for m in models)
    eps = 1e-9
    trace = []
    step_idx = 0
    while s < 1.0 - eps:
        candidates = []
        for j, m in enumerate(models):
            if s + step > 1.0 + eps:
                continue
            s_new = s + step
            p_new = p_cur + step * m.precision
            r_new = r_cur + step * m.recall
            t_new = t_cur + step * m.response_time
            remaining = 1.0 - s_new
            p_max_possible = p_new + remaining * max_p
            r_max_possible = r_new + remaining * max_r
            t_min_possible = t_new + remaining * min_t
            if p_max_possible + eps < constraints.p_min:
                continue
            if r_max_possible + eps < constraints.r_min:
                continue
            if t_min_possible - eps > constraints.t_max:
                continue
            candidates.append((m.cost_per_query, m.response_time, j, p_new, r_new, t_new, s_new))
        if not candidates:
            break
        candidates.sort(key=lambda v: (v[0], v[1]))
        _, _, j_best, p_cur, r_cur, t_cur, s = candidates[0]
        x[j_best] += step
        objective = sum(m.cost_per_query * x[i] for i, m in enumerate(models))
        trace.append(
            {
                "step": step_idx,
                "model_index": j_best,
                "model_name": models[j_best].name,
                "added_share": step,
                "x": x.copy(),
                "precision": p_cur,
                "recall": r_cur,
                "time": t_cur,
                "objective": objective,
            }
        )
        step_idx += 1
    if abs(sum(x) - 1.0) > 1e-6:
        raise RuntimeError("No feasible solution found by greedy algorithm")
    p_final = sum(m.precision * x[i] for i, m in enumerate(models))
    r_final = sum(m.recall * x[i] for i, m in enumerate(models))
    t_final = sum(m.response_time * x[i] for i, m in enumerate(models))
    if p_final + eps < constraints.p_min or r_final + eps < constraints.r_min or t_final - eps > constraints.t_max:
        raise RuntimeError("Greedy solution does not satisfy constraints")
    objective = sum(m.cost_per_query * x[i] for i, m in enumerate(models))
    solution = LPSolution(x=x, objective=objective)
    return solution, trace


def save_solution(conn: sqlite3.Connection, solution: LPSolution, algorithm_name: str = "greedy") -> None:
    cur = conn.cursor()
    x1 = solution.x[0] if len(solution.x) > 0 else 0.0
    x2 = solution.x[1] if len(solution.x) > 1 else 0.0
    x3 = solution.x[2] if len(solution.x) > 2 else 0.0
    cur.execute(
        """
        INSERT INTO solutions (x1, x2, x3, objective, algorithm_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (x1, x2, x3, solution.objective, algorithm_name, datetime.now().isoformat()),
    )
    conn.commit()


def ensure_initial_data(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM embedding_models")
    cnt_models = cur.fetchone()[0]
    if cnt_models == 0:
        save_models(conn, get_default_models())
    cur.execute("SELECT COUNT(*) FROM constraints")
    cnt_constraints = cur.fetchone()[0]
    if cnt_constraints == 0:
        save_constraints(conn, get_default_constraints())


class StepWindow(tk.Toplevel):
    def __init__(self, master, models, constraints, step):
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
        self.text = tk.Text(self, width=90, height=25)
        self.text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text.insert(tk.END, f"Час виконання жадібного алгоритму = {self.elapsed:.6f} с\n\n")
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.btn_next = tk.Button(btn_frame, text="Наступний крок", command=self.show_next)
        self.btn_next.pack(side=tk.LEFT, padx=5)
        self.btn_final = tk.Button(btn_frame, text="Підсумок", command=self.show_final)
        self.btn_final.pack(side=tk.LEFT, padx=5)
        self.show_next()

    def show_next(self):
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
        self.text.insert(tk.END, "\nПідсумковий розв'язок:\n")
        for i, m in enumerate(self.models):
            self.text.insert(tk.END, f"{m.name}: x{i+1} = {self.solution.x[i]:.2f}\n")
        self.text.insert(tk.END, f"Цільова функція Z = {self.solution.objective:.6f}\n")
        self.text.see(tk.END)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Жадібний алгоритм ЗЛП (ембеддинги)")
        self.geometry("950x650")
        self.conn = sqlite3.connect("lp_embeddings.db")
        create_tables(self.conn)
        ensure_initial_data(self.conn)
        self.models = load_models(self.conn)
        self.constraints = load_constraints(self.conn)
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        frame_models = ttk.Frame(notebook)
        frame_greedy = ttk.Frame(notebook)
        frame_objective = ttk.Frame(notebook)

        notebook.add(frame_models, text="Моделі та обмеження")
        notebook.add(frame_greedy, text="Жадібний алгоритм")
        notebook.add(frame_objective, text="Цільова функція")

        self.create_models_tab(frame_models)
        self.create_greedy_tab(frame_greedy)
        self.create_objective_tab(frame_objective)

    def create_models_tab(self, parent):
        upper = ttk.Frame(parent)
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

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        lower = ttk.Frame(parent)
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

        self.selected_model_id = None

        btn_update = ttk.Button(lower, text="Зберегти вибрану модель", command=self.save_selected_model)
        btn_reload = ttk.Button(lower, text="Оновити з БД", command=self.reload_models)
        btn_add = ttk.Button(lower, text="Додати нову модель", command=self.add_new_model)

        btn_update.grid(row=2, column=0, columnspan=2, pady=4)
        btn_reload.grid(row=2, column=2, columnspan=2, pady=4)
        btn_add.grid(row=2, column=4, columnspan=2, pady=4)

        sep = ttk.Separator(parent, orient="horizontal")
        sep.pack(fill=tk.X, padx=5, pady=5)

        constr_frame = ttk.Frame(parent)
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

        btn_save_constr = ttk.Button(constr_frame, text="Зберегти обмеження", command=self.save_constraints_gui)
        btn_save_constr.grid(row=1, column=0, columnspan=2, pady=4)

        btn_reload_constr = ttk.Button(constr_frame, text="Оновити обмеження", command=self.reload_constraints)
        btn_reload_constr.grid(row=1, column=2, columnspan=2, pady=4)

        self.refresh_tree()
        self.fill_constraints_entries()

    def create_greedy_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(top, text="Крок step:").pack(side=tk.LEFT)
        self.entry_step = ttk.Entry(top, width=10)
        self.entry_step.insert(0, "0.01")
        self.entry_step.pack(side=tk.LEFT, padx=5)

        btn_run = ttk.Button(top, text="Запустити жадібний алгоритм", command=self.run_greedy_gui)
        btn_run.pack(side=tk.LEFT, padx=5)

        btn_trace = ttk.Button(top, text="Покрокова демонстрація", command=self.show_trace_window)
        btn_trace.pack(side=tk.LEFT, padx=5)

        self.text_result = tk.Text(parent, width=100, height=25)
        self.text_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_objective_tab(self, parent):
        frm = ttk.Frame(parent)
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

        btn_calc = ttk.Button(frm, text="Обчислити", command=self.compute_objective_gui)
        btn_calc.grid(row=1, column=0, columnspan=2, pady=4)

        self.text_obj = tk.Text(parent, width=100, height=20)
        self.text_obj.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for m in self.models:
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

    def fill_constraints_entries(self):
        self.entry_pmin.delete(0, tk.END)
        self.entry_rmin.delete(0, tk.END)
        self.entry_tmax.delete(0, tk.END)
        self.entry_pmin.insert(0, str(self.constraints.p_min))
        self.entry_rmin.insert(0, str(self.constraints.r_min))
        self.entry_tmax.insert(0, str(self.constraints.t_max))

    def on_tree_select(self, event):
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

    def save_selected_model(self):
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
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE embedding_models
            SET name = ?, var_code = ?, cost_per_query = ?, response_time = ?, precision_value = ?, recall_value = ?
            WHERE id = ?
            """,
            (name, var_code, cost, time_val, precision, recall, self.selected_model_id),
        )
        self.conn.commit()
        self.reload_models()
        messagebox.showinfo("Готово", "Модель оновлено")

    def add_new_model(self):
        name = self.entry_name.get().strip()
        var_code = self.entry_var.get().strip()
        if not name:
            messagebox.showerror("Помилка", "Вкажіть назву моделі")
            return
        self.models = load_models(self.conn)
        if not var_code:
            existing_vars = {m.var_code for m in self.models}
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
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO embedding_models
            (name, var_code, cost_per_query, response_time, precision_value, recall_value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, var_code, cost, time_val, precision, recall),
        )
        self.conn.commit()
        self.reload_models()
        messagebox.showinfo("Готово", f"Додано нову модель з змінною {var_code}")

    def reload_models(self):
        self.models = load_models(self.conn)
        self.refresh_tree()

    def save_constraints_gui(self):
        try:
            pmin = float(self.entry_pmin.get().strip())
            rmin = float(self.entry_rmin.get().strip())
            tmax = float(self.entry_tmax.get().strip())
        except ValueError:
            messagebox.showerror("Помилка", "Невірні числові значення")
            return
        self.constraints = LPConstraints(p_min=pmin, r_min=rmin, t_max=tmax)
        save_constraints(self.conn, self.constraints)
        messagebox.showinfo("Готово", "Обмеження оновлено")

    def reload_constraints(self):
        self.constraints = load_constraints(self.conn)
        self.fill_constraints_entries()

    def run_greedy_gui(self):
        try:
            step = float(self.entry_step.get().strip())
            if step <= 0 or step > 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Помилка", "Невірне значення кроку step")
            return
        self.models = load_models(self.conn)
        self.constraints = load_constraints(self.conn)
        self.text_result.delete("1.0", tk.END)
        try:
            start = time.perf_counter()
            solution = greedy_solve(self.models, self.constraints, step=step)
            elapsed = time.perf_counter() - start
        except RuntimeError as e:
            messagebox.showerror("Помилка", str(e))
            return
        save_solution(self.conn, solution, algorithm_name="greedy")
        self.text_result.insert(tk.END, f"Час виконання жадібного алгоритму = {elapsed:.6f} с\n\n")
        for i, m in enumerate(self.models):
            self.text_result.insert(tk.END, f"{m.name}: x{i+1} = {solution.x[i]:.4f}\n")
        self.text_result.insert(tk.END, f"\nЦільова функція Z = {solution.objective:.6f}\n")
        p_final = sum(m.precision * solution.x[i] for i, m in enumerate(self.models))
        r_final = sum(m.recall * solution.x[i] for i, m in enumerate(self.models))
        t_final = sum(m.response_time * solution.x[i] for i, m in enumerate(self.models))
        self.text_result.insert(tk.END, f"Середня точність = {p_final:.4f}\n")
        self.text_result.insert(tk.END, f"Середня повнота = {r_final:.4f}\n")
        self.text_result.insert(tk.END, f"Середній час відповіді = {t_final:.4f}\n")

    def show_trace_window(self):
        try:
            step = float(self.entry_step.get().strip())
            if step <= 0 or step > 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Помилка", "Невірне значення кроку step")
            return
        self.models = load_models(self.conn)
        self.constraints = load_constraints(self.conn)
        StepWindow(self, self.models, self.constraints, step)

    def compute_objective_gui(self):
        self.models = load_models(self.conn)
        if len(self.models) < 3:
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
        objective = sum(self.models[i].cost_per_query * x[i] for i in range(3))
        p_val = sum(self.models[i].precision * x[i] for i in range(3))
        r_val = sum(self.models[i].recall * x[i] for i in range(3))
        t_val = sum(self.models[i].response_time * x[i] for i in range(3))
        self.text_obj.delete("1.0", tk.END)
        self.text_obj.insert(tk.END, f"Сума часток x1+x2+x3 = {s:.4f}\n")
        self.text_obj.insert(tk.END, f"Цільова функція Z = {objective:.6f}\n")
        self.text_obj.insert(tk.END, f"Середня точність = {p_val:.4f}\n")
        self.text_obj.insert(tk.END, f"Середня повнота = {r_val:.4f}\n")
        self.text_obj.insert(tk.END, f"Середній час відповіді = {t_val:.4f}\n")

    def on_close(self):
        try:
            self.conn.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
