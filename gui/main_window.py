import tkinter as tk
from tkinter import ttk
from database import DatabaseManager
from .models_tab import ModelsTab
from .greedy_tab import GreedyTab
from .objective_tab import ObjectiveTab


class App(tk.Tk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.title("Жадібний алгоритм ЗЛП (ембеддинги)")
        self.geometry("950x650")
        
        self.db = DatabaseManager("lp_embeddings.db")
        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self):
        """Create the main notebook with tabs."""
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        frame_models = ttk.Frame(notebook)
        frame_greedy = ttk.Frame(notebook)
        frame_objective = ttk.Frame(notebook)

        notebook.add(frame_models, text="Моделі та обмеження")
        notebook.add(frame_greedy, text="Жадібний алгоритм")
        notebook.add(frame_objective, text="Цільова функція")

        # Create tab instances
        self.models_tab = ModelsTab(frame_models, self.db)
        self.greedy_tab = GreedyTab(frame_greedy, self.db)
        self.objective_tab = ObjectiveTab(frame_objective, self.db)

    def _on_close(self):
        """Clean up resources on window close."""
        self.db.close()
        self.destroy()
