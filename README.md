# Greedy Algorithm for Linear Programming Problem

This application solves a linear programming problem for embedding model selection using a greedy algorithm.

## Project Structure

```
lab2_solution/
├── main.py                 # Entry point
├── models/                 # Data models
│   ├── __init__.py
│   └── data_models.py     # EmbeddingModel, LPConstraints, LPSolution
├── database/              # Database operations
│   ├── __init__.py
│   ├── db_manager.py      # DatabaseManager class
│   └── default_data.py    # Default models and constraints
├── algorithms/            # Algorithm implementations
│   ├── __init__.py
│   └── greedy_solver.py   # Greedy algorithm
└── gui/                   # GUI components
    ├── __init__.py
    ├── main_window.py     # Main application window
    ├── models_tab.py      # Models and constraints tab
    ├── greedy_tab.py      # Greedy algorithm tab
    ├── objective_tab.py   # Objective function tab
    └── step_window.py     # Step-by-step trace window
```

## Running the Application

```bash
python main.py
```

## Features

- Manage embedding models (add, edit, view)
- Configure constraints (precision, recall, time)
- Run greedy algorithm to optimize model selection
- View step-by-step algorithm execution
- Calculate objective function for custom inputs
