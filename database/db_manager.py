import sqlite3
from typing import List
from datetime import datetime
from models import EmbeddingModel, LPConstraints, LPSolution
from .default_data import get_default_models, get_default_constraints


class DatabaseManager:
    def __init__(self, db_path: str = "lp_embeddings.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self._ensure_initial_data()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
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
        self.conn.commit()

    def _ensure_initial_data(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM embedding_models")
        cnt_models = cur.fetchone()[0]
        if cnt_models == 0:
            self.save_models(get_default_models())
        cur.execute("SELECT COUNT(*) FROM constraints")
        cnt_constraints = cur.fetchone()[0]
        if cnt_constraints == 0:
            self.save_constraints(get_default_constraints())

    def save_models(self, models: List[EmbeddingModel]) -> None:
        cur = self.conn.cursor()
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
        self.conn.commit()

    def save_constraints(self, constraints: LPConstraints) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO constraints (id, p_min, r_min, t_max)
            VALUES (1, ?, ?, ?)
            """,
            (constraints.p_min, constraints.r_min, constraints.t_max),
        )
        self.conn.commit()

    def load_models(self) -> List[EmbeddingModel]:
        cur = self.conn.cursor()
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

    def load_constraints(self) -> LPConstraints:
        cur = self.conn.cursor()
        cur.execute("SELECT p_min, r_min, t_max FROM constraints WHERE id = 1")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Constraints not found")
        return LPConstraints(p_min=row[0], r_min=row[1], t_max=row[2])

    def delete_model(self, model_id: int) -> None:
        """Delete a model from the database."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM embedding_models WHERE id = ?", (model_id,))
        self.conn.commit()

    def save_solution(self, solution: LPSolution, algorithm_name: str = "greedy") -> None:
        cur = self.conn.cursor()
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
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
