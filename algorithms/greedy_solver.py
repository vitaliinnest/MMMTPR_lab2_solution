from typing import List, Tuple, Dict, Any
from models import EmbeddingModel, LPConstraints, LPSolution


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


def greedy_solve_with_trace(
    models: List[EmbeddingModel], 
    constraints: LPConstraints, 
    step: float = 0.01
) -> Tuple[LPSolution, List[Dict[str, Any]]]:
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
