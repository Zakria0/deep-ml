import numpy as np
import math

def jacks_car_rental(max_cars, max_move, rental_credit, move_cost,
                     request_lambda, return_lambda, gamma,
                     poisson_upper_bound=11, theta=0.01):
    """
    Solve a simplified car rental MDP using policy iteration.
    
    Args:
        max_cars: int, maximum cars at each location
        max_move: int, maximum cars moved per night
        rental_credit: float, revenue per rental
        move_cost: float, cost per car moved
        request_lambda: tuple (lambda1, lambda2) for Poisson request rates
        return_lambda: tuple (lambda1, lambda2) for Poisson return rates
        gamma: float, discount factor
        poisson_upper_bound: int, Poisson truncation (default 11)
        theta: float, policy evaluation convergence threshold (default 0.01)
    
    Returns:
        tuple: (policy, value, num_iterations)
            policy: list of lists of int, optimal cars to move from loc1 to loc2
            value: list of lists of float, rounded to 2 decimals
            num_iterations: int, number of policy improvement steps
    """
    def pmf(l, n):
        n = np.asarray(n)
        log_factorial = np.vectorize(math.lgamma)(n + 1)
        log_pmf = n * np.log(l) - l - log_factorial
        return np.exp(log_pmf)
    
    pmf_req1 = [pmf(request_lambda[0], n) for n in range(poisson_upper_bound)]
    pmf_req2 = [pmf(request_lambda[1], n) for n in range(poisson_upper_bound)]
    pmf_ret1 = [pmf(return_lambda[0], n) for n in range(poisson_upper_bound)]
    pmf_ret2 = [pmf(return_lambda[1], n) for n in range(poisson_upper_bound)]

    def transitions(i, j, a):
        c1, c2 = i - a, j + a
        out = {}
        for n1 in range(poisson_upper_bound):
            for n2 in range(poisson_upper_bound):
                r1, r2 = min(c1, n1), min(c2, n2)
                reward = rental_credit * (r1 + r2)
                for m1 in range(poisson_upper_bound):
                    for m2 in range(poisson_upper_bound):
                        p = pmf_req1[n1] * pmf_req2[n2] * pmf_ret1[m1] * pmf_ret2[m2]
                        nxt = (min(max_cars, c1 - r1 + m1), min(max_cars, c2 - r2 + m2))

                        out[(nxt, reward)] = out.get((nxt, reward), 0.) + p
        return [(p, nxt, reward) for (nxt, reward), p in out.items()]
        
    transitions = {(i, j): {a: transitions(i, j, a) for a in range(max(-max_move, -j, i - max_cars), min(max_move, i, max_cars - j)+1)} for i in range(max_cars+1) for j in range(max_cars+1)}

    value  = np.zeros((max_cars+1, max_cars+1))
    policy = np.zeros((max_cars+1, max_cars+1), dtype=np.int32)

    def policy_eval():
        nonlocal value
        while True:
            new_V = value.copy()
            delta = 0.0
            for (i, j), actions in transitions.items():
                q = 0.0
                for p, nxt, reward in actions[policy[i, j]]:
                    q += p * (reward + gamma * value[nxt])
                q -= move_cost * abs(policy[i, j]) 
                delta = max(delta, abs(q - value[i, j]))
                new_V[i, j] = q
            value = new_V
            if delta < theta:
                break

    num_iterations = 0
    while 1:
        policy_eval()
        policy_stable = True
        for state in transitions:
            old_action = policy[state]
            best_action_val = -np.inf
            for action, triples in transitions[state].items():
                q = 0
                for p, nxt, reward in triples:
                    q += p * (reward + gamma * value[nxt])
                q -= move_cost * abs(action)
                if best_action_val < q:
                    policy[state] = action
                    best_action_val = q
            if old_action != policy[state]:
                policy_stable = False
        num_iterations += 1
        if policy_stable:
            break
    
    return policy.tolist(), np.round(value, 2).tolist(), num_iterations





