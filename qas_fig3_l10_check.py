#!/usr/bin/env python3
"""Reproducao do fluxo QAS do autor (caso fechado) para checagem de fidelidade.

Foco:
- modelo do script do autor (model=8, ising ladder)
- closed system
- compara QAS vs exato
- imprime fidelidade para K escolhido (padrao K=3)
"""

from __future__ import annotations

import argparse
import operator
from functools import reduce
from pathlib import Path

import numpy as np
import qutip as qt
import scipy.linalg
from qutip_qip.operations import csign, rx, ry, rz


def prod(factors):
    return reduce(operator.mul, factors, 1)


def gen_fock_op(op, position, size, levels=2):
    op_list = [qt.qeye(levels) for _ in range(size)]
    op_list[position] = op
    return qt.tensor(op_list)


def multiply_paulis(curr_paulis, to_mult_paulis):
    new_paulis = []
    for i in range(len(curr_paulis)):
        for j in range(len(to_mult_paulis)):
            add_pauli = np.zeros(len(curr_paulis[i]), dtype=int)
            for k in range(len(curr_paulis[i])):
                if curr_paulis[i][k] == 1 and to_mult_paulis[j][k] == 2:
                    add_pauli[k] = 3
                elif curr_paulis[i][k] == 2 and to_mult_paulis[j][k] == 1:
                    add_pauli[k] = 3
                else:
                    add_pauli[k] = abs(curr_paulis[i][k] - to_mult_paulis[j][k])
            new_paulis.append(add_pauli)
    new_paulis = list(np.unique(new_paulis, axis=0))
    return new_paulis


def get_hamiltonian_string_model8(L, J, h):
    h_strings, h_values = [], []
    l_rung = L // 2

    if J != 0:
        for i in range(l_rung - 1):
            s = np.zeros(L, dtype=int)
            s[i] = 3
            s[i + 1] = 3
            h_strings.append(list(s))
            h_values.append(0.25 * J)
        for i in range(l_rung - 1):
            s = np.zeros(L, dtype=int)
            s[i + l_rung] = 3
            s[i + 1 + l_rung] = 3
            h_strings.append(list(s))
            h_values.append(0.25 * J)
        for i in range(l_rung):
            s = np.zeros(L, dtype=int)
            s[i] = 3
            s[i + l_rung] = 3
            h_strings.append(list(s))
            h_values.append(0.25 * J)

    if h != 0:
        for i in range(L):
            s = np.zeros(L, dtype=int)
            s[i] = 1
            h_strings.append(list(s))
            h_values.append(h)

    return h_strings, h_values


def get_pauli_op(pauli_string, op_id, op_x, op_y, op_z):
    circuit = op_id
    for i, p in enumerate(pauli_string):
        if p == 1:
            circuit = circuit * op_x[i]
        elif p == 2:
            circuit = circuit * op_y[i]
        elif p == 3:
            circuit = circuit * op_z[i]
    return circuit


def get_ini_state(ini_type, n_qubits, depth, seed, op_csign):
    rng = np.random.default_rng(seed)
    if ini_type == 1:
        st = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
    elif ini_type == 2:
        rand_angles = rng.random((depth, n_qubits)) * 2 * np.pi
        rand_pauli = rng.integers(1, 4, size=(depth, n_qubits))
        st = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
        st = qt.tensor([ry(np.pi / 4) for _ in range(n_qubits)]) * st
        ent_layer = prod([op_csign[j] for j in range(n_qubits)])
        for d in range(depth):
            rots = []
            for q in range(n_qubits):
                angle = rand_angles[d][q]
                p = rand_pauli[d][q]
                if p == 1:
                    rots.append(rx(angle))
                elif p == 2:
                    rots.append(ry(angle))
                else:
                    rots.append(rz(angle))
            st = qt.tensor(rots) * st
            st = ent_layer * st
    else:
        # ini_type 0 do autor (|+...+>)
        st = qt.tensor([qt.basis(2, 1) + qt.basis(2, 0) for _ in range(n_qubits)])
    st = st / st.norm()
    return st


def run_qas_check(n_qubits=10, k_moment=3, seed=1, tfinal=6.0, n_timesteps=101, inv_cond=1e-10):
    J, h = 1.0, 1.0
    depth = n_qubits

    times = np.linspace(0.0, tfinal, num=n_timesteps)

    op_z = [gen_fock_op(qt.sigmaz(), i, n_qubits) for i in range(n_qubits)]
    op_x = [gen_fock_op(qt.sigmax(), i, n_qubits) for i in range(n_qubits)]
    op_y = [gen_fock_op(qt.sigmay(), i, n_qubits) for i in range(n_qubits)]
    op_id = gen_fock_op(qt.qeye(2), 0, n_qubits)
    op_csign = [csign(n_qubits, i, (i + 1) % n_qubits) for i in range(n_qubits)]

    h_strings, h_values = get_hamiltonian_string_model8(n_qubits, J, h)
    H = 0
    for i in range(len(h_values)):
        H = H + h_values[i] * get_pauli_op(h_strings[i], op_id, op_x, op_y, op_z)

    # Config do autor para Fig. 3: ini_type=2, ini_evolve_type=1
    initial_state = get_ini_state(2, n_qubits, depth, seed, op_csign)
    initial_evolve_state = get_ini_state(1, n_qubits, depth, seed + 123, op_csign)

    exact_states = qt.mesolve(H, initial_evolve_state, times, [], []).states

    # K-moment expansion
    base_expand_strings = [np.zeros(n_qubits, dtype=int)]
    for _ in range(k_moment):
        base_expand_strings += list(multiply_paulis(base_expand_strings, h_strings))
        new_strings, string_index = list(np.unique(base_expand_strings, axis=0, return_index=True))
        sorted_index = np.sort(string_index)
        base_expand_strings = [base_expand_strings[k] for k in sorted_index]
        _ = new_strings

    expand_states = [get_pauli_op(s, op_id, op_x, op_y, op_z) * initial_state for s in base_expand_strings]
    n_expand = len(expand_states)

    E = np.zeros((n_expand, n_expand), dtype=np.complex128)
    D = np.zeros((n_expand, n_expand), dtype=np.complex128)
    for m in range(n_expand):
        for n in range(n_expand):
            E[m, n] = expand_states[m].overlap(expand_states[n])
            D[m, n] = expand_states[m].overlap(H * expand_states[n])

    # IQAE step para estado inicial da evolucao
    ini_projector = -initial_evolve_state * initial_evolve_state.dag()
    ini_matrix = np.zeros((n_expand, n_expand), dtype=np.complex128)
    for m in range(n_expand):
        for n in range(m, n_expand):
            ini_matrix[m, n] = expand_states[m].overlap(ini_projector * expand_states[n])
    for m in range(n_expand):
        for n in range(m + 1, n_expand):
            ini_matrix[n, m] = np.conjugate(ini_matrix[m, n])

    e_vals, e_vecs = scipy.linalg.eigh(E)
    e_vals_inverted = np.array(e_vals)
    e_vals_cond = np.array(e_vals)
    for k in range(len(e_vals)):
        if e_vals_inverted[k] < inv_cond:
            e_vals_inverted[k] = 0.0
            e_vals_cond[k] = 0.0
        else:
            e_vals_inverted[k] = 1.0 / e_vals_inverted[k]

    s_inv = np.dot(np.diag(np.sqrt(e_vals_inverted)), np.transpose(np.conjugate(e_vecs)))
    toeig = np.dot(s_inv, np.dot(ini_matrix, np.transpose(np.conjugate(s_inv))))
    qae_energy, qae_vectors = scipy.linalg.eigh(toeig)
    _ = qae_energy
    ini_alpha = qae_vectors[:, 0]
    ini_alpha = np.dot(np.transpose(np.conjugate(s_inv)), ini_alpha)
    norm_ini_alpha = np.sqrt(np.abs(np.dot(np.transpose(np.conjugate(ini_alpha)), np.dot(E, ini_alpha))))
    ini_alpha = ini_alpha / norm_ini_alpha

    # Closed-system QAS evolution
    h_toeig = np.dot(s_inv, np.dot(D, np.transpose(np.conjugate(s_inv))))
    h_energy, h_vectors = scipy.linalg.eigh(h_toeig)
    alpha_eig_vecs = [np.dot(np.transpose(np.conjugate(s_inv)), h_vectors[:, i]) for i in range(n_expand)]
    coeffs = np.array(
        [np.dot(np.transpose(np.conjugate(alpha_eig_vecs[i])), np.dot(E, ini_alpha)) for i in range(n_expand)]
    )

    alpha_time = [
        sum([coeffs[i] * alpha_eig_vecs[i] * np.exp(-1j * h_energy[i] * t) for i in range(n_expand)])
        for t in times
    ]
    evolved_qas = [sum([expand_states[i] * alpha_time[j][i] for i in range(n_expand)]) for j in range(len(times))]
    fidelity = np.array([np.abs(evolved_qas[j].overlap(exact_states[j])) ** 2 for j in range(len(times))], dtype=float)

    return {
        "n_expand_states": n_expand,
        "fidelity": fidelity,
        "fidelity_min": float(np.min(fidelity)),
        "fidelity_mean": float(np.mean(fidelity)),
        "fidelity_final": float(fidelity[-1]),
        "times": times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tfinal", type=float, default=6.0)
    parser.add_argument("--nsteps", type=int, default=101)
    parser.add_argument("--save-npz", type=str, default="")
    args = parser.parse_args()

    res = run_qas_check(
        n_qubits=args.L,
        k_moment=args.K,
        seed=args.seed,
        tfinal=args.tfinal,
        n_timesteps=args.nsteps,
    )

    print(f"L={args.L}, K={args.K}")
    print(f"n_expand_states={res['n_expand_states']}")
    print(f"fidelity_min={res['fidelity_min']:.8f}")
    print(f"fidelity_mean={res['fidelity_mean']:.8f}")
    print(f"fidelity_final={res['fidelity_final']:.8f}")

    if args.save_npz:
        out = Path(args.save_npz).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            times=np.asarray(res["times"], dtype=float),
            fidelity=np.asarray(res["fidelity"], dtype=float),
            n_expand_states=int(res["n_expand_states"]),
            L=int(args.L),
            K=int(args.K),
            seed=int(args.seed),
        )
        print(f"saved={out}")


if __name__ == "__main__":
    main()
