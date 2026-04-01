# qas_statevec_4.py

import numpy as np
from itertools import product, combinations
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
import matplotlib.pyplot as plt


def kjall_disordered_ising_pauli_terms(
    L: int,
    J: float = 1.0,
    deltaJ: float = 0.0,
    J2: float = 0.3,
    h: float = 0.6,
    J_bonds: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    if J_bonds is None:
        dJ = rng.uniform(-deltaJ, deltaJ, size=L - 1)
        J_bonds = J + dJ
    else:
        J_bonds = np.asarray(J_bonds)
        assert J_bonds.shape == (L - 1,)

    terms = []

    for i in range(L - 1):
        label = ["I"] * L
        label[i] = "Z"
        label[i + 1] = "Z"
        coef = -J_bonds[i]
        terms.append((coef, "".join(label)))

    for i in range(L - 2):
        label = ["I"] * L
        label[i] = "Z"
        label[i + 2] = "Z"
        coef = -J2
        terms.append((coef, "".join(label)))

    for i in range(L):
        label = ["I"] * L
        label[i] = "X"
        coef = -h
        terms.append((coef, "".join(label)))

    return terms, J_bonds


# ----------------------------------------------------------------------
#  Conjunto CS_K gerado só com os termos X_i (K-moment)
# ----------------------------------------------------------------------
def x_moment_words_from_terms(terms, L: int, K: int):

    x_terms_site_and_index = []
    for idx, (coef, label) in enumerate(terms):
        sites_with_X = [q for q, p in enumerate(label) if p == "X"]
        sites_with_Z = [q for q, p in enumerate(label) if p == "Z"]
        sites_with_Y = [q for q, p in enumerate(label) if p == "Y"]
        if len(sites_with_X) == 1 and len(sites_with_Z) == 0 and len(sites_with_Y) == 0:
            site = sites_with_X[0]
            x_terms_site_and_index.append((site, idx))

    x_terms_site_and_index.sort(key=lambda x: x[0])

    assert len(x_terms_site_and_index) == L, (
        f"Esperava {L} termos X_i, encontrei {len(x_terms_site_and_index)}."
    )

    x_indices_ordered = [idx for (site, idx) in x_terms_site_and_index]

    max_flips = min(K, L)
    words = [()] 

    for hw in range(1, max_flips + 1):
        for subset_sites in combinations(range(L), hw):
            word = tuple(x_indices_ordered[site] for site in subset_sites)
            words.append(word)

    return words

def k_moment_words(num_terms: int, K: int):
    words = [()]
    for L in range(1, K + 1):
        for tup in product(range(num_terms), repeat=L):
            words.append(tup)
    return words


# ----------------------------------------------------------------------
#  Construção dos estados da base QAS via circuitos + statevector
# ----------------------------------------------------------------------
def apply_pauli_string(qc: QuantumCircuit, label: str):
    for q, p in enumerate(label):
        if p == "X":
            qc.x(q)
        elif p == "Y":
            qc.y(q)
        elif p == "Z":
            qc.z(q)
    return qc


def build_basis_state_circuit(num_qubits: int, terms, word) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    # estado inicial |0...0>
    for idx in word:
        _, label = terms[idx]
        apply_pauli_string(qc, label)
    return qc


def compute_basis_statevectors(num_qubits: int, terms, words):
    states = []
    for w in words:
        qc = build_basis_state_circuit(num_qubits, terms, w)
        sv = Statevector.from_instruction(qc)
        states.append(sv)
    return states


# ----------------------------------------------------------------------
#  Matrizes E e D no subespaço
# ----------------------------------------------------------------------
def compute_E_D(states, terms):
    m = len(states)

    E = np.zeros((m, m), dtype=complex)
    D = np.zeros((m, m), dtype=complex)

    for i in range(m):
        for j in range(m):
            E[i, j] = np.vdot(states[i].data, states[j].data)

    pauli_mats = []
    coeffs = []
    for coeff, label in terms:
        op = SparsePauliOp.from_list([(label, 1.0)])
        pauli_mats.append(op.to_matrix())
        coeffs.append(coeff)
    coeffs = np.array(coeffs, dtype=complex)

    for i in range(m):
        psi_i = states[i].data
        for j in range(m):
            psi_j = states[j].data
            acc = 0.0 + 0.0j
            for k, Pk in enumerate(pauli_mats):
                acc += coeffs[k] * np.vdot(psi_i, Pk @ psi_j)
            D[i, j] = acc

    return E, D


def qas_overlaps_statevector_kjall(
    L: int,
    K: int = 1,
    J: float = 1.0,
    deltaJ: float = 0.0,
    J2: float = 0.3,
    h: float = 0.6,
    J_bonds: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
):
    terms, J_bonds = kjall_disordered_ising_pauli_terms(
        L=L, J=J, deltaJ=deltaJ, J2=J2, h=h, J_bonds=J_bonds, rng=rng
    )

    words = x_moment_words_from_terms(terms, L=L, K=K)
    states = compute_basis_statevectors(L, terms, words)
    E, D = compute_E_D(states, terms)

    return E, D, words, states, J_bonds


# ----------------------------------------------------------------------
#  Operadores de magnetização
# ----------------------------------------------------------------------
def build_local_Z_operators(states):
    L = states[0].num_qubits
    m = len(states)
    Z_ops = []

    for site in range(L):
        label = "I" * site + "Z" + "I" * (L - site - 1)
        P = SparsePauliOp.from_list([(label, 1.0)]).to_matrix()

        O = np.zeros((m, m), dtype=complex)
        for i in range(m):
            psi_i = states[i].data
            for j in range(m):
                psi_j = states[j].data
                O[i, j] = np.vdot(psi_i, P @ psi_j)

        Z_ops.append(O)

    return Z_ops


def magnetization_matrix(L: int):
    mats = []
    for site in range(L):
        label = "I" * site + "Z" + "I" * (L - site - 1)
        mats.append(SparsePauliOp.from_list([(label, 1.0)]).to_matrix())
    return np.sum(mats, axis=0)


# ----------------------------------------------------------------------
#  Dinâmica analítica no subespaço QAS
# ----------------------------------------------------------------------
def orthonormalize_E(E, eps: float = 1e-10):
    evals, U = np.linalg.eigh(E)
    evals_clipped = np.clip(evals, eps, None)
    S = U @ np.diag(1.0 / np.sqrt(evals_clipped))
    E_tilde = S.conj().T @ E @ S
    return S, E_tilde, evals


def precompute_qas_generator(E, D, eps: float = 1e-10):
    S, E_tilde, evals_E = orthonormalize_E(E, eps=eps)
    D_tilde = S.conj().T @ D @ S
    evals_D, V = np.linalg.eigh(D_tilde)
    return S, D_tilde, evals_D, V


def qas_time_evolution(E, D, t_list):
    t_list = np.asarray(t_list, dtype=float)
    m = E.shape[0]

    S, D_tilde, evals_D, V = precompute_qas_generator(E, D)

    alpha0 = np.zeros(m, dtype=complex)
    alpha0[0] = 1.0

    beta0 = np.linalg.solve(S, alpha0)
    c0 = V.conj().T @ beta0

    alphas = np.zeros((len(t_list), m), dtype=complex)
    for idx, t in enumerate(t_list):
        phases = np.exp(-1j * evals_D * t)
        beta_t = V @ (phases * c0)
        alpha_t = S @ beta_t
        alphas[idx, :] = alpha_t

    return alphas


def qas_local_magnetization_vs_time(E, D, states, t_list):
    t_list = np.asarray(t_list, dtype=float)
    L = states[0].num_qubits

    Z_ops = build_local_Z_operators(states)
    alphas = qas_time_evolution(E, D, t_list)
    ntimes, m = alphas.shape

    mags = np.zeros((ntimes, L), dtype=float)
    for ti in range(ntimes):
        alpha_t = alphas[ti, :]
        alpha_t_dag = np.conjugate(alpha_t)
        for j in range(L):
            O_j = Z_ops[j]
            mags[ti, j] = np.real(alpha_t_dag @ (O_j @ alpha_t))
    return mags


# ----------------------------------------------------------------------
#  Média em desordem (QAS e exato)
# ----------------------------------------------------------------------
def qas_disorder_average(
    L: int,
    K: int,
    n_realizations: int,
    t_list,
    J: float = 1.0,
    deltaJ: float = 0.0,
    J2: float = 0.3,
    h: float = 0.6,
    seed_base: int = 1234,
):

    t_list = np.asarray(t_list, float)
    ntimes = len(t_list)

    mags_accum = None
    J_bonds_all = []
    Mz_final_all = []

    for r in range(n_realizations):
        rng = np.random.default_rng(seed_base + r)

        E, D, words, states, J_bonds = qas_overlaps_statevector_kjall(
            L=L, K=K, J=J, deltaJ=deltaJ, J2=J2, h=h, rng=rng
        )

        mags = qas_local_magnetization_vs_time(E, D, states, t_list)
        Mz_vs_t_r = mags.sum(axis=1)  # (ntimes,)

        Mz_final_r = float(np.real(Mz_vs_t_r[-1]))
        Mz_final_all.append(Mz_final_r)

        if mags_accum is None:
            mags_accum = mags
        else:
            mags_accum += mags

        J_bonds_all.append(J_bonds)

    mags_mean = mags_accum / n_realizations
    J_bonds_all = np.array(J_bonds_all)
    Mz_final_all = np.array(Mz_final_all, dtype=float)

    return mags_mean, J_bonds_all, Mz_final_all


def exact_Mz_final_disorder_average(
    L: int,
    t_final: float,
    J: float,
    J2: float,
    h: float,
    deltaJ: float,
    J_bonds_all: np.ndarray,
):

    M_full = magnetization_matrix(L)
    Mz_vals = []

    for J_bonds in J_bonds_all:
        terms_ex, _ = kjall_disordered_ising_pauli_terms(
            L=L,
            J=J,
            deltaJ=0.0,
            J2=J2,
            h=h,
            J_bonds=J_bonds,
            rng=None,
        )

        pauli_list = [(label, coeff) for (coeff, label) in terms_ex]
        H = SparsePauliOp.from_list(pauli_list, num_qubits=L).to_matrix()

        w, V = np.linalg.eigh(H)

        psi0 = Statevector.from_label("0" * L).data
        c0 = V.conj().T @ psi0

        phases = np.exp(-1j * w * t_final)
        psi_t = V @ (phases * c0)

        Mz = np.real(np.vdot(psi_t, M_full @ psi_t))
        Mz_vals.append(Mz)

    Mz_vals = np.array(Mz_vals, dtype=float)
    Mz_exact_mean = float(Mz_vals.mean())
    Mz_exact_abs_mean = float(np.mean(np.abs(Mz_vals)))
    return Mz_exact_mean, Mz_exact_abs_mean, Mz_vals

if __name__ == "__main__":
    J = 1.0
    J2 = 0.3
    h = 0.6      

    t_max = 400.0
    ntimes = 201
    t_list = np.linspace(0.0, t_max, ntimes)
    t_final = t_list[-1]

    deltaJ_min = 0.0
    deltaJ_max = 12.0
    deltaJ_step = 0.1
    deltaJ_values = np.round(
        np.arange(deltaJ_min, deltaJ_max + 0.5 * deltaJ_step, deltaJ_step), 2
    )

    n_realizations = 500  

    for L in [4]:
        K = L   
        print(f"=== L = {L}, K = {K} ===")

        Mz_qas_vs_dJ   = []
        Mz_exact_vs_dJ = []

        for idx_dJ, deltaJ in enumerate(deltaJ_values):
            print(f"  deltaJ = {deltaJ:.2f}  ({idx_dJ+1}/{len(deltaJ_values)})")

            (
                mags_mean,
                J_bonds_all,
                Mz_final_all,      
            ) = qas_disorder_average(
                L=L,
                K=K,
                n_realizations=n_realizations,
                t_list=t_list,
                J=J,
                deltaJ=deltaJ,
                J2=J2,
                h=h,
                seed_base=1234 + 10000 * idx_dJ,
            )

            Mz_qas_mean_final = float(np.mean(Mz_final_all))
            Mz_qas_vs_dJ.append(Mz_qas_mean_final)

            Mz_exact_mean, Mz_exact_abs_mean, Mz_vals = exact_Mz_final_disorder_average(
                L=L,
                t_final=t_final,
                J=J,
                J2=J2,
                h=h,
                deltaJ=deltaJ,
                J_bonds_all=J_bonds_all,
            )
            Mz_exact_vs_dJ.append(Mz_exact_mean)

        Mz_qas_vs_dJ   = np.array(Mz_qas_vs_dJ)
        Mz_exact_vs_dJ = np.array(Mz_exact_vs_dJ)

        np.savez(
            f"qas_exact_kjall_statevec_L{L}_K{K}_scanDeltaJ_N{n_realizations}.npz",
            deltaJ_values=deltaJ_values,
            Mz_qas_timeavg=Mz_qas_vs_dJ,     
            Mz_exact_timeavg=Mz_exact_vs_dJ,  
        )




