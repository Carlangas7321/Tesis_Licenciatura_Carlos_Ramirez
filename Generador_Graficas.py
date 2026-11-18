#!/usr/bin/env python3
"""
WP2 (cierre) y WP3 (LQR) -- Convertidor boost ideal.
Genera los numeros, tablas y figuras del Entregable 5.0.
Uso (desde la raiz del repositorio):
    python3 codigo/wp3_lqr.py > salidas/salida_wp3.txt
Salidas: figuras/*.pdf, datos/resultados_wp3.json
"""
import json, os
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=120)
os.makedirs("figuras", exist_ok=True); os.makedirs("datos", exist_ok=True)
RES = {}
def H(t): print("\n" + "=" * 8, t, "=" * 8)

# ------------------------------------------------------------------ parametros
Vs, vC_s, L, C, R = 24.0, 48.0, 1.0e-3, 470e-6, 20.0
Ts = 100e-6
D_MIN, D_MAX = 0.05, 0.95
dS = 1 - Vs / vC_s
iS = vC_s / (R * (1 - dS))
xS = np.array([iS, vC_s])
IMAX, VMAX, DMAX = 1.0, 2.0, 0.05          # escalas de diseno (Bryson), provisionales
Dx = np.diag([IMAX, VMAX]); Du = DMAX

def linmodel(d, vC, iL):
    Ac = np.array([[0, -(1 - d) / L], [(1 - d) / C, -1 / (R * C)]])
    Bc = np.array([[vC / L], [-iL / C]])
    Ec = np.array([[0.0], [-1 / C]])
    return Ac, Bc, Ec

def zoh(Ac, Bc, Ec, Ts):
    M = np.zeros((4, 4)); M[:2, :2] = Ac; M[:2, 2:3] = Bc; M[:2, 3:4] = Ec
    E = expm(M * Ts)
    return E[:2, :2], E[:2, 2:3], E[:2, 3:4]

def f(x, d, ip=0.0):
    iL, vC = x
    return np.array([(Vs - (1 - d) * vC) / L, ((1 - d) * iL - vC / R - ip) / C])

def step_nl(x, d, Ts=Ts, n=20, ip=0.0):
    h = Ts / n
    for _ in range(n):
        k1 = f(x, d, ip); k2 = f(x + h / 2 * k1, d, ip)
        k3 = f(x + h / 2 * k2, d, ip); k4 = f(x + h * k3, d, ip)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

# ---------------------------------------------------------- 0. modelo y trazabilidad con WP2
H("0. Modelo discreto (re-derivado) y comparacion con Entregable 4.0")
Ac, Bc, Ec = linmodel(dS, vC_s, iS)
Ad, Bd, Ed = zoh(Ac, Bc, Ec, Ts)
Ad_w2 = np.array([[0.997351, -0.049691], [0.105725, 0.986778]])
Bd_w2 = np.array([[4.821187], [-0.760660]]); Ed_w2 = np.array([[0.005298], [-0.211451]])
print("d*, i*_L =", dS, iS)
print("Ac=\n", Ac, "\nBc=", Bc.ravel(), "\nEc=", Ec.ravel())
print("Ad=\n", Ad, "\nBd=", Bd.ravel(), "\nEd=", Ed.ravel())
dif = max(abs(Ad - Ad_w2).max(), abs(Bd - Bd_w2).max(), abs(Ed - Ed_w2).max())
print("max |dif| vs valores publicados en WP2 (6 decimales):", dif)
lam = np.linalg.eigvals(Ad); print("eig(Ad)=", lam, " |.|=", abs(lam))
RES["Ad"] = Ad.tolist(); RES["Bd"] = Bd.ravel().tolist(); RES["Ed"] = Ed.ravel().tolist()
RES["dif_WP2"] = float(dif)

# ------------------------------------------- 1. controlabilidad, observabilidad y escalamiento
H("1. Controlabilidad / observabilidad / escalamiento")
Ctrl = np.hstack([Bd, Ad @ Bd]); print("cond(Ctrl) fisica =", np.linalg.cond(Ctrl), " det =", np.linalg.det(Ctrl))
Abar = np.linalg.inv(Dx) @ Ad @ Dx; Bbar = np.linalg.inv(Dx) @ Bd * Du
Ctrl_s = np.hstack([Bbar, Abar @ Bbar])
print("Abar=\n", Abar, "\nBbar=", Bbar.ravel())
print("cond(Ctrl) escalada =", np.linalg.cond(Ctrl_s), " det =", np.linalg.det(Ctrl_s))
for nm, Cy in [("solo iL", [[1, 0]]), ("solo vC", [[0, 1]]), ("ambos", np.eye(2))]:
    Cy = np.array(Cy, float); O = np.vstack([Cy, Cy @ Ad])
    print(f"O[{nm}]: rango={np.linalg.matrix_rank(O)}  sv={np.linalg.svd(O, compute_uv=False)}")
O_il = np.vstack([[1, 0], np.array([1, 0]) @ Ad]); O_vc = np.vstack([[0, 1], np.array([0, 1]) @ Ad])
print("det O(solo iL) =", np.linalg.det(O_il), " (= +a12 =", Ad[0, 1], ")  det O(solo vC) =", np.linalg.det(O_vc), " (= -a21 =", -Ad[1, 0], ")")
RES["cond_ctrl_fis"] = float(np.linalg.cond(Ctrl)); RES["cond_ctrl_esc"] = float(np.linalg.cond(Ctrl_s))
RES["cond_Ad_fis"] = float(np.linalg.cond(Ad)); RES["cond_Ad_esc"] = float(np.linalg.cond(Abar))
Q0f = np.linalg.inv(Dx) @ np.linalg.inv(Dx); R0f = np.array([[1 / Du**2]])

# ------------------------------------------------------------------ 2. LQR
def lqr(A, B, Q, Rm):
    P = solve_discrete_are(A, B, Q, Rm)
    K = -np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A)
    return P, K

def riccati_resid(A, B, Q, Rm, P):
    return A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A) + Q - P

H("2a. Ejemplo 2x2 de WP1 (verificacion)")
A1 = 0.5 * np.eye(2); B1 = np.array([[1.], [1.]])
P1, K1 = lqr(A1, B1, np.eye(2), np.array([[1.]]))
print("P=\n", P1, "\nK=", K1, "\neig(A+BK)=", np.linalg.eigvals(A1 + B1 @ K1))
RES["wp1_P"] = P1.tolist(); RES["wp1_K"] = K1.ravel().tolist()

H("2b. LQR nominal del boost (Q=Dx^-2, R=Du^-2)")
P, K = lqr(Ad, Bd, Q0f, R0f)
AK = Ad + Bd @ K
res = riccati_resid(Ad, Bd, Q0f, R0f, P)
Pv = Q0f.copy()
for _ in range(500000):
    Pn = Ad.T @ Pv @ Ad - Ad.T @ Pv @ Bd @ np.linalg.solve(R0f + Bd.T @ Pv @ Bd, Bd.T @ Pv @ Ad) + Q0f
    if abs(Pn - Pv).max() < 1e-13: Pv = Pn; break
    Pv = Pn
lyap = AK.T @ P @ AK - P + (Q0f + K.T @ R0f @ K)
print("P=\n", P, "\neig(P)=", np.linalg.eigvalsh(P))
print("K=", K, "\nAK=\n", AK, "\neig(AK)=", np.linalg.eigvals(AK), " |.|=", abs(np.linalg.eigvals(AK)))
print("residuo DARE =", abs(res).max(), "  |P - P_iteracion| =", abs(P - Pv).max())
print("identidad de Lyapunov: max|AK'P AK - P + Q + K'RK| =", abs(lyap).max())
print("Delta V (matriz) =\n", AK.T @ P @ AK - P, "\n-(Q+K'RK) =\n", -(Q0f + K.T @ R0f @ K))
print("eig(Delta V)=", np.linalg.eigvalsh(AK.T @ P @ AK - P))
Kbar = K @ Dx / Du
print("Kbar (escalada) =", Kbar)
eigAK = np.linalg.eigvals(AK)
RES.update(P=P.tolist(), K=K.ravel().tolist(), eigAK=[[z.real, z.imag] for z in eigAK],
           eigAK_abs=[float(abs(z)) for z in eigAK], res_dare=float(abs(res).max()),
           res_lyap=float(abs(lyap).max()), P_vs_iter=float(abs(P - Pv).max()), Kbar=Kbar.ravel().tolist())
sK = np.log(eigAK.astype(complex)) / Ts; print("polos equivalentes continuos de AK:", sK)
RES["polosK_cont"] = [[z.real, z.imag] for z in sK]

# ------------------------------------------------------------------ 3. barrido Q/R
def sim_lin(AKm, x0, n=300):
    X = [x0]
    for _ in range(n): X.append(AKm @ X[-1])
    return np.array(X)

def metrics(Kc, x0=np.array([0.0, 1.0]), n=1500):
    AKc = Ad + Bd @ Kc
    X = sim_lin(AKc, x0, n); U = (Kc @ X.T).ravel()
    nrm = np.linalg.norm(X @ np.linalg.inv(Dx).T, axis=1)
    idx = np.where(nrm > 0.02 * nrm[0])[0]
    ts = (idx[-1] + 1) * Ts * 1e3 if len(idx) and idx[-1] + 1 < len(nrm) else float("nan")
    return dict(rho_max=float(max(abs(np.linalg.eigvals(AKc)))), peak_d=float(abs(U).max()), ts_ms=float(ts))

H("3. Barrido R = rho*R0 (Q = Q0)")
rhos = np.logspace(-2, 4, 25); sweep = []
for r in rhos:
    Pr, Kr = lqr(Ad, Bd, Q0f, r * R0f); m = metrics(Kr)
    m.update(rho=float(r), K=Kr.ravel().tolist(), eig=[[z.real, z.imag] for z in np.linalg.eigvals(Ad + Bd @ Kr)])
    sweep.append(m)
print("rho     |lam|max   peak|d~|(vC0=1V)  ts2%[ms]   K")
for i in [0, 4, 8, 12, 16, 20, 24]:
    m = sweep[i]; print(f"{m['rho']:8.3g} {m['rho_max']:.5f}   {m['peak_d']:.4f}   {m['ts_ms']:.2f}   {np.round(m['K'],4)}")
RES["sweep"] = sweep
H("3b. Barrido de ponderacion relativa gamma = q_v/q_i (rho=1)")
gam_tab = []
for g in [0.01, 0.1, 1, 10, 100]:
    Qg = np.diag([1 / IMAX**2, g / VMAX**2]); Pg, Kg = lqr(Ad, Bd, Qg, R0f); m = metrics(Kg)
    m.update(gamma=g, K=Kg.ravel().tolist()); gam_tab.append(m)
    print(f"gamma={g:6g}  |lam|max={m['rho_max']:.5f}  peak|d|={m['peak_d']:.4f}  ts={m['ts_ms']:.2f} ms  K={np.round(Kg.ravel(),4)}")
RES["gamma"] = gam_tab
m0 = metrics(K); RES["nominal_metrics"] = m0
print("nominal: ", m0)
print("dvC0 maximo admisible por duty, lineal: |d~|<=0.45 ->", 0.45 / m0["peak_d"], "V ;  |d~|<=0.05 ->", 0.05 / m0["peak_d"], "V")
RES["dv_adm_045"] = 0.45 / m0["peak_d"]; RES["dv_adm_005"] = 0.05 / m0["peak_d"]

th = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(np.cos(th), np.sin(th), "k:", lw=.8)
for m in sweep:
    for e in m["eig"]: ax[0].plot(e[0], e[1], ".", color=plt.cm.viridis((np.log10(m["rho"]) + 2) / 6), ms=5)
for e in np.linalg.eigvals(Ad): ax[0].plot(e.real, e.imag, "rx")
ax[0].set_xlim(0.9, 1.005); ax[0].set_ylim(-0.12, 0.12); ax[0].set_xlabel("Re"); ax[0].set_ylabel("Im"); ax[0].grid(alpha=.3)
ax[0].set_title(r"autovalores de $A_d+B_dK$ al variar $\rho$ (x: lazo abierto)", fontsize=9)
ax[1].loglog(rhos, [m["peak_d"] for m in sweep], "b-o", ms=3)
ax[1].set_xlabel(r"$\rho$  ($R=\rho R_0$)"); ax[1].set_ylabel(r"pico $|\tilde d|$ ($\tilde v_C(0)=1$ V)", color="b"); ax[1].grid(alpha=.3, which="both")
a2 = ax[1].twinx(); a2.semilogx(rhos, [m["ts_ms"] for m in sweep], "r-s", ms=3); a2.set_ylabel(r"$t_{s,2\%}$ [ms]", color="r")
plt.tight_layout(); plt.savefig("figuras/fig_lqr_barrido.pdf"); plt.close()

# ----------------------------------------------- 4. lazo cerrado sobre el modelo promediado no lineal
H("4. Lazo cerrado LQR sobre el modelo promediado no lineal (ZOH, Ts=100us)")
def closed_loop(Kc, x0, n=400, sat=True):
    x = x0.copy(); X = [x.copy()]; D = []; U = []
    for _ in range(n):
        d = dS + float((Kc @ (x - xS)).ravel()[0]); U.append(d)
        if sat: d = min(max(d, D_MIN), D_MAX)
        D.append(d); x = step_nl(x, d); X.append(x.copy())
        if not np.all(np.isfinite(x)) or abs(x).max() > 1e6: break
    return np.array(X), np.array(D), np.array(U)

cases = {"dv=+1 V": np.array([iS, vC_s + 1]), "dv=+5 V": np.array([iS, vC_s + 5]),
         "dv=-5 V": np.array([iS, vC_s - 5]), "arranque": np.array([Vs / R, Vs])}
nl_out = {}; fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for nm, x0 in cases.items():
    X, D, U = closed_loop(K, x0)
    xl = [x0 - xS]
    for _ in range(len(X) - 1): xl.append(AK @ xl[-1])
    xl = np.array(xl) + xS
    fin = np.abs(X[-1] - xS)
    sat_frac = float(np.mean((D <= D_MIN + 1e-12) | (D >= D_MAX - 1e-12)))
    print(f"{nm:10s} d_sin_sat in [{U.min():.3f},{U.max():.3f}] frac_sat={sat_frac:.3f}  |x_fin-x*|={fin}  max|NL-lin|={np.abs(X-xl).max(axis=0)}")
    nl_out[nm] = dict(d_unsat_max=float(U.max()), d_unsat_min=float(U.min()), frac_sat=sat_frac,
                      fin=fin.tolist(), dev_nl_lin=np.abs(X - xl).max(axis=0).tolist(), x0=x0.tolist())
    t = np.arange(len(X)) * Ts * 1e3
    ax[0].plot(t, X[:, 0], label=nm); ax[1].plot(t, X[:, 1]); ax[2].plot(t[:len(D)], D)
ax[0].set_ylabel(r"$i_L$ [A]"); ax[1].set_ylabel(r"$v_C$ [V]"); ax[2].set_ylabel("$d$ aplicado (saturado)")
for a in ax: a.set_xlabel("t [ms]"); a.grid(alpha=.3)
ax[2].axhline(D_MAX, color="k", ls=":", lw=.8); ax[2].axhline(D_MIN, color="k", ls=":", lw=.8)
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig("figuras/fig_lqr_nolineal.pdf"); plt.close()
RES["nl_cases"] = nl_out

def converge(dv):
    X, D, U = closed_loop(K, np.array([iS, vC_s + dv]), n=800)
    return len(X) == 801 and np.all(np.isfinite(X)) and np.abs(X[-1] - xS).max() < 0.05
pos = max([dv for dv in np.arange(5, 100.01, 5.0) if converge(dv)], default=0)
neg = max([dv for dv in np.arange(5, 47.99, 1.0) if converge(-dv)], default=0)
print(f"mayor dv explorado (paso 5 V arriba / 1 V abajo, iL=i*) con convergencia saturando d en [0.05,0.95]: +{pos} V, -{neg} V")
RES["attr_pos"] = float(pos); RES["attr_neg"] = float(neg)

H("4b. Sintonia agresiva (rho=0.01, gamma=100): efecto de la saturacion del duty")
Pa, Ka = lqr(Ad, Bd, np.diag([1 / IMAX**2, 100 / VMAX**2]), 0.01 * R0f)
print("K agresiva =", Ka, " eig =", np.linalg.eigvals(Ad + Bd @ Ka))
ag = {}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for nm, x0 in {"dv=+5 V": np.array([iS, vC_s + 5]), "dv=-20 V": np.array([iS, vC_s - 20]), "arranque": np.array([Vs / R, Vs])}.items():
    for sat in (True,):
        X, D, U = closed_loop(Ka, x0, n=400)
        sat_frac = float(np.mean((D <= D_MIN + 1e-12) | (D >= D_MAX - 1e-12)))
        fin = np.abs(X[-1] - xS)
        print(f"{nm:10s} d_sin_sat in [{U.min():.3f},{U.max():.3f}] frac_sat={sat_frac:.3f} |x_fin-x*|={fin} len={len(X)}")
        ag[nm] = dict(d_unsat_min=float(U.min()), d_unsat_max=float(U.max()), frac_sat=sat_frac, fin=fin.tolist())
        t = np.arange(len(X)) * Ts * 1e3
        ax[0].plot(t, X[:, 0], label=nm); ax[1].plot(t, X[:, 1]); ax[2].plot(t[:len(D)], D)
ax[0].set_ylabel(r"$i_L$ [A]"); ax[1].set_ylabel(r"$v_C$ [V]"); ax[2].set_ylabel("$d$ aplicado (saturado)")
for a in ax: a.set_xlabel("t [ms]"); a.grid(alpha=.3)
ax[2].axhline(D_MAX, color="k", ls=":", lw=.8); ax[2].axhline(D_MIN, color="k", ls=":", lw=.8)
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig("figuras/fig_lqr_agresivo.pdf"); plt.close()
RES["agresivo"] = ag; RES["K_agresiva"] = Ka.ravel().tolist()
RES["eig_agresiva"] = [[z.real, z.imag] for z in np.linalg.eigvals(Ad + Bd @ Ka)]

# ---------------------------------------------------- 5. elipsoide y limite de duty
H("5. Conjuntos de nivel x'Px<=alpha y limite de duty")
Pi = np.linalg.inv(P); kPk = float((K @ Pi @ K.T).ravel()[0])
fig, ax = plt.subplots(figsize=(4.2, 3.6))
for dmax, c in ((0.45, "b"), (0.05, "r")):
    al = dmax**2 / kPk; ex = np.sqrt(al * np.diag(Pi))
    print(f"|d~|<={dmax}: alpha_max={al:.4f}; semiejes proyectados: iL +-{ex[0]:.3f} A, vC +-{ex[1]:.3f} V")
    RES[f"alpha_{dmax}"] = float(al); RES[f"semiejes_{dmax}"] = ex.tolist()
    Lc = np.linalg.cholesky(P); z = np.vstack([np.cos(th), np.sin(th)]) * np.sqrt(al)
    e = np.linalg.solve(Lc.T, z); ax.plot(e[0], e[1], c, label=fr"$|\tilde d|\leq{dmax}$")
ax.set_xlabel(r"$\tilde i_L$ [A]"); ax.set_ylabel(r"$\tilde v_C$ [V]"); ax.grid(alpha=.3); ax.legend()
ax.set_title(r"$\{x:\,x^\top P x\leq\alpha\}$ con $|Kx|\leq\tilde d_{max}$", fontsize=9)
plt.tight_layout(); plt.savefig("figuras/fig_elipsoide.pdf"); plt.close()

# -------------------------------------------- 6. validez lineal segun la ventana de evaluacion
H("6. Validez lineal vs ventana de evaluacion (metrica eps de la propuesta)")
def eps_window(dt, T):
    n = int(round(T / Ts)); t = np.arange(1, n + 1) * Ts
    sol = solve_ivp(lambda tt, y: f(y, dS + dt), [0, T], xS, t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12)
    xn = sol.y.T - xS
    xl = np.array([np.linalg.solve(Ac, (expm(Ac * tt) - np.eye(2)) @ (Bc * dt)).ravel() for tt in t])
    return [np.linalg.norm(xl[:, i] - xn[:, i]) / (np.linalg.norm(xn[:, i]) + 1e-9) for i in (0, 1)]
wins = [0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3, 100e-3]
dgrid = np.logspace(-4, np.log10(0.45), 22); tab6 = {}
def thr(e, lvl):
    if e.max() < lvl: return float("nan")
    i = int(np.argmax(e >= lvl))
    if i == 0: return float(dgrid[0])
    return float(np.exp(np.interp(np.log(lvl), np.log(e[i-1:i+1]), np.log(dgrid[i-1:i+1]))))
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for T in wins:
    e_pos = np.array([max(eps_window(+d, T)) for d in dgrid]); e_neg = np.array([max(eps_window(-d, T)) for d in dgrid])
    tab6[T] = dict(d2=min(thr(e_pos, .02), thr(e_neg, .02)), d5=min(thr(e_pos, .05), thr(e_neg, .05)),
                   d10=min(thr(e_pos, .10), thr(e_neg, .10)), emax045=float(max(e_pos[-1], e_neg[-1])))
    ax.loglog(dgrid, np.maximum(e_pos, e_neg), label=f"{T*1e3:g} ms")
    print(f"ventana {T*1e3:6.1f} ms: |d~|max eps<=2%: {tab6[T]['d2']:.4f}  5%: {tab6[T]['d5']:.4f}  10%: {tab6[T]['d10']:.4f}  eps(0.45)={tab6[T]['emax045']:.3f}")
ax.axhline(0.05, color="k", ls=":"); ax.set_xlabel(r"$|\tilde d|$"); ax.set_ylabel(r"$\max(\varepsilon_{i_L},\varepsilon_{v_C})$")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7, title="ventana", title_fontsize=7)
plt.tight_layout(); plt.savefig("figuras/fig_validez_ventana.pdf"); plt.close()
RES["validez_ventana"] = {f"{T*1e3:g}": v for T, v in tab6.items()}

# ----------------------------------------------- 7. residuo no lineal a un paso
H("7. Residuo no lineal a un paso: w = Phi_NL(x,d) - (Ad x + Bd u)")
def resid_box(imax, vmax, dmax, ng=9):
    W = []; lin = []
    for a in np.linspace(-imax, imax, ng):
        for b in np.linspace(-vmax, vmax, ng):
            for c in np.linspace(-dmax, dmax, ng):
                xn = step_nl(xS + np.array([a, b]), dS + c) - xS
                W.append(xn - (Ad @ np.array([a, b]) + (Bd * c).ravel())); lin.append(Bd.ravel() * c)
    return abs(np.array(W)).max(axis=0), np.abs(np.array(lin)).max(axis=0)
tab7 = []
for (im, vm, dm) in [(0.5, 1, 0.01), (1, 2, 0.05), (2, 5, 0.10), (3.2, 8, 0.45)]:
    wb, lb = resid_box(im, vm, dm)
    print(f"|iL~|<={im} A, |vC~|<={vm} V, |d~|<={dm}: max|w|=({wb[0]:.2e} A, {wb[1]:.2e} V); max|Bd d~|=({lb[0]:.3f} A, {lb[1]:.3f} V)")
    tab7.append(dict(im=im, vm=vm, dm=dm, w=wb.tolist(), lin=lb.tolist()))
RES["residuo_paso"] = tab7

# ------------------------------------------ 8. varios modelos locales: K fija vs re-disenada
H("8. Estabilidad del LQR de d*=0.5 en otros equilibrios (R=20 ohm)")
des = np.linspace(0.05, 0.90, 86); rf, rs = [], []
for de in des:
    ve = Vs / (1 - de); ie = ve / (R * (1 - de))
    A_, B_, E_ = linmodel(de, ve, ie); Ad_, Bd_, _ = zoh(A_, B_, E_, Ts)
    rf.append(max(abs(np.linalg.eigvals(Ad_ + Bd_ @ K))))
    _, Ke = lqr(Ad_, Bd_, Q0f, R0f); rs.append(max(abs(np.linalg.eigvals(Ad_ + Bd_ @ Ke))))
rf, rs = np.array(rf), np.array(rs)
bad = des[rf >= 1]; ok = des[rf < 1]
print("K fija estabiliza para d_e en [%.2f, %.2f]; inestable en: %s" % (ok.min(), ok.max(), "ninguno" if len(bad) == 0 else f"{bad.min():.2f}..{bad.max():.2f}"))
tab8 = []
for de in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    j = int(np.argmin(abs(des - de)))
    print(f"d_e={des[j]:.2f}: v*={Vs/(1-des[j]):.1f} V  |lam|max K fija={rf[j]:.5f}  re-disenada={rs[j]:.5f}")
    tab8.append(dict(d=float(des[j]), v=float(Vs / (1 - des[j])), fijo=float(rf[j]), agend=float(rs[j])))
RES["K_fija_ok"] = [float(ok.min()), float(ok.max())]; RES["K_fija_bad"] = [float(bad.min()), float(bad.max())] if len(bad) else []
RES["tabla_local"] = tab8
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(des, rf, label=r"$K$ fija ($d^*=0.5$)"); ax.plot(des, rs, "--", label=r"$K(d_e)$ re-diseñada")
ax.axhline(1, color="k", lw=.8); ax.axvline(0.5, color="gray", ls=":"); ax.set_ylim(0.97, 1.02)
ax.set_xlabel(r"$d_e$ (equilibrio)"); ax.set_ylabel(r"$\max|\lambda(A_d+B_dK)|$"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figuras/fig_modelos_locales.pdf"); plt.close()

json.dump(RES, open("datos/resultados_wp3.json", "w"), indent=1)
print("\nFIN: resultados en datos/resultados_wp3.json y figuras/")