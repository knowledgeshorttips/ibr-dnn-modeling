#!/usr/bin/env python3
"""
Deep Neural Network Modeling of Inverter-Based Resources for High-Fidelity Power System Dynamics
ONE file: simulator + input sweep + PyTorch training + evaluation.  Runs on Colab or on your own computer.

THE POWER SYSTEM (per-unit on the converter rating, 500 kVA, 415 V, 50 Hz)
    500 kVA grid-forming converter (LC filter, voltage and current PI loops, 1.2 pu current limiter,
    swing-equation / droop outer loops) -> 415 V / 11 kV step-up transformer -> 11 kV bus
    synchronous generator (classical model + governor, rated (1-share)/share x 500 kVA) -> same bus
    series R-L load at the bus (power factor 0.9)
    This is an islanded, inverter-dominated system: the converter and the generator share the load
    according to their droops.  Disturbance: a load step at t = 0.5 s and (optionally) a 100 ms
    short circuit at the 11 kV bus, which drives the converter into its current limit.

THREE MODELS ARE COMPARED, all against the same detailed simulator (the "reference"):
    1. Conventional reduced-order model: converter = ideal voltage source behind the transformer
       with a standard voltage-drop current limiter (no filter, no inner loops).
    2. Hybrid model (Model C): the same reduced-order model, but a PyTorch DNN learned from detailed
       simulations tells it how the real converter limits its current (magnitude and direction
       of the limited current).  It runs inside the system simulation like any other model.
    3. Direct DNN (Models A and B): a PyTorch network that maps the case inputs straight to
       the waveforms (frequency, bus voltage, converter current) and to five performance indicators.

WHAT IT DOES
    inputs (CONFIG) -> Latin-hypercube sample of cases -> detailed simulation of every case ->
    dataset -> train DNNs -> test on cases never seen in training -> tables, plots, saved models.

HOW TO RUN
    Colab : paste this whole file into ONE cell and run it (or upload it and run  !python ibr_dnn_pipeline.py).
            PyTorch, NumPy, SciPy and Matplotlib are already installed in Colab.
    Local : pip install numpy scipy matplotlib torch   then   python ibr_dnn_pipeline.py
    Time  : about 10-15 minutes with the default 300 cases on Colab (2 CPU cores, no GPU needed).
            The dataset is saved, so running again skips the simulations (see reuse_dataset).
            For a quick test use  main(n_cases=60, epochs=300, wave_epochs=300, lim_epochs=200).

WHERE TO CHANGE THE INPUTS: the CONFIG dictionary in the PIPELINE section (ranges of the six inputs,
number of cases, network size ...), or call  main(n_cases=500).
After a run, RESULT holds everything, and  predict_case(RESULT, share=0.7, Pl0=0.5, dL=0.2, Hc=3.0,
Hs=4.0, g_fault=5.0)  predicts the indicators of a new case and checks them with the simulator.
check_system()  prints the small-signal stability of the model over the input ranges.

KEY LIMITATIONS (state them in the paper)
    * All data come from simulation, not from measurements; the detailed model is itself an averaged
      (no switching) model with a classical generator model.
    * Cases where the converter loses synchronism are discarded, so the DNNs only describe stable
      cases.  Below an inverter share of about 0.45 this happens often after severe faults.
    * The hybrid model improves on the conventional one mainly in fault cases (current limiting),
      and its closed-loop runs can occasionally diverge; the evaluation reports how many.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# --------------------------------------------------------------------------- #
# Parameters (per-unit on the converter base: 500 kVA, 415 V, 50 Hz)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    # ratings of the grid-forming converter and its transformer
    S_base: float = 500e3       # VA
    V_lv: float = 415.0         # V line-line RMS (converter / LV side)
    V_hv: float = 11e3          # V line-line RMS (system bus)
    f0: float = 50.0            # Hz

    # converter output filter
    Rf: float = 0.003
    Lf: float = 0.08
    Cf: float = 0.05
    # step-up transformer 415 V / 11 kV
    Rt: float = 0.01
    Xt: float = 0.05
    # 11 kV system bus (cable / stray capacitance)
    Cb: float = 0.02

    # ---- case inputs (what the study sweeps) ----
    share: float = 0.5          # converter share of the installed generation capacity (0..1)
    Pl0: float = 0.5            # initial load as a fraction of the installed capacity
    Hc: float = 3.0             # virtual inertia of the grid-forming converter [s]
    Hs: float = 4.0             # inertia constant of the synchronous generator [s]
    pf: float = 0.9             # load power factor (lagging)

    # ---- synchronous generator (classical model, own-rating per-unit) ----
    Ra: float = 0.01
    Xs: float = 0.30            # transient reactance
    Dsg: float = 2.0            # damping
    Tg: float = 0.5             # governor + turbine lag [s]
    Rsg: float = 0.04           # governor droop (4 %)

    # ---- grid-forming outer loops ----
    Dp: float = 40.0            # P-f droop (2.5 %)
    Dd: float = 100.0           # extra damping on w - w_lowpass
    tau_w: float = 0.5
    kq: float = 0.05            # Q-V droop
    tau_pq: float = 0.01
    # ---- inner loops ----
    kpc: float = 1.0
    kic: float = 12.0
    kpv: float = 2.0
    kiv: float = 50.0
    Kff: float = 1.0
    Imax: float = 1.2           # converter current limit [pu]
    Kaw: float = 20.0

    @property
    def wb(self) -> float:
        return 2.0 * math.pi * self.f0

    @property
    def r(self) -> float:       # SG rating relative to the converter (500 kVA)
        return (1.0 - self.share) / self.share

    @property
    def cap(self) -> float:     # installed capacity [pu of 500 kVA]
        return 1.0 + self.r

    @property
    def tanphi(self) -> float:
        return math.tan(math.acos(self.pf))


@dataclass
class Scene:
    """Quantities that can change during a simulation (set-points, load, fault)."""
    PL: float = 0.5             # load active power at 1 pu voltage [pu of 500 kVA]
    gf: float = 0.0             # shunt fault conductance at the 11 kV bus [pu]
    Pref: float = 0.5           # converter active-power set-point
    Qref: float = 0.0           # converter reactive-power set-point
    Vref: float = 1.0
    wref: float = 1.0
    Pm0: float = 0.5            # SG governor power set-point (common base)
    Esg: float = 1.0            # SG internal EMF magnitude


D_NAMES = ["ws", "Pm", "delta", "wc", "Pf", "Qf", "wlp", "vcd", "vcq", "id", "iq",
           "xvd", "xvq", "xcd", "xcq", "igd", "igq", "vbd", "vbq", "isd", "isq",
           "ild", "ilq"]
R_NAMES = ["ws", "Pm", "delta", "wc", "Pf", "Qf", "wlp",
           "igd", "igq", "vbd", "vbq", "isd", "isq", "ild", "ilq"]
DI = {n: i for i, n in enumerate(D_NAMES)}
RI = {n: i for i, n in enumerate(R_NAMES)}
# indices of the reduced-model states inside the detailed state vector
D_TO_R = [DI[n] for n in R_NAMES]


def load_RX(p: Params, PL: float):
    """Series R-L load that draws PL at 1 pu voltage and power factor p.pf."""
    t = p.tanphi
    d = PL * (1.0 + t * t)
    return 1.0 / d, t / d


# --------------------------------------------------------------------------- #
# Detailed model: SG + grid-forming converter (filter, inner loops, limiter) + load
# --------------------------------------------------------------------------- #
def make_rhs_detailed(p: Params, sc: Scene):
    wb, r = p.wb, p.r
    Ls, Rs = p.Xs / r, p.Ra / r
    Rl, Xl = load_RX(p, sc.PL)
    Lf, Rf, Cf, Cb = p.Lf, p.Rf, p.Cf, p.Cb
    Lt, Rt = p.Xt, p.Rt
    kpc, kic, kpv, kiv = p.kpc, p.kic, p.kpv, p.kiv
    twoHs, twoHc = 2.0 * p.Hs * r, 2.0 * p.Hc
    Esg, gf = sc.Esg, sc.gf

    def f(t, x):
        (ws, Pm, dl, wc, Pf, Qf, wlp, vcd, vcq, i_d, i_q, xvd, xvq, xcd, xcq,
         igd, igq, vbd, vbq, isd, isq, ild, ilq) = x
        c, s = math.cos(dl), math.sin(dl)

        # synchronous generator (reference frame rotates with its rotor: E on the d axis)
        Pe = Esg * isd
        dws = (Pm - Pe - p.Dsg * r * (ws - 1.0)) / twoHs
        dPm = (sc.Pm0 - (r / p.Rsg) * (ws - 1.0) - Pm) / p.Tg
        disd = wb / Ls * (Esg - vbd - Rs * isd + ws * Ls * isq)
        disq = wb / Ls * (-vbq - Rs * isq - ws * Ls * isd)

        # converter measurements rotated into the converter's own frame
        vcd_c = vcd * c + vcq * s
        vcq_c = -vcd * s + vcq * c
        igd_c = igd * c + igq * s
        igq_c = -igd * s + igq * c
        id_c = i_d * c + i_q * s
        iq_c = -i_d * s + i_q * c

        P = vcd * igd + vcq * igq
        Q = vcq * igd - vcd * igq
        d_delta = wb * (wc - ws)
        dwc = (sc.Pref - Pf - p.Dp * (wc - sc.wref) - p.Dd * (wc - wlp)) / twoHc
        dwlp = (wc - wlp) / p.tau_w
        dPf = (P - Pf) / p.tau_pq
        dQf = (Q - Qf) / p.tau_pq
        E = sc.Vref + p.kq * (sc.Qref - Qf)

        # voltage loop, current limiter, current loop (converter frame)
        evd = E - vcd_c
        evq = -vcq_c
        ifd_u = kpv * evd + kiv * xvd - wc * Cf * vcq_c + p.Kff * igd_c
        ifq_u = kpv * evq + kiv * xvq + wc * Cf * vcd_c + p.Kff * igq_c
        mag = math.hypot(ifd_u, ifq_u)
        k = 1.0 if mag <= p.Imax else p.Imax / mag
        ifd_r, ifq_r = ifd_u * k, ifq_u * k
        dxvd = evd + p.Kaw * (ifd_r - ifd_u)
        dxvq = evq + p.Kaw * (ifq_r - ifq_u)
        eid = ifd_r - id_c
        eiq = ifq_r - iq_c
        vid_c = kpc * eid + kic * xcd - wc * Lf * iq_c + vcd_c
        viq_c = kpc * eiq + kic * xcq + wc * Lf * id_c + vcq_c
        vid = vid_c * c - viq_c * s
        viq = vid_c * s + viq_c * c

        # power circuit in the common frame (speed ws)
        did = wb / Lf * (vid - vcd - Rf * i_d + ws * Lf * i_q)
        diq = wb / Lf * (viq - vcq - Rf * i_q - ws * Lf * i_d)
        dvcd = wb / Cf * (i_d - igd + ws * Cf * vcq)
        dvcq = wb / Cf * (i_q - igq - ws * Cf * vcd)
        digd = wb / Lt * (vcd - vbd - Rt * igd + ws * Lt * igq)
        digq = wb / Lt * (vcq - vbq - Rt * igq - ws * Lt * igd)
        dvbd = wb / Cb * (igd + isd - ild - gf * vbd + ws * Cb * vbq)
        dvbq = wb / Cb * (igq + isq - ilq - gf * vbq - ws * Cb * vbd)
        dild = wb / Xl * (vbd - Rl * ild + ws * Xl * ilq)
        dilq = wb / Xl * (vbq - Rl * ilq - ws * Xl * ild)
        return np.array([dws, dPm, d_delta, dwc, dPf, dQf, dwlp, dvcd, dvcq, did, diq,
                         dxvd, dxvq, eid, eiq, digd, digq, dvbd, dvbq, disd, disq,
                         dild, dilq])

    return f


# --------------------------------------------------------------------------- #
# Reduced-order model: the converter is an ideal voltage source behind the
# transformer (swing + droop) with a voltage-drop current limiter.
# nn=None -> conventional model (current limit: |voltage drop| clipped to Imax*|Zt|, same
# direction).  nn=callable -> hybrid model: a DNN learned from the detailed model gives the
# magnitude and the rotation of the limited drop, i.e. how the real converter limits its current.
# --------------------------------------------------------------------------- #
def nn_features(igd_c, igq_c, vbd_c, vbq_c, E, dw):
    return np.array([igd_c, igq_c, vbd_c, vbq_c, E, dw])


def make_rhs_reduced(p: Params, sc: Scene, nn=None):
    wb, r = p.wb, p.r
    Ls, Rs = p.Xs / r, p.Ra / r
    Rl, Xl = load_RX(p, sc.PL)
    Cb, Lt, Rt = p.Cb, p.Xt, p.Rt
    Zlim = p.Imax * math.hypot(p.Rt, p.Xt)
    twoHs, twoHc = 2.0 * p.Hs * r, 2.0 * p.Hc
    Esg, gf = sc.Esg, sc.gf

    def f(t, x):
        (ws, Pm, dl, wc, Pf, Qf, wlp, igd, igq, vbd, vbq, isd, isq, ild, ilq) = x
        c, s = math.cos(dl), math.sin(dl)
        Pe = Esg * isd
        dws = (Pm - Pe - p.Dsg * r * (ws - 1.0)) / twoHs
        dPm = (sc.Pm0 - (r / p.Rsg) * (ws - 1.0) - Pm) / p.Tg
        disd = wb / Ls * (Esg - vbd - Rs * isd + ws * Ls * isq)
        disq = wb / Ls * (-vbq - Rs * isq - ws * Ls * isd)

        E = sc.Vref + p.kq * (sc.Qref - Qf)
        # ideal voltage source and its voltage drop across the transformer
        ddx, ddy = E * c - vbd, E * s - vbq
        dm = math.hypot(ddx, ddy)
        sr = dm / Zlim
        if sr <= 1.0:
            kx, ky = ddx, ddy                         # not limiting: ideal source
        else:
            kx, ky = ddx / sr, ddy / sr               # conventional limiter: |drop| = Imax*|Zt|
        gi = math.hypot(igd, igq) / p.Imax
        vbd_c = vbd * c + vbq * s
        dd_c = (E - vbd_c) / Zlim                     # depth of the voltage collapse (converter d axis)
        if nn is not None and dd_c > 2.0 and gi > 0.9:
            # the converter is at its current limit with a depressed bus voltage (the regime the
            # DNN was trained on): the DNN gives the magnitude ratio and the rotation angle of the
            # limited drop, i.e. how the real converter limits its current
            igd_c = igd * c + igq * s
            igq_c = -igd * s + igq * c
            vbq_c = -vbd * s + vbq * c
            mq, th = nn(nn_features(igd_c, igq_c, vbd_c, vbq_c, E, wc - ws))
            mq = min(max(mq, 0.85), 1.1)
            th = min(max(th, 0.0), 1.6)
            w = min(1.0, (dd_c - 2.0) / 2.0) * min(1.0, (gi - 0.9) / 0.1)
            w = w * w * (3.0 - 2.0 * w)               # smooth switch-on of the DNN
            ct, st = math.cos(th), math.sin(th)
            ux, uy = ddx / dm, ddy / dm
            nx = Zlim * mq * (ux * ct - uy * st)
            ny = Zlim * mq * (ux * st + uy * ct)
            kx, ky = (1.0 - w) * kx + w * nx, (1.0 - w) * ky + w * ny
        vcd = vbd + kx
        vcq = vbq + ky

        P = vcd * igd + vcq * igq
        Q = vcq * igd - vcd * igq
        d_delta = wb * (wc - ws)
        dwc = (sc.Pref - Pf - p.Dp * (wc - sc.wref) - p.Dd * (wc - wlp)) / twoHc
        dwlp = (wc - wlp) / p.tau_w
        dPf = (P - Pf) / p.tau_pq
        dQf = (Q - Qf) / p.tau_pq

        digd = wb / Lt * (vcd - vbd - Rt * igd + ws * Lt * igq)
        digq = wb / Lt * (vcq - vbq - Rt * igq - ws * Lt * igd)
        dvbd = wb / Cb * (igd + isd - ild - gf * vbd + ws * Cb * vbq)
        dvbq = wb / Cb * (igq + isq - ilq - gf * vbq - ws * Cb * vbd)
        dild = wb / Xl * (vbd - Rl * ild + ws * Xl * ilq)
        dilq = wb / Xl * (vbq - Rl * ilq - ws * Xl * ild)
        return np.array([dws, dPm, d_delta, dwc, dPf, dQf, dwlp, digd, digq, dvbd, dvbq,
                         disd, disq, dild, dilq])

    return f


# --------------------------------------------------------------------------- #
# Operating point (dispatch proportional to installed capacity, bus voltage 1 pu,
# nominal frequency).  Solves for the 23 states and the 4 dispatch parameters.
# --------------------------------------------------------------------------- #
def operating_point(p: Params):
    r, alpha = p.r, p.share
    PL = p.Pl0 * p.cap
    QL = PL * p.tanphi
    Rl, Xl = load_RX(p, PL)
    Zl = complex(Rl, Xl)
    Zs = complex(p.Ra, p.Xs) / r
    Zt = complex(p.Rt, p.Xt)

    # phasor initial guess (bus at 1 pu), rotated so the SG emf lies on the d axis
    Vb = 1.0 + 0j
    IL = Vb / Zl
    ig = alpha * IL
    is_ = IL - ig
    Esg_c = Vb + Zs * is_
    rot = np.exp(-1j * np.angle(Esg_c))
    Vb, IL, ig, is_, Esg_c = Vb * rot, IL * rot, ig * rot, is_ * rot, Esg_c * rot
    vc = Vb + Zt * ig
    dl = float(np.angle(vc))
    i_f = ig + 1j * p.Cf * vc
    Sc = vc * np.conj(ig)
    Ec = abs(vc)
    Pm = float((Esg_c * np.conj(is_)).real)
    x0 = np.array([1.0, Pm, dl, 1.0, Sc.real, Sc.imag, 1.0, vc.real, vc.imag,
                   i_f.real, i_f.imag, 0.0, 0.0, 0.0, 0.0, ig.real, ig.imag,
                   Vb.real, Vb.imag, is_.real, is_.imag, IL.real, IL.imag])
    par0 = np.array([Sc.real, Pm, abs(Esg_c), Sc.imag + (1.0 - Ec) / p.kq])
    u0 = np.concatenate([x0, par0])

    def resid(u):
        x, (Pref, Pm0, Esg, Qref) = u[:23], u[23:]
        sc = Scene(PL=PL, Pref=Pref, Qref=Qref, Pm0=Pm0, Esg=Esg)
        res = make_rhs_detailed(p, sc)(0.0, x)
        PLa = x[DI["vbd"]] * x[DI["ild"]] + x[DI["vbq"]] * x[DI["ilq"]]
        QLa = x[DI["vbq"]] * x[DI["ild"]] - x[DI["vbd"]] * x[DI["ilq"]]
        extra = [x[DI["ws"]] - 1.0,
                 x[DI["Pf"]] - alpha * PLa,
                 math.hypot(x[DI["vbd"]], x[DI["vbq"]]) - 1.0,
                 x[DI["Qf"]] - alpha * QLa]
        return np.concatenate([res, extra])

    sol, info, ier, msg = fsolve(resid, u0, full_output=True, xtol=1e-12)
    if ier != 1 or np.max(np.abs(resid(sol))) > 1e-7:
        raise RuntimeError(f"operating point not found: {msg}")
    x, (Pref, Pm0, Esg, Qref) = sol[:23], sol[23:]
    sc = Scene(PL=PL, Pref=Pref, Qref=Qref, Pm0=Pm0, Esg=Esg)
    return x, sc


def eigenvalues(p: Params):
    x0, sc = operating_point(p)
    f = make_rhs_detailed(p, sc)
    n = len(x0)
    J = np.zeros((n, n))
    for j in range(n):
        h = 1e-6 * max(1.0, abs(x0[j]))
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (f(0.0, xp) - f(0.0, xm)) / (2 * h)
    return np.linalg.eigvals(J)


# --------------------------------------------------------------------------- #
# Simulation with events (detailed / conventional reduced / hybrid reduced)
# --------------------------------------------------------------------------- #
def _signals(p: Params, X: np.ndarray, reduced: bool) -> dict:
    names = R_NAMES if reduced else D_NAMES
    ix = {n: i for i, n in enumerate(names)}
    igd, igq = X[:, ix["igd"]], X[:, ix["igq"]]
    vbd, vbq = X[:, ix["vbd"]], X[:, ix["vbq"]]
    return {
        "f_sys_hz": X[:, ix["ws"]] * p.f0,
        "f_gfm_hz": X[:, ix["wc"]] * p.f0,
        "V_bus": np.hypot(vbd, vbq),
        "I_conv": np.hypot(igd, igq),
        "P_conv": vbd * igd + vbq * igq,
        "delta": X[:, ix["delta"]],
        "Pm": X[:, ix["Pm"]],
    }


class _Budget(Exception):
    pass


def simulate(p: Params, events, t_end: float, dt_out: float = 2e-3,
             model: str = "detailed", nn=None, rtol: float = 1e-6, atol: float = 1e-8,
             max_step: float = 4e-3, max_nfev: int = 250000):
    """
    events : list of (time, {"PL": new load [pu], "gf": fault conductance [pu]})
    model  : "detailed" (23 states), "reduced" (conventional), "hybrid" (reduced + DNN, needs nn)
    Returns dict with t, X, signals and ok.
    """
    x_full, sc = operating_point(p)
    reduced = model != "detailed"
    x = x_full[D_TO_R].copy() if reduced else x_full.copy()
    maker = make_rhs_detailed if not reduced else (
        (lambda pp, ss: make_rhs_reduced(pp, ss, nn)) if model == "hybrid"
        else (lambda pp, ss: make_rhs_reduced(pp, ss, None)))

    counter = [0]

    def guard(fun):                      # stop runs that need absurdly many evaluations
        def g(t, y):
            counter[0] += 1
            if counter[0] > max_nfev:
                raise _Budget()
            return fun(t, y)
        return g

    ev_times = sorted({e[0] for e in events if 0.0 < e[0] < t_end})
    t_grid = np.arange(0.0, t_end + 0.5 * dt_out, dt_out)
    eps = 1e-12
    T, XS = [], []
    seg_start, ok = 0.0, True
    for seg_end in ev_times + [t_end]:
        last = seg_end == t_end
        mask = (t_grid >= seg_start - eps) & ((t_grid <= t_end + eps) if last
                                              else (t_grid < seg_end - eps))
        samples = t_grid[mask]
        t_eval = samples if last else np.append(samples, seg_end)
        try:
            sol = solve_ivp(guard(maker(p, sc)), (seg_start, seg_end), x, method="Radau",
                            t_eval=t_eval, rtol=rtol, atol=atol, max_step=max_step)
        except _Budget:
            ok = False
            break
        if not sol.success or not np.all(np.isfinite(sol.y)):
            ok = False
            break
        Y = sol.y.T
        T.append(samples)
        XS.append(Y[:len(samples)])
        x = Y[-1].copy()
        if not last:
            for t_e, ch in events:
                if abs(t_e - seg_end) < eps:
                    for k, v in ch.items():
                        setattr(sc, k, v)
        seg_start = seg_end
    if not T:
        return {"ok": False}
    X = np.vstack(XS)
    out = _signals(p, X, reduced)
    out.update(t=np.concatenate(T), X=X, ok=ok, sc0=sc)
    return out


# --------------------------------------------------------------------------- #
# Training samples for the hybrid model (from detailed trajectories)
# --------------------------------------------------------------------------- #
def training_samples(p: Params, sc: Scene, X: np.ndarray, s_min: float = 2.0):
    """(features, targets) for the limiter DNN, from one detailed trajectory.
    Only samples where the detailed converter is really limiting its current
    (|i_filter| >= 0.95*Imax) and the bus voltage is depressed so that the ideal source would exceed
    the limit by a factor s_min (same condition that switches the DNN on in make_rhs_reduced).
    targets = (m, theta): |actual drop| / (Imax*|Zt|)  and  its rotation relative to the ideal drop."""
    dl = X[:, DI["delta"]]
    c, s_ = np.cos(dl), np.sin(dl)
    E = sc.Vref + p.kq * (sc.Qref - X[:, DI["Qf"]])
    vb = X[:, DI["vbd"]] + 1j * X[:, DI["vbq"]]
    vc = X[:, DI["vcd"]] + 1j * X[:, DI["vcq"]]
    ideal = E * np.exp(1j * dl) - vb
    actual = vc - vb
    Zlim = p.Imax * math.hypot(p.Rt, p.Xt)
    depth = np.real(ideal * np.exp(-1j * dl))                  # d-axis drop in the converter frame
    keep = (depth > s_min * Zlim) & (np.hypot(X[:, DI["id"]], X[:, DI["iq"]]) >= 0.95 * p.Imax)
    rot = lambda a, b: (a * c + b * s_, -a * s_ + b * c)    # common -> converter frame
    igd_c, igq_c = rot(X[:, DI["igd"]], X[:, DI["igq"]])
    vbd_c, vbq_c = rot(X[:, DI["vbd"]], X[:, DI["vbq"]])
    Z = np.stack([igd_c, igq_c, vbd_c, vbq_c, E, X[:, DI["wc"]] - X[:, DI["ws"]]], axis=1)
    m = np.abs(actual) / Zlim
    th = np.angle(actual / ideal)
    return Z[keep], np.stack([m[keep], th[keep]], axis=1)


# =========================================================================== #
# =========================================================================== #
#                                 PIPELINE
#   sweep inputs -> simulate -> dataset -> PyTorch models -> metrics + plots
# =========================================================================== #
# =========================================================================== #
import copy
import importlib.util
import json

# --------------------------------------------------------------------------- #
# CONFIG: change the inputs of the study here (nothing else needs editing)
# --------------------------------------------------------------------------- #
CONFIG = dict(
    # ---- which cases to simulate -------------------------------------------
    n_cases=300,            # number of simulations (about 1.5-3 s each)
    seed=0,                 # random seed (sampling, train/test split, network initialisation)
    n_jobs=2,               # parallel processes (Colab free tier has 2 CPU cores)
    # Range (min, max) of every input. Cases are spread over these ranges with
    # Latin-hypercube sampling.
    ranges=dict(
        share=(0.45, 0.85),     # converter share of the installed generation capacity
                                # (0.85 = inverter-dominated, 0.45 = about half and half)
        Pl0=(0.30, 0.70),       # initial load as a fraction of the installed capacity
        dL=(-0.15, 0.25),       # load step at t_event, fraction of installed capacity
        Hc=(2.0, 6.0),          # virtual inertia of the grid-forming converter [s]
        Hs=(2.0, 6.0),          # inertia constant of the synchronous generator [s]
        g_fault=(-3.0, 10.0),   # shunt fault conductance at the 11 kV bus [pu of 500 kVA].
                                # Values < 0.5 mean "no fault" (about 27 % of the cases).
    ),
    # ---- simulation timing -------------------------------------------------
    t_end=4.0,              # simulated time [s]
    t_event=0.5,            # load step (and fault) start here [s]
    fault_duration=0.1,     # duration of the fault [s]
    dt_out=2e-3,            # output sampling used for the indicators [s]
    wave_dt=0.02,           # waveforms are stored as averages over this window [s]
    wave_pre=0.1,           # waveforms start this long before the event [s]
    # ---- files -------------------------------------------------------------
    dataset_file="ibr_dataset.npz",
    reuse_dataset=True,     # reuse the saved dataset if it matches the settings above
    out_dir="ibr_results",  # plots, CSV, metrics and trained models are saved here
    # ---- neural networks ---------------------------------------------------
    test_frac=0.15,         # fraction of cases kept for the final test
    val_frac=0.15,          # fraction used for early stopping
    hidden=64, depth=3,     # network size for the performance indicators
    epochs=1500, lr=3e-3, batch=32, patience=200, weight_decay=1e-5,
    train_waveforms=True,   # direct DNN: case inputs -> waveforms
    wave_hidden=128, wave_epochs=1500,
    train_hybrid=True,      # hybrid model: reduced-order physics + limiter DNN
    lim_hidden=64, lim_depth=2, lim_epochs=800, lim_batch=128, lim_noise=0.03,
    n_eval=None,            # test cases used for the closed-loop comparison (None = all)
)

INPUT_NAMES = ["share", "Pl0", "dL", "Hc", "Hs", "g_fault"]
IND_NAMES = ["peak_current_pu", "freq_dev_max_hz", "rocof_hz_s", "v_bus_min_pu", "f_steady_dev_hz"]
WAVE_NAMES = ["system frequency deviation [Hz]", "11 kV bus voltage [pu]", "converter current [pu]"]
WAVE_SHORT = ["f_dev", "V_bus", "I_conv"]


def _in_notebook() -> bool:
    """True inside Jupyter / Colab."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# One case: inputs -> simulation set-up, indicators, waveforms
# --------------------------------------------------------------------------- #
def case_to_sim(x, cfg: dict):
    """x = [share, Pl0, dL, Hc, Hs, g_fault]  ->  (Params, events)."""
    share, pl0, dl, hc, hs, g = [float(v) for v in x]
    p = Params(share=share, Pl0=pl0, Hc=hc, Hs=hs)
    t0 = cfg["t_event"]
    change = {"PL": (pl0 + dl) * p.cap}
    events = [(t0, change)]
    if g > 0.0:
        change["gf"] = g
        events.append((t0 + cfg["fault_duration"], {"gf": 0.0}))
    return p, events


def indicators(r: dict, t_event: float, f0: float = 50.0) -> np.ndarray:
    """Five scalar performance indicators of one run (measured after the event)."""
    t = r["t"]
    k = t >= t_event - 1e-9
    f = r["f_sys_hz"][k] - f0
    dt = t[1] - t[0]
    n = max(1, int(round(0.1 / dt)))                       # RoCoF over a 100 ms window
    rocof = np.abs(f[n:] - f[:-n]).max() / (n * dt) if len(f) > n else 0.0
    return np.array([
        r["I_conv"][k].max(),                              # peak converter current [pu]
        np.abs(f).max(),                                   # largest frequency deviation [Hz]
        rocof,                                             # largest rate of change of frequency [Hz/s]
        r["V_bus"][k].min(),                               # lowest 11 kV bus voltage [pu]
        f[-max(1, int(round(0.2 / dt))):].mean(),          # frequency deviation at the end [Hz]
    ])


def waveforms(r: dict, cfg: dict):
    """(3, T) array of 20 ms block averages: frequency deviation, bus voltage, converter current."""
    dt = cfg["dt_out"]
    stride = int(round(cfg["wave_dt"] / dt))
    i0 = int(round((cfg["t_event"] - cfg["wave_pre"]) / dt))
    chans = [r["f_sys_hz"] - 50.0, r["V_bus"], r["I_conv"]]
    out = []
    for c in chans:
        seg = c[i0:]
        n = (len(seg) // stride) * stride
        out.append(seg[:n].reshape(-1, stride).mean(axis=1))
    tt = r["t"][i0:][:n].reshape(-1, stride).mean(axis=1) - cfg["t_event"]
    return np.stack(out), tt


def _run_case(job):
    """Simulate one case with the detailed model. Returns (index, None) if it failed or
    lost synchronism."""
    i, x, cfg = job
    try:
        p, events = case_to_sim(x, cfg)
        r = simulate(p, events, cfg["t_end"], dt_out=cfg["dt_out"], model="detailed")
        if (not r["ok"]) or abs(r["X"][-1, DI["wc"]] - r["X"][-1, DI["ws"]]) > 2e-3:
            return i, None
        Z, T = training_samples(p, r["sc0"], r["X"])
        W, tw = waveforms(r, cfg)
        return i, (indicators(r, cfg["t_event"]), W, tw, Z, T)
    except Exception:
        return i, None


# --------------------------------------------------------------------------- #
# Dataset: sample inputs, run simulations, save
# --------------------------------------------------------------------------- #
def sample_inputs(cfg: dict) -> np.ndarray:
    """Latin-hypercube sample of the input ranges -> array (n_cases, 6)."""
    n = cfg["n_cases"]
    rng = np.random.default_rng(cfg["seed"])
    u = np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in INPUT_NAMES], axis=1)
    X = np.zeros_like(u)
    for j, name in enumerate(INPUT_NAMES):
        lo, hi = cfg["ranges"][name]
        X[:, j] = lo + u[:, j] * (hi - lo)
    # keep the load after the step inside [0.1, 0.95] of the installed capacity
    X[:, 2] = np.clip(X[:, 1] + X[:, 2], 0.10, 0.95) - X[:, 1]
    # faults weaker than 0.5 pu are treated as "no fault"
    X[X[:, 5] < 0.5, 5] = 0.0
    return X


def _meta(cfg: dict) -> str:
    keys = ["n_cases", "seed", "ranges", "t_end", "t_event", "fault_duration",
            "dt_out", "wave_dt", "wave_pre"]
    return json.dumps({k: cfg[k] for k in keys}, sort_keys=True)


def _map_parallel(fn, jobs, n_jobs, label, every=25):
    """Run fn over jobs in parallel (falls back to serial). Returns a list in job order."""
    n = len(jobs)
    t_start = time.time()
    out = []

    def report(done):
        if done % every == 0 or done == n:
            el = time.time() - t_start
            print(f"  {label}: {done}/{n} done, {el:.0f} s elapsed, about {el / done * (n - done):.0f} s left",
                  flush=True)

    if n_jobs > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as ex:
                for k, res in enumerate(ex.map(fn, jobs, chunksize=1), 1):
                    out.append(res)
                    report(k)
            return out
        except Exception as e:                                  # e.g. no fork on this OS
            print(f"  parallel run failed ({e!r}); running serially instead")
            out = []
    for k, job in enumerate(jobs, 1):
        out.append(fn(job))
        report(k)
    return out


def generate_dataset(cfg: dict) -> dict:
    X_all = sample_inputs(cfg)
    n = len(X_all)
    jobs = [(i, X_all[i], cfg) for i in range(n)]
    print(f"Simulating {n} cases with the detailed model ({cfg['n_jobs']} process(es)) ...")
    t_start = time.time()
    results = dict(_map_parallel(_run_case, jobs, cfg["n_jobs"], "detailed"))
    keep = [i for i in range(n) if results[i] is not None]
    if len(keep) < 40:
        raise RuntimeError(f"only {len(keep)} of {n} simulations succeeded; narrow the ranges")
    Lz = [results[i][3] for i in keep]
    data = dict(
        X=X_all[keep],
        Y=np.array([results[i][0] for i in keep]),
        W=np.array([results[i][1] for i in keep]),
        t_w=results[keep[0]][2],
        LZ=np.vstack(Lz),
        LT=np.vstack([results[i][4] for i in keep]),
        Lc=np.concatenate([np.full(len(z), j) for j, z in enumerate(Lz)]),
        sim_seconds=np.array(time.time() - t_start),
    )
    print(f"Kept {len(keep)} of {n} cases ({n - len(keep)} discarded: the converter lost synchronism "
          f"or the solver failed) in {time.time() - t_start:.0f} s")
    print(f"Current-limiter samples for the hybrid model: {len(data['LZ'])} "
          f"(from {len(np.unique(data['Lc']))} cases that reached the current limit)")
    np.savez_compressed(cfg["dataset_file"], meta=np.array(_meta(cfg)), **data)
    print(f"Saved dataset to {cfg['dataset_file']}")
    return data


def load_or_generate_dataset(cfg: dict) -> dict:
    path = cfg["dataset_file"]
    if cfg["reuse_dataset"] and os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        if str(d["meta"]) == _meta(cfg):
            print(f"Loaded existing dataset {path} ({len(d['X'])} cases). "
                  f"Delete the file or set reuse_dataset=False to re-simulate.")
            return {k: d[k] for k in d.files if k != "meta"}
        print(f"{path} exists but was made with different settings; re-simulating.")
    return generate_dataset(cfg)


def save_indicator_csv(data: dict, path: str):
    header = ",".join(INPUT_NAMES + IND_NAMES)
    np.savetxt(path, np.c_[data["X"], data["Y"]], delimiter=",", header=header, comments="")
    print(f"  saved {path}  (inputs + indicators, one row per simulation)")


# --------------------------------------------------------------------------- #
# Helpers for the neural-network part
# --------------------------------------------------------------------------- #
def _features(X: np.ndarray) -> np.ndarray:
    F = np.array(X, dtype=float, copy=True)
    F[:, 5] = np.log1p(F[:, 5])                       # fault conductance enters as log(1 + g)
    return F


class Scaler:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, A):
        std = A.std(axis=0)
        return cls(A.mean(axis=0), np.where(std < 1e-12, 1.0, std))

    def tf(self, A):
        return (A - self.mean) / self.std

    def inv(self, A):
        return A * self.std + self.mean


def _split(n: int, cfg: dict):
    rng = np.random.default_rng(cfg["seed"] + 1)
    idx = rng.permutation(n)
    n_te = max(1, int(round(cfg["test_frac"] * n)))
    n_va = max(1, int(round(cfg["val_frac"] * n)))
    return idx[n_te + n_va:], idx[n_te:n_te + n_va], idx[:n_te]       # train, val, test


def _r2(T, P):
    ss_res = ((T - P) ** 2).sum(axis=0)
    ss_tot = ((T - T.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def _linear_fit(Xs, Ys):
    coef, *_ = np.linalg.lstsq(np.c_[Xs, np.ones(len(Xs))], Ys, rcond=None)
    return coef


def _linear_predict(coef, Xs):
    return np.c_[Xs, np.ones(len(Xs))] @ coef


def fit_mlp(Xtr, Ytr, Xva, Yva, cfg, hidden, epochs, label="", seed=0,
            depth=None, batch=None, noise=0.0):
    """Train a fully connected network (PyTorch) with early stopping on the validation set."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    depth = cfg["depth"] if depth is None else depth
    batch = cfg["batch"] if batch is None else batch
    dims = [Xtr.shape[1]] + [hidden] * depth
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    layers.append(nn.Linear(dims[-1], Ytr.shape[1]))
    net = nn.Sequential(*layers)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=max(10, cfg["patience"] // 4))
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(Ytr, dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(Yva, dtype=torch.float32)
    n, bs = len(xt), min(batch, len(xt))
    gen = torch.Generator().manual_seed(seed)

    best_val, best_state, wait, ep = float("inf"), None, 0, 0
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        tr_loss = 0.0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            xb = xt[idx]
            if noise > 0:                                  # input noise makes the network robust
                xb = xb + noise * torch.randn(xb.shape, generator=gen)
            opt.zero_grad()
            loss = F.mse_loss(net(xb), yt[idx])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val = F.mse_loss(net(xv), yv).item()
        sched.step(val)
        if val < best_val - 1e-7:
            best_val, wait = val, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
        if ep % 200 == 0:
            print(f"  [{label}] epoch {ep:5d}   train MSE {tr_loss / n:.4f}   validation MSE {val:.4f}")
    if best_state is not None:                             # (None only if the loss became NaN)
        net.load_state_dict(best_state)
    net.eval()
    print(f"  [{label}] finished after {ep} epochs, best validation MSE {best_val:.4f} "
          f"(in standardised units)")
    return net


def _predict(net, A):
    import torch
    with torch.no_grad():
        return net(torch.tensor(A, dtype=torch.float32)).numpy().astype(float)


def net_forward_fn(net):
    """Copy a trained PyTorch network into plain numpy (fast to call inside the ODE solver)."""
    import torch.nn as tnn
    lay = [(m.weight.detach().numpy().astype(float).T, m.bias.detach().numpy().astype(float))
           for m in net if isinstance(m, tnn.Linear)]

    def f(A):
        a = np.atleast_2d(A)
        for W, b in lay[:-1]:
            z = a @ W + b
            a = z / (1.0 + np.exp(-z))                    # SiLU
        W, b = lay[-1]
        return a @ W + b
    return f


def _plt():
    import matplotlib
    if not _in_notebook():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "lines.linewidth": 1.5})
    return plt


def _finish(fig, path):
    plt = _plt()
    fig.tight_layout()
    fig.savefig(path, dpi=300)                         # high resolution for the journal
    fig.savefig(os.path.splitext(path)[0] + ".pdf")    # vector version
    print(f"  saved {path}  (and a .pdf copy)")
    if _in_notebook():
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Model A: performance indicators (case inputs -> five numbers)
# --------------------------------------------------------------------------- #
def train_indicator_model(data: dict, cfg: dict, split) -> dict:
    tr, va, te = split
    X, Y = data["X"], data["Y"]
    xs = Scaler.fit(_features(X[tr]))
    ys = Scaler.fit(Y[tr])
    Xtr, Xva, Xte = (xs.tf(_features(X[i])) for i in (tr, va, te))
    print(f"\n=== Model A: performance indicators  ({len(tr)} train / {len(va)} validation / "
          f"{len(te)} test cases) ===")
    net = fit_mlp(Xtr, ys.tf(Y[tr]), Xva, ys.tf(Y[va]), cfg, cfg["hidden"], cfg["epochs"],
                  "indicators")
    P_mlp = ys.inv(_predict(net, Xte))
    P_lin = ys.inv(_linear_predict(_linear_fit(Xtr, ys.tf(Y[tr])), Xte))
    T = Y[te]
    r2_m, r2_l = _r2(T, P_mlp), _r2(T, P_lin)
    mae_m = np.abs(T - P_mlp).mean(axis=0)

    print("\nTest-set accuracy (cases the network never saw):")
    print(f"{'indicator':<18}{'mean':>9}{'std':>9}{'MAE (NN)':>11}{'R2 (NN)':>9}{'R2 (linear)':>13}")
    for j, name in enumerate(IND_NAMES):
        print(f"{name:<18}{T[:, j].mean():9.3f}{T[:, j].std():9.3f}{mae_m[j]:11.4f}"
              f"{r2_m[j]:9.3f}{r2_l[j]:13.3f}")
    print("(R2 = 1 is perfect; R2 near 0 means no better than predicting the average.)")

    plt = _plt()
    fig, axes = plt.subplots(1, len(IND_NAMES), figsize=(4.0 * len(IND_NAMES), 4.0))
    for j, ax in enumerate(axes):
        lo = min(T[:, j].min(), P_mlp[:, j].min(), P_lin[:, j].min())
        hi = max(T[:, j].max(), P_mlp[:, j].max(), P_lin[:, j].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.scatter(T[:, j], P_lin[:, j], s=14, alpha=0.5, color="0.6",
                   label=f"linear (R2={r2_l[j]:.2f})")
        ax.scatter(T[:, j], P_mlp[:, j], s=14, label=f"neural net (R2={r2_m[j]:.2f})")
        ax.set_title(IND_NAMES[j], fontsize=10)
        ax.set_xlabel("detailed simulator")
        ax.set_ylabel("predicted")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    _finish(fig, os.path.join(cfg["out_dir"], "indicators_test.png"))
    return dict(net=net, xs=xs, ys=ys, r2_nn=r2_m, r2_linear=r2_l, mae_nn=mae_m, pred_test=P_mlp)


# --------------------------------------------------------------------------- #
# Model B: direct DNN, case inputs -> system waveforms
# --------------------------------------------------------------------------- #
def train_waveform_model(data: dict, cfg: dict, split, xs: Scaler) -> dict:
    tr, va, te = split
    W, X = data["W"], data["X"]
    N, C, T = W.shape
    mu = W[tr].mean(axis=(0, 2))
    sd = W[tr].std(axis=(0, 2))
    sd = np.where(sd < 1e-12, 1.0, sd)
    Wf = ((W - mu[None, :, None]) / sd[None, :, None]).reshape(N, C * T)
    Xtr, Xva, Xte = (xs.tf(_features(X[i])) for i in (tr, va, te))
    print(f"\n=== Model B: direct DNN for the waveforms ({C} signals x {T} time points per case) ===")
    net = fit_mlp(Xtr, Wf[tr], Xva, Wf[va], cfg, cfg["wave_hidden"], cfg["wave_epochs"],
                  "waveforms")

    def to_phys(A):
        return A.reshape(-1, C, T) * sd[None, :, None] + mu[None, :, None]

    P_mlp = to_phys(_predict(net, Xte))
    P_lin = to_phys(_linear_predict(_linear_fit(Xtr, Wf[tr]), Xte))
    true = W[te]

    def r2_channel(P):
        res = ((true - P) ** 2).sum(axis=(0, 2))
        tot = ((true - true.mean(axis=(0, 2), keepdims=True)) ** 2).sum(axis=(0, 2))
        return 1.0 - res / np.maximum(tot, 1e-12)

    rmse = np.sqrt(((true - P_mlp) ** 2).mean(axis=(0, 2)))
    r2_m, r2_l = r2_channel(P_mlp), r2_channel(P_lin)
    print("\nTest-set accuracy over all test cases and time points:")
    print(f"{'signal':<34}{'RMSE (NN)':>11}{'R2 (NN)':>9}{'R2 (linear)':>13}")
    for c in range(C):
        print(f"{WAVE_NAMES[c]:<34}{rmse[c]:11.4f}{r2_m[c]:9.3f}{r2_l[c]:13.3f}")
    print("(waveforms are %.0f ms block averages, so very fast transients are smoothed)"
          % (1e3 * cfg["wave_dt"]))
    return dict(net=net, mu=mu, sd=sd, r2_nn=r2_m, r2_linear=r2_l, rmse_nn=rmse,
                pred_test=P_mlp, to_phys=to_phys)


# --------------------------------------------------------------------------- #
# Model C: hybrid model = reduced-order physics + DNN for the converter current limiter
# --------------------------------------------------------------------------- #
def train_limiter_model(data: dict, cfg: dict, split):
    tr, va, te = split
    Lc, Z, T = data["Lc"], data["LZ"], data["LT"]
    in_tr, in_va, in_te = np.isin(Lc, tr), np.isin(Lc, va), np.isin(Lc, te)
    print(f"\n=== Model C: hybrid model, DNN for the converter current limiter ===")
    print(f"  limiter samples: {in_tr.sum()} train / {in_va.sum()} validation / {in_te.sum()} test")
    if in_tr.sum() < 200 or in_va.sum() < 20:
        print("  too few current-limiting samples (few cases with a severe fault): hybrid model skipped.")
        return None
    zs, ts = Scaler.fit(Z[in_tr]), Scaler.fit(T[in_tr])
    net = fit_mlp(zs.tf(Z[in_tr]), ts.tf(T[in_tr]), zs.tf(Z[in_va]), ts.tf(T[in_va]), cfg,
                  cfg["lim_hidden"], cfg["lim_epochs"], "limiter",
                  depth=cfg["lim_depth"], batch=cfg["lim_batch"], noise=cfg["lim_noise"])
    fwd = net_forward_fn(net)
    if in_te.sum() > 5:
        P = ts.inv(fwd(zs.tf(Z[in_te])))
        r2 = _r2(T[in_te], P)
        print(f"  held-out limiter samples: R2 magnitude ratio m = {r2[0]:.3f}, "
              f"rotation angle theta = {r2[1]:.3f}")
        print(f"  (the detailed converter limits its current with m = {T[:, 0].mean():.2f} +- {T[:, 0].std():.2f}"
              f" and a rotation of {np.degrees(T[:, 1]).mean():.0f} +- {np.degrees(T[:, 1]).std():.0f} degrees;"
              f" the conventional model assumes m = 1, 0 degrees)")

    def nn(z):
        o = ts.inv(fwd(zs.tf(np.asarray(z, dtype=float))))[0]
        return float(o[0]), float(o[1])

    return dict(net=net, zs=zs, ts=ts, nn=nn)


_NN = None          # set before the evaluation pool is created (inherited by the workers)


def _eval_case(job):
    """Run the conventional and the hybrid reduced-order model on one test case."""
    i, x, cfg, use_hybrid = job
    p, events = case_to_sim(x, cfg)
    out = {}
    models = ["reduced"] + (["hybrid"] if use_hybrid else [])
    for m in models:
        t0 = time.perf_counter()
        try:
            r = simulate(p, events, cfg["t_end"], dt_out=cfg["dt_out"], model=m,
                         nn=_NN if m == "hybrid" else None, max_nfev=120000)
        except Exception:
            r = {"ok": False}
        dt = time.perf_counter() - t0
        if r["ok"]:
            W, _ = waveforms(r, cfg)
            out[m] = (W, indicators(r, cfg["t_event"]), dt)
        else:
            out[m] = (None, None, dt)
    return i, out


def evaluate_system_models(data, cfg, split, ind, wav, lim) -> dict:
    """Closed-loop comparison on the test cases: conventional reduced-order model, hybrid
    model and direct DNN, all judged against the detailed simulator."""
    global _NN
    te = split[2]
    if cfg["n_eval"]:
        te = te[:cfg["n_eval"]]
    X, W, Y = data["X"], data["W"], data["Y"]
    use_h = lim is not None
    _NN = lim["nn"] if use_h else None
    print(f"\n=== Closed-loop evaluation on {len(te)} unseen cases (detailed model = reference) ===")
    jobs = [(i, X[i], cfg, use_h) for i in te]
    res = dict(_map_parallel(_eval_case, jobs, cfg["n_jobs"], "reduced / hybrid", every=10))

    n, C, T = len(te), W.shape[1], W.shape[2]
    nan_w, nan_i = np.full((n, C, T), np.nan), np.full((n, Y.shape[1]), np.nan)
    Wc, Wh, Ic, Ih = nan_w.copy(), nan_w.copy(), nan_i.copy(), nan_i.copy()
    tc, th = np.zeros(n), np.zeros(n)
    for k, i in enumerate(te):
        o = res[i]
        if o["reduced"][0] is not None:
            Wc[k], Ic[k] = o["reduced"][0], o["reduced"][1]
        tc[k] = o["reduced"][2]
        if use_h:
            if o["hybrid"][0] is not None:
                Wh[k], Ih[k] = o["hybrid"][0], o["hybrid"][1]
            th[k] = o["hybrid"][2]
    Wd, Id = W[te], Y[te]
    # direct DNNs (Model A and Model B) on the same cases
    pos = {int(i): k for k, i in enumerate(split[2])}
    sel = np.array([pos[int(i)] for i in te])
    Wn = wav["pred_test"][sel] if wav is not None else nan_w
    In = ind["pred_test"][sel]

    # ---- per-case results of every model, saved as CSV for your tables and graphs ----------
    ind_sets = {"detailed": Id, "conventional": Ic}
    wav_sets = {"detailed": Wd, "conventional": Wc}
    if use_h:
        ind_sets["hybrid"], wav_sets["hybrid"] = Ih, Wh
    ind_sets["direct_dnn"] = In
    if wav is not None:
        wav_sets["direct_dnn"] = Wn
    head = ["case_id"] + INPUT_NAMES + [f"{m}_{nm}" for m in ind_sets for nm in IND_NAMES]
    tab = np.column_stack([np.asarray(te), X[te]] + [ind_sets[m] for m in ind_sets])
    p_ind = os.path.join(cfg["out_dir"], "ibr_test_indicators_by_model.csv")
    np.savetxt(p_ind, tab, delimiter=",", header=",".join(head), comments="", fmt="%.6g")
    print(f"  saved {p_ind}  (one row per test case: inputs + indicators of every model)")
    tw_ = data["t_w"]
    for m, A in wav_sets.items():
        rows = [np.column_stack([np.full(len(tw_), te[k]), tw_, A[k].T]) for k in range(n)]
        p_w = os.path.join(cfg["out_dir"], f"ibr_test_waveforms_{m}.csv")
        np.savetxt(p_w, np.vstack(rows), delimiter=",", comments="", fmt="%.6g",
                   header=",".join(["case_id", "time_s"] + WAVE_NAMES))
        print(f"  saved {p_w}  (waveforms of the {m} model; a failed run is written as nan)")

    fault = X[te][:, 5] > 0
    ok_c = ~np.isnan(Wc[:, 0, 0])
    ok_h = ~np.isnan(Wh[:, 0, 0]) if use_h else np.zeros(n, bool)
    print(f"  conventional model: {ok_c.sum()} of {n} runs completed;  "
          f"hybrid model: {ok_h.sum()} of {n} completed"
          + (f" ({n - ok_h.sum()} diverged and are excluded below)" if use_h and ok_h.sum() < n else ""))

    def rmse(A, mask):
        return np.sqrt(np.nanmean((A[mask] - Wd[mask]) ** 2, axis=(0, 2)))

    def mae(A, mask):
        return np.nanmean(np.abs(A[mask] - Id[mask]), axis=0)

    summary = {}
    for label, sub in (("ALL test cases", np.ones(n, bool)), ("test cases WITH a fault (converter at its current limit)", fault)):
        mask = sub & ok_c & (ok_h if use_h else True)
        if mask.sum() < 2:
            continue
        print(f"\n  {label}  ({mask.sum()} cases where every model produced a result)")
        print(f"  Waveform RMSE vs detailed model (smaller is better)")
        print(f"  {'signal':<34}{'conventional':>13}{'hybrid':>10}{'direct DNN':>12}")
        for c in range(C):
            hh = f"{rmse(Wh, mask)[c]:10.4f}" if use_h else f"{'n/a':>10}"
            dd = f"{rmse(Wn, mask)[c]:12.4f}" if wav is not None else f"{'n/a':>12}"
            print(f"  {WAVE_NAMES[c]:<34}{rmse(Wc, mask)[c]:13.4f}{hh}{dd}")
        print(f"  Indicator mean absolute error vs detailed model")
        print(f"  {'indicator':<34}{'conventional':>13}{'hybrid':>10}{'direct DNN':>12}")
        for j, name in enumerate(IND_NAMES):
            hh = f"{mae(Ih, mask)[j]:10.4f}" if use_h else f"{'n/a':>10}"
            print(f"  {name:<34}{mae(Ic, mask)[j]:13.4f}{hh}{mae(In, mask)[j]:12.4f}")
        summary[label] = dict(
            n=int(mask.sum()),
            waveform_rmse=dict(conventional=rmse(Wc, mask).tolist(),
                               hybrid=rmse(Wh, mask).tolist() if use_h else None,
                               direct_dnn=rmse(Wn, mask).tolist() if wav is not None else None),
            indicator_mae=dict(conventional=mae(Ic, mask).tolist(),
                               hybrid=mae(Ih, mask).tolist() if use_h else None,
                               direct_dnn=mae(In, mask).tolist()))

    t_det = float(data["sim_seconds"]) * cfg["n_jobs"] / max(1, len(data["X"])) if "sim_seconds" in data else float("nan")
    print("\n  Computing time per case (one CPU core): detailed model about %.1f s | conventional %.1f s"
          " | hybrid %.1f s | direct DNN < 0.01 s" % (t_det, np.nanmean(tc), np.nanmean(th) if use_h else float("nan")))

    # ---- figure: four fault cases from mild to severe ---------------------------------------
    cand = np.nonzero(fault & ok_c & (ok_h if use_h else True))[0]
    if len(cand):
        cand = cand[np.argsort(Id[cand, 3])[::-1]]                    # sort by lowest bus voltage
        pick = cand[np.linspace(0, len(cand) - 1, min(4, len(cand))).astype(int)]
        plt = _plt()
        fig, axes = plt.subplots(C, len(pick), figsize=(4.6 * len(pick), 2.7 * C), sharex=True)
        axes = np.atleast_2d(axes).reshape(C, len(pick))
        tw = data["t_w"]
        for col, k in enumerate(pick):
            x = X[te][k]
            axes[0, col].set_title(f"share={x[0]:.2f} load {x[1]:.2f}->{x[1] + x[2]:.2f} g={x[5]:.1f}\n"
                                   f"Hc={x[3]:.1f} Hs={x[4]:.1f}", fontsize=8)
            for c in range(C):
                ax = axes[c, col]
                ax.plot(tw, Wd[k, c], "k", lw=1.8, label="detailed (reference)")
                ax.plot(tw, Wc[k, c], "C1--", lw=1.3, label="conventional reduced-order")
                if use_h:
                    ax.plot(tw, Wh[k, c], "C0-", lw=1.3, label="hybrid (physics + DNN)")
                if wav is not None:
                    ax.plot(tw, Wn[k, c], "C2:", lw=1.5, label="direct DNN")
                ax.grid(alpha=0.3)
                if col == 0:
                    ax.set_ylabel(WAVE_NAMES[c], fontsize=8)
        axes[0, 0].legend(fontsize=6)
        for ax in axes[-1]:
            ax.set_xlabel("time after the event [s]")
        _finish(fig, os.path.join(cfg["out_dir"], "system_waveforms_test.png"))
    return summary


# --------------------------------------------------------------------------- #
# Use the trained indicator model on a new case
# --------------------------------------------------------------------------- #
def predict_case(res: dict, share, Pl0, dL, Hc, Hs, g_fault=0.0, check_with_simulator=True):
    """
    Predict the five indicators for one new case. Example (after RESULT = main()):
        predict_case(RESULT, share=0.7, Pl0=0.5, dL=0.2, Hc=3.0, Hs=4.0, g_fault=5.0)
    """
    cfg, ind = res["cfg"], res["indicators"]
    g = 0.0 if g_fault < 0.5 else g_fault
    x = np.array([[share, Pl0, dL, Hc, Hs, g]], dtype=float)
    pred = ind["ys"].inv(_predict(ind["net"], ind["xs"].tf(_features(x))))[0]
    truth = None
    if check_with_simulator:
        _, out = _run_case((0, x[0], cfg))
        truth = None if out is None else out[0]
        if out is None:
            print("(the detailed simulator reports loss of synchronism or a solver failure for this case)")
    print("case:", {k: round(float(v), 3) for k, v in zip(INPUT_NAMES, x[0])})
    print(f"{'indicator':<18}{'neural net':>12}{'simulator':>12}")
    for j, name in enumerate(IND_NAMES):
        sim = f"{truth[j]:12.4f}" if truth is not None else f"{'n/a':>12}"
        print(f"{name:<18}{pred[j]:12.4f}{sim}")
    return pred


def check_system():
    """Quick sanity check of the power-system model: operating point and small-signal
    stability over the corners of the input ranges (all eigenvalues must be negative)."""
    print(f"{'share':>6} {'Pl0':>5} {'Hc':>4} {'Hs':>4} {'max Re(eig) [1/s]':>20}")
    for share in (0.45, 0.65, 0.85):
        for Pl0 in (0.3, 0.7):
            for Hc, Hs in ((2.0, 2.0), (6.0, 6.0)):
                ev = eigenvalues(Params(share=share, Pl0=Pl0, Hc=Hc, Hs=Hs))
                print(f"{share:6.2f} {Pl0:5.2f} {Hc:4.0f} {Hs:4.0f} {ev.real.max():20.3f}")


# --------------------------------------------------------------------------- #
def _banner(k, total, text):
    print("\n" + "=" * 78)
    print(f"STEP {k}/{total}: {text}")
    print("=" * 78)


def final_results(res: dict):
    """Print (and save as CSV) the final results: everything is measured on test cases that the
    networks never saw during training, against the detailed simulator."""
    cfg, d = res["cfg"], res["data"]
    ind, wav, lim, summ = res["indicators"], res["waveforms"], res["limiter"], res["summary"]
    print("\n" + "#" * 78)
    print("FINAL RESULTS  (measured on test cases the networks never saw in training)")
    print("#" * 78)
    print(f"Dataset: {len(d['X'])} detailed simulations, {len(res['split'][2])} of them kept for the final test.")

    print("\nA) PyTorch DNN: case inputs -> performance indicators   (R2: 1 = perfect, 0 = no skill)")
    print(f"   {'indicator':<18}{'DNN':>8}{'linear model':>14}")
    for n, a, b in zip(IND_NAMES, ind["r2_nn"], ind["r2_linear"]):
        print(f"   {n:<18}{a:8.3f}{b:14.3f}")
    if wav is not None:
        print("\nB) PyTorch DNN: case inputs -> system waveforms   (R2)")
        print(f"   {'signal':<34}{'DNN':>8}{'linear model':>14}")
        for n, a, b in zip(WAVE_NAMES, wav["r2_nn"], wav["r2_linear"]):
            print(f"   {n:<34}{a:8.3f}{b:14.3f}")

    key = [k for k in summ if "WITH a fault" in k] or list(summ)
    rows = []
    if key:
        s = summ[key[0]]
        print(f"\nC) Closed-loop models vs the detailed simulator, {key[0]} ({s['n']} cases)")
        print("   Error vs the detailed simulator (smaller is better); change = improvement over the")
        print("   conventional reduced-order model.")
        print(f"   {'':<34}{'conventional':>13}{'hybrid':>10}{'(change)':>10}{'direct DNN':>12}{'(change)':>10}")
        for grp, names, label in (("waveform_rmse", WAVE_NAMES, "waveform RMSE"),
                                  ("indicator_mae", IND_NAMES, "indicator MAE")):
            for j, name in enumerate(names):
                c = s[grp]["conventional"][j]
                h = s[grp]["hybrid"][j] if s[grp]["hybrid"] else None
                n_ = s[grp]["direct_dnn"][j] if s[grp]["direct_dnn"] else None
                ch = f"{100 * (c - h) / c:+9.0f}%" if h is not None and c > 0 else f"{'n/a':>10}"
                cn = f"{100 * (c - n_) / c:+9.0f}%" if n_ is not None and c > 0 else f"{'n/a':>10}"
                hs = f"{h:10.4f}" if h is not None else f"{'n/a':>10}"
                ns = f"{n_:12.4f}" if n_ is not None else f"{'n/a':>12}"
                print(f"   {label + ': ' + name:<34}{c:13.4f}{hs}{ch}{ns}{cn}")
                rows.append([key[0], label, name, c, h, n_])
        path = os.path.join(cfg["out_dir"], "ibr_final_results.csv")
        with open(path, "w") as fh:
            fh.write("subset,metric,name,conventional,hybrid,direct_dnn\n")
            for r in rows:
                fh.write(",".join(f'"{r[0]}"' if k == 0 else str(v) for k, v in enumerate(r)) + "\n")
        print(f"\n   saved {path}")
    print("\nHow to read this: the direct DNN is the most accurate but only covers this family of")
    print("scenarios; the hybrid model is an ordinary simulation model that can be used in other")
    print("networks, and it should beat the conventional model wherever the converter hits its")
    print("current limit.  Positive change = better than the conventional model.")


def main(**overrides):
    """Run everything. Any CONFIG entry can be overridden: main(n_cases=500, epochs=800)."""
    cfg = {**CONFIG, **overrides}
    os.makedirs(cfg["out_dir"], exist_ok=True)
    # check for PyTorch without importing it (the parallel simulations start before training)
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("PyTorch is not installed. Run:  pip install torch")

    _banner(1, 4, "simulate the power system for many input combinations")
    data = load_or_generate_dataset(cfg)
    save_indicator_csv(data, os.path.join(cfg["out_dir"], "ibr_indicators.csv"))
    split = _split(len(data["X"]), cfg)

    _banner(2, 4, "gather the simulation results as training data for PyTorch")
    print(f"  inputs      X: {data['X'].shape}   columns = {INPUT_NAMES}")
    print(f"  indicators  Y: {data['Y'].shape}   columns = {IND_NAMES}")
    print(f"  waveforms   W: {data['W'].shape}   (cases, signals, time points)")
    print(f"  limiter samples for the hybrid model: {data['LZ'].shape[0]}")
    print(f"  split by case: {len(split[0])} train / {len(split[1])} validation / {len(split[2])} test")

    import torch
    _banner(3, 4, f"train the PyTorch networks (PyTorch {getattr(torch, '__version__', '?')}, CPU)")
    res = dict(cfg=cfg, data=data, split=split)
    res["indicators"] = train_indicator_model(data, cfg, split)
    res["waveforms"] = (train_waveform_model(data, cfg, split, res["indicators"]["xs"])
                        if cfg["train_waveforms"] else None)
    res["limiter"] = train_limiter_model(data, cfg, split) if cfg["train_hybrid"] else None

    _banner(4, 4, "test on unseen cases, compare with the detailed simulator, save")
    res["summary"] = evaluate_system_models(data, cfg, split, res["indicators"], res["waveforms"],
                                            res["limiter"])
    with open(os.path.join(cfg["out_dir"], "ibr_summary.json"), "w") as fh:
        json.dump(res["summary"], fh, indent=2)

    path = os.path.join(cfg["out_dir"], "ibr_models.pt")
    torch.save(dict(
        input_names=INPUT_NAMES, indicator_names=IND_NAMES,
        indicator_net=res["indicators"]["net"].state_dict(),
        indicator_x_scaler=(res["indicators"]["xs"].mean.tolist(), res["indicators"]["xs"].std.tolist()),
        indicator_y_scaler=(res["indicators"]["ys"].mean.tolist(), res["indicators"]["ys"].std.tolist()),
        waveform_net=res["waveforms"]["net"].state_dict() if res["waveforms"] else None,
        limiter_net=res["limiter"]["net"].state_dict() if res["limiter"] else None,
        limiter_z_scaler=((res["limiter"]["zs"].mean.tolist(), res["limiter"]["zs"].std.tolist())
                          if res["limiter"] else None),
        limiter_t_scaler=((res["limiter"]["ts"].mean.tolist(), res["limiter"]["ts"].std.tolist())
                          if res["limiter"] else None),
        config={k: v for k, v in cfg.items()}), path)
    print(f"\nSaved trained models to {path}")
    final_results(res)
    print("\nDone. Plots, the dataset, CSV files and the metrics are in", os.path.abspath(cfg["out_dir"]))
    return res


if __name__ == "__main__":
    # In Jupyter/Colab: RESULT stays available afterwards, e.g.
    #   predict_case(RESULT, share=0.7, Pl0=0.5, dL=0.2, Hc=3.0, Hs=4.0, g_fault=5.0)
    RESULT = main()
