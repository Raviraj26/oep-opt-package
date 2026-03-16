import numpy as np
import json
import time
from pathlib import Path


class BFGSCallback:
    """
    Callback for scipy.optimize.minimize.
    Records at every iteration:
      - x vector (current exponents)
      - function value f(x)
      - gradient vector g(x)  [if available]
      - gradient norm ||g||_2
      - wall time since start
      - iteration counter

    Usage:
        cb = BFGSCallback(
            fun=fun_wrapped,
            jac=jac_wrapped,        # pass None if no gradient
            logdir="./opt_log",     # directory for per-iter files
            verbose=True,
        )
        res = minimize(..., callback=cb)
        cb.summary()                # print convergence table
        cb.save_json("history.json")
    """

    def __init__(self, fun, jac=None, logdir=None, verbose=True):
        self.fun     = fun
        self.jac     = jac
        self.logdir  = Path(logdir) if logdir else None
        self.verbose = verbose

        # history lists — one entry per iteration
        self.iter_count = 0
        self.x_history  = []    # list of 1D arrays
        self.f_history  = []    # list of floats
        self.g_history  = []    # list of 1D arrays or None
        self.gnorm_hist = []    # list of floats
        self.time_hist  = []    # wall seconds since start

        self._t0 = time.perf_counter()

        if self.logdir:
            self.logdir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            header = (
                f"{'iter':>6}  {'f(x)':>16}  {'||g||_2':>14}"
                f"  {'wall(s)':>10}"
            )
            print(header)
            print("-" * len(header))

    # ------------------------------------------------------------------
    # scipy calls this once per iteration with the CURRENT x
    # ------------------------------------------------------------------
    def __call__(self, xk):
        self.iter_count += 1
        t_elapsed = time.perf_counter() - self._t0

        x   = np.array(xk, dtype=float).copy()
        #x   = np.exp(x) 
        f   = float(self.fun(x))
        g   = np.array(self.jac(x), dtype=float).copy() if self.jac else None
        gn  = float(np.linalg.norm(g)) if g is not None else float("nan")

        self.x_history.append(x)
        self.f_history.append(f)
        self.g_history.append(g)
        self.gnorm_hist.append(gn)
        self.time_hist.append(t_elapsed)

        if self.verbose:
            gn_str = f"{gn:>14.6e}" if np.isfinite(gn) else f"{'n/a':>14}"
            print(
                f"{self.iter_count:>6}  {f:>16.10e}  {gn_str}"
                f"  {t_elapsed:>10.2f}"
            )

        # write per-iteration files if logdir given
        if self.logdir:
            it = self.iter_count
            np.savetxt(self.logdir / f"x_{it:04d}.txt",        x,   fmt="%.16e")
            np.savetxt(self.logdir / f"f_{it:04d}.txt",        [f], fmt="%.16e")
            if g is not None:
                np.savetxt(self.logdir / f"g_{it:04d}.txt",    g,   fmt="%.16e")
            with open(self.logdir / "grad.log", "a") as fh:
                # each line: iteration  f  gnorm  g[0]  g[1]  ...
                g_str = "  ".join(f"{v:.16e}" for v in (g if g is not None else []))
                fh.write(
                    f"{it}  {f:.16e}  {gn:.16e}  {g_str}\n"
                )
            with open(self.logdir / "coeff.log", "a") as fs:
                # each line: iteration  f  gnorm  g[0]  g[1]  ...
                c_str = "  ".join(f"{np.exp(v):.17f}" for v in (x if x is not None else []))
                fs.write(
                    f"{it}  {f:.16e}  {gn:.16e}  {c_str}\n"
                )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def summary(self):
        """Print a convergence table at the end of optimization."""
        print("\n" + "=" * 60)
        print(f"BFGS CALLBACK SUMMARY  ({self.iter_count} iterations)")
        print("=" * 60)
        print(f"  f_initial  : {self.f_history[0]:.10e}")
        print(f"  f_final    : {self.f_history[-1]:.10e}")
        print(f"  f_reduction: {self.f_history[0] - self.f_history[-1]:.4e}")
        if self.g_history[-1] is not None:
            print(f"  ||g||_final: {self.gnorm_hist[-1]:.4e}")
        print(f"  total time : {self.time_hist[-1]:.2f} s")
        print(f"  x_final    :")
        for j, v in enumerate(self.x_history[-1]):
            print(f"    x[{j:2d}] = {v:.10e}")

    def save_json(self, path):
        """Serialise full history to JSON for later analysis."""
        history = {
            "iterations": self.iter_count,
            "f":          self.f_history,
            "gnorm":      self.gnorm_hist,
            "time":       self.time_hist,
            "x":          [x.tolist() for x in self.x_history],
            "g":          [g.tolist() if g is not None else None
                           for g in self.g_history],
        }
        with open(path, "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"History saved to {path}")

    def x_array(self):
        """Return x history as 2D array, shape (n_iter, n_exps)."""
        return np.vstack(self.x_history)

    def g_array(self):
        """Return gradient history as 2D array, shape (n_iter, n_exps).
           Returns None if no jac was provided."""
        if self.g_history[0] is None:
            return None
        return np.vstack(self.g_history)