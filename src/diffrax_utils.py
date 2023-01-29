import numpy as np
import jax
import jax.numpy as jnp
from equinox.internal import ω
from diffrax.solver.runge_kutta import AbstractERK, ButcherTableau, CalculateJacobian, _scan
from diffrax.misc import linear_rescale
from diffrax.local_interpolation import AbstractLocalInterpolation
from diffrax.custom_types import Array, PyTree, Scalar
from diffrax.solver.base import vector_tree_dot
from diffrax.solution import RESULTS, is_okay
from diffrax import NoAdjoint
from typing import Optional


def _leaf_map(leaf):
    if isinstance(leaf, jnp.ndarray):
        if leaf.dtype == jnp.dtype(jnp.float64):
            return leaf.astype(jnp.float32)
        if leaf.dtype == jnp.dtype(jnp.complex128):
            return leaf.astype(jnp.complex64)
    return leaf

class NoAdjointFloat32(NoAdjoint):
    def loop(self, *args, **kwargs):
        # Process init_state argument
        if "init_state" in kwargs:
            kwargs["init_state"] = jax.tree_util.tree_map(_leaf_map, kwargs["init_state"])
        return super().loop(
            *args,
            **kwargs,
        )

# CODE BELOW BASED ON DIFFRAX
# Adding explicit float32 dtypes where needed

_tsit5_tableau_float32 = ButcherTableau(
    a_lower=(
        np.array([161 / 1000], dtype=np.float32),
        np.array(
            [
                -0.8480655492356988544426874250230774675121177393430391537369234245294192976164141156943e-2,  # noqa: E501
                0.3354806554923569885444268742502307746751211773934303915373692342452941929761641411569,  # noqa: E501
            ], dtype=np.float32
        ),
        np.array(
            [
                2.897153057105493432130432594192938764924887287701866490314866693455023795137503079289,  # noqa: E501
                -6.359448489975074843148159912383825625952700647415626703305928850207288721235210244366,  # noqa: E501
                4.362295432869581411017727318190886861027813359713760212991062156752264926097707165077,  # noqa: E501
            ], dtype=np.float32
        ),
        np.array(
            [
                5.325864828439256604428877920840511317836476253097040101202360397727981648835607691791,  # noqa: E501
                -11.74888356406282787774717033978577296188744178259862899288666928009020615663593781589,  # noqa: E501
                7.495539342889836208304604784564358155658679161518186721010132816213648793440552049753,  # noqa: E501
                -0.9249506636175524925650207933207191611349983406029535244034750452930469056411389539635e-1,  # noqa: E501
            ], dtype=np.float32
        ),
        np.array(
            [
                5.861455442946420028659251486982647890394337666164814434818157239052507339770711679748,  # noqa: E501
                -12.92096931784710929170611868178335939541780751955743459166312250439928519268343184452,  # noqa: E501
                8.159367898576158643180400794539253485181918321135053305748355423955009222648673734986,  # noqa: E501
                -0.7158497328140099722453054252582973869127213147363544882721139659546372402303777878835e-1,  # noqa: E501
                -0.2826905039406838290900305721271224146717633626879770007617876201276764571291579142206e-1,  # noqa: E501
            ], dtype=np.float32
        ),
        np.array(
            [
                0.9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1,  # noqa: E501
                1 / 100,
                0.4798896504144995747752495322905965199130404621990332488332634944254542060153074523509,  # noqa: E501
                1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331,  # noqa: E501
                -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677,  # noqa: E501
                2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841,  # noqa: E501
            ], dtype=np.float32
        ),
    ),
    b_sol=np.array(
        [
            0.9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1,  # noqa: E501
            1 / 100,
            0.4798896504144995747752495322905965199130404621990332488332634944254542060153074523509,  # noqa: E501
            1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331,  # noqa: E501
            -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677,  # noqa: E501
            2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841,  # noqa: E501
            0.0,
        ], dtype=np.float32
    ),
    b_error=np.array(
        [
            0.9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1  # noqa: E501
            - 0.9468075576583945807478876255758922856117527357724631226139574065785592789071067303271e-1,  # noqa: E501
            1 / 100
            - 0.9183565540343253096776363936645313759813746240984095238905939532922955247253608687270e-2,  # noqa: E501
            0.4798896504144995747752495322905965199130404621990332488332634944254542060153074523509  # noqa: E501
            - 0.4877705284247615707855642599631228241516691959761363774365216240304071651579571959813,  # noqa: E501
            1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331  # noqa: E501
            - 1.234297566930478985655109673884237654035539930748192848315425833500484878378061439761,  # noqa: E501
            -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677  # noqa: E501
            + 2.707712349983525454881109975059321670689605166938197378763992255714444407154902012702,  # noqa: E501
            2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841  # noqa: E501
            - 1.866628418170587035753719399566211498666255505244122593996591602841258328965767580089,  # noqa: E501
            -1 / 66,
        ], dtype=np.float32
    ),
    c=np.array(
        [
            161 / 1000,
            327 / 1000,
            9 / 10,
            0.9800255409045096857298102862870245954942137979563024768854764293221195950761080302604,  # noqa: E501
            1.0,
            1.0,
        ], dtype=np.float32
    ),
)


class _Tsit5Interpolation_float32(AbstractLocalInterpolation):
    y0: PyTree[Array[...]]
    y1: PyTree[Array[...]]  # Unused, just here for API compatibility
    k: PyTree[Array["order":7, ...]]  # noqa: F821

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:  # noqa: F821
        del left
        t0 = t0.astype(jnp.float32)
        if t1 is not None:
            t1 = t1.astype(jnp.float32)
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0.astype(jnp.float32), t0, self.t1.astype(jnp.float32))

        # TODO: write as a matrix-multiply or vmap'd polyval
        b1 = (
            np.float32(-1.0530884977290216)
            * t
            * (t - np.float32(1.3299890189751412))
            * (t**2 - np.float32(1.4364028541716351) * t + np.float32(0.7139816917074209))
        )
        b2 = np.float32(0.1017) * t**2 * (t**2 - np.float32(2.1966568338249754) * t + np.float32(1.2949852507374631))
        b3 = (
            np.float32(2.490627285651252793)
            * t**2
            * (t**2 - np.float32(2.38535645472061657) * t + np.float32(1.57803468208092486))
        )
        b4 = (
            np.float32(-16.54810288924490272)
            * (t - np.float32(1.21712927295533244))
            * (t - np.float32(0.61620406037800089))
            * t**2
        )
        b5 = (
            np.float32(47.37952196281928122)
            * (t - np.float32(1.203071208372362603))
            * (t - np.float32(0.658047292653547382))
            * t**2
        )
        b6 = np.float32(-34.87065786149660974) * (t - np.float32(1.2)) * (t - np.float32(0.666666666666666667)) * t**2
        b7 = np.float32(2.5) * (t - 1) * (t - np.float32(0.6)) * t**2
        return (
            self.y0**ω
            + vector_tree_dot(jnp.stack([b1, b2, b3, b4, b5, b6, b7]), self.k) ** ω
        ).ω


class Tsit5Float32(AbstractERK):
    tableau = _tsit5_tableau_float32
    interpolation_cls = _Tsit5Interpolation_float32

    def order(self, terms):
        return 5

    def _first(self, terms, t0, t1, y0, args):
        vf_expensive = terms.is_vf_expensive(t0, t1, y0, args)
        implicit_first_stage = (
            self.tableau.a_diagonal is not None and self.tableau.a_diagonal[0] != 0
        )
        # The gamut of conditions under which we need to evaluate `f0` or `k0`.
        #
        # If we're computing the Jacobian at the start of the step, then we
        # need this as a linearisation point.
        #
        # If the first stage is implicit, then we need this as a predictor for
        # where to start iterating from.
        #
        # If we're not scanning stages then we're definitely not deferring this
        # evaluation to the scan loop, so get it done now.
        need_f0_or_k0 = (
            self.calculate_jacobian == CalculateJacobian.every_step
            or implicit_first_stage
            or not self.scan_stages
        )
        fsal = self.tableau.fsal
        if fsal and vf_expensive:
            # If the vector field is expensive then we want to use vf_prods instead.
            # FSAL implies evaluating just the vector field, since we need to contract
            # the same vector field evaluation against two different controls.
            #
            # But "evaluating just the vector field" is, as just established, expensive.
            fsal = False
        if fsal and self.scan_stages and not need_f0_or_k0:
            # If we're scanning stages then we'd like to disable FSAL.
            # FSAL implies evaluating the vector field in `init` as well as in `step`.
            # But `scan_stages` is a please-compile-faster flag, so we should avoid the
            # extra tracing.
            #
            # However we disable-the-disabling if `need_f0_or_k0`, since in this case
            # we evaluate `f0` or `k0` anyway, so it wouldn't help. So we might as well
            # take advantage of the runtime benefits of FSAL.
            fsal = False
        return vf_expensive, implicit_first_stage, need_f0_or_k0, fsal

    def step(
        self,
        terms,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state,
        made_jump,
    ):
        #
        # Some Runge--Kutta methods have special structure that we can use to improve
        # efficiency.
        #
        # The famous one is FSAL; "first same as last". That is, the final evaluation
        # of the vector field on the previous step is the same as the first evaluation
        # on the subsequent step. We can reuse it and save an evaluation.
        # However note that this requires saving a vf evaluation, not a
        # vf-control-product. (This comes up when we have a different control on the
        # next step, e.g. as with adaptive step sizes, or with SDEs.)
        # As such we disable FSAL if a vf is expensive and a vf-control-product is
        # cheap. (The canonical example is the optimise-then-discretise adjoint SDE.
        # For this SDE, the vf-control product is a vector-Jacobian product, which is
        # notably cheaper than evaluating a full Jacobian.)
        #
        # Next we have SSAL; "solution same as last". That is, the output of the step
        # has already been calculated during the internal stage calculations. We can
        # reuse those and save a dot product.
        #
        # Finally we have a choice whether to save and work with vector field
        # evaluations (fs), or to save and work with (vector field)-control products
        # (ks).
        # The former is needed for implicit FSAL solvers: they need to obtain the
        # final f1 for the FSAL property, which means they need to do the implicit
        # solve in vf-space rather than (vf-control-product)-space, which means they
        # need to use `fs` to predict the initial point for the root finding operation.
        # Meanwhile the latter is needed when solving optimise-then-discretise adjoint
        # SDEs, for which vector field evaluations are prohibitively expensive, and we
        # must necessarily work only with the (much cheaper) vf-control-products. (In
        # this case this is the difference between computing a Jacobian and computing a
        # vector-Jacobian product.)
        # For other problems, we choose to use `ks`. This doesn't have a strong
        # rationale although it does have some minor efficiency points in its favour,
        # e.g. we need `ks` to perform dense interpolation if needed.
        #

        _implicit_later_stages = self.tableau.a_diagonal is not None and any(
            self.tableau.a_diagonal[1:] != 0
        )
        _vf_expensive, implicit_first_stage, need_f0_or_k0, fsal = self._first(
            terms, t0, t1, y0, args
        )
        ssal = self.tableau.ssal
        if _implicit_later_stages and fsal:
            use_fs = True
        elif _vf_expensive:
            use_fs = False
        else:  # Choice not as important here; we use ks for minor efficiency reasons.
            use_fs = False
        del _vf_expensive, _implicit_later_stages

        control = terms.contr(t0, t1)
        dt = t1 - t0

        #
        # Calculate `f0` and `k0`. If this is just a first explicit stage then we'll
        # sort that out later. But we might need these values for something else too
        # (as a predictor for implicit stages; as a linearisation point for a Jacobian).
        #

        f0 = None
        k0 = None
        if fsal:
            f0 = solver_state
            if not use_fs:
                # `made_jump` can be a tracer, hence the `is`.
                if made_jump is False:
                    # Fast-path for compilation in the common case.
                    k0 = terms.prod(f0, control)
                else:
                    k0 = jax.lax.cond(
                        made_jump,
                        lambda: terms.vf_prod(t0, y0, args, control),
                        lambda: terms.prod(f0, control),  # noqa: F821
                    )
        else:
            if need_f0_or_k0:
                if use_fs:
                    f0 = terms.vf(t0, y0, args)
                else:
                    k0 = terms.vf_prod(t0, y0, args, control)

        #
        # Calculate `jac_f` and `jac_k` (maybe). That is to say, the Jacobian for use
        # throughout an implicit method. In practice this is for SDIRK and ESDIRK
        # methods, which use the same Jacobian throughout every stage.
        #

        jac_f = None
        jac_k = None
        if self.calculate_jacobian == CalculateJacobian.every_step:
            assert self.tableau.a_diagonal is not None
            # Skipping the first element to account for ESDIRK methods.
            assert all(
                x == self.tableau.a_diagonal[1] for x in self.tableau.a_diagonal[2:]
            )
            diagonal0 = self.tableau.a_diagonal[1]
            if use_fs:
                if y0 is not None:
                    assert f0 is not None
                jac_f = self.nonlinear_solver.jac(
                    _implicit_relation_f,
                    f0,
                    (diagonal0, terms.vf, terms.prod, t0, y0, args, control),
                )
            else:
                if y0 is not None:
                    assert k0 is not None
                jac_k = self.nonlinear_solver.jac(
                    _implicit_relation_k,
                    k0,
                    (diagonal0, terms.vf_prod, t0, y0, args, control),
                )
            del diagonal0

        #
        # Allocate `fs` or `ks` as a place to store the stage evaluations.
        #

        if use_fs or (fsal and self.scan_stages):
            if f0 is None:
                # Only perform this trace if we have to; tracing can actually be
                # a bit expensive.
                f0_struct = eqx.filter_eval_shape(terms.vf, t0, y0, args)
            else:
                f0_struct = jax.eval_shape(lambda: f0)  # noqa: F821
        # else f0_struct deliberately left undefined, and is unused.

        num_stages = len(self.tableau.c) + 1
        if use_fs:
            fs = jax.tree_util.tree_map(lambda f: jnp.empty((num_stages,) + f.shape, dtype=jnp.float32), f0_struct)
            ks = None
        else:
            fs = None
            ks = jax.tree_util.tree_map(lambda k: jnp.empty((num_stages,) + jnp.shape(k), dtype=jnp.float32), y0)

        #
        # First stage. Defines `result`, `scan_first_stage`. Places `f0` and `k0` into
        # `fs` and `ks`. (+Redefines them if it's an implicit first stage.) Consumes
        # `f0` and `k0`.
        #

        if fsal:
            scan_first_stage = False
            result = RESULTS.successful
        else:
            if implicit_first_stage:
                scan_first_stage = False
                assert self.tableau.a_diagonal is not None
                diagonal0 = self.tableau.a_diagonal[0]
                if self.tableau.diagonal[0] == 1:
                    # No floating point error
                    t0_ = t1
                else:
                    t0_ = t0 + self.tableau.diagonal[0] * dt
                if use_fs:
                    if y0 is not None:
                        assert jac_f is not None
                    nonlinear_sol = self.nonlinear_solver(
                        _implicit_relation_f,
                        f0,
                        (diagonal0, terms.vf, terms.prod, t0_, y0, args, control),
                        jac_f,
                    )
                    f0 = nonlinear_sol.root
                    result = nonlinear_sol.result
                else:
                    if y0 is not None:
                        assert jac_k is not None
                    nonlinear_sol = self.nonlinear_solver(
                        _implicit_relation_k,
                        k0,
                        (diagonal0, terms.vf_prod, t0_, y0, args, control),
                        jac_k,
                    )
                    k0 = nonlinear_sol.root
                    result = nonlinear_sol.result
                del diagonal0, t0_, nonlinear_sol
            else:
                scan_first_stage = self.scan_stages
                result = RESULTS.successful

        if scan_first_stage:
            assert f0 is None
            assert k0 is None
        else:
            if use_fs:
                if y0 is not None:
                    assert f0 is not None
                fs = ω(fs).at[0].set(ω(f0)).ω
            else:
                if y0 is not None:
                    assert k0 is not None
                ks = ω(ks).at[0].set(ω(k0)).ω

        del f0, k0

        #
        # Iterate through the stages. Fills in `fs` and `ks`. Consumes
        # `scan_first_stage`.
        #

        if self.scan_stages:

            def _vector_tree_dot(_x, _y, _i):
                del _i
                return vector_tree_dot(_x, _y)

        else:

            def _vector_tree_dot(_x, _y, _i):
                return vector_tree_dot(_x, ω(_y)[:_i].ω)

        def eval_stage(_carry, _input):
            _, _, _fs, _ks, _result = _carry
            _i, _a_lower_i, _a_diagonal_i, _a_predictor_i, _c_i = _input

            #
            # Evaluate the linear combination of previous stages
            #

            if use_fs:
                _increment = _vector_tree_dot(_a_lower_i, _fs, _i)  # noqa: F821
                _increment = terms.prod(_increment, control)
            else:
                _increment = _vector_tree_dot(_a_lower_i, _ks, _i)  # noqa: F821
            _yi_partial = (y0**ω + _increment**ω).ω

            #
            # Is this an implicit or explicit stage?
            #

            if self.tableau.a_diagonal is None:
                _implicit_stage = False
            else:
                if self.scan_stages:
                    if scan_first_stage:  # noqa: F821
                        _diagonal = self.tableau.a_diagonal
                    else:
                        _diagonal = self.tableau.a_diagonal[1:]
                    _implicit_stage = any(_diagonal != 0)
                    if _implicit_stage and any(_diagonal == 0):
                        assert False, (
                            "Cannot have a mix of implicit and "
                            "explicit stages when scanning"
                        )
                    del _diagonal
                else:
                    _implicit_stage = _a_diagonal_i != 0

            #
            # Figure out if we're computing a vector field ("f") or a
            # vector-field-product ("k")
            #
            # Ask for fi if we're using fs; ask for ki if we're using ks. Makes sense!
            # In addition, ask for fi if we're on the last stage and are using
            # an FSAL scheme, as we'll be passing that on to the next step. If
            # we're scanning the stages then every stage uses the same logic so
            # override the last iteration check.
            #

            _last_iteration = _i == num_stages - 1
            _return_fi = use_fs or (fsal and (self.scan_stages or _last_iteration))
            _return_ki = not use_fs
            del _last_iteration

            #
            # Evaluate the stage
            #

            _ti = jnp.where(_c_i == 1, t1, t0 + _c_i * dt)  # No floating point error
            if _implicit_stage:
                assert _a_diagonal_i is not None
                # Predictor for where to start iterating from
                if _return_fi:
                    _f_pred = _vector_tree_dot(_a_predictor_i, fs, _i)  # noqa: F821
                else:
                    _k_pred = _vector_tree_dot(_a_predictor_i, ks, _i)  # noqa: F821
                # Determine Jacobian to use at this stage
                if self.calculate_jacobian == CalculateJacobian.every_stage:
                    if _return_fi:
                        _jac_f = self.nonlinear_solver.jac(
                            _implicit_relation_f,
                            _f_pred,
                            (
                                _a_diagonal_i,
                                terms.vf,
                                terms.prod,
                                _ti,
                                _yi_partial,
                                args,
                                control,
                            ),
                        )
                        _jac_k = None
                    else:
                        _jac_f = None
                        _jac_k = self.nonlinear_solver.jac(
                            _implicit_relation_k,
                            _k_pred,
                            (
                                _a_diagonal_i,
                                terms.vf,
                                terms.prod,
                                _ti,
                                _yi_partial,
                                args,
                                control,
                            ),
                        )
                else:
                    assert self.calculate_jacobian == CalculateJacobian.every_step
                    _jac_f = jac_f
                    _jac_k = jac_k
                # Solve nonlinear problem
                if _return_fi:
                    if y0 is not None:
                        assert _jac_f is not None
                    _nonlinear_sol = self.nonlinear_solver(
                        _implicit_relation_f,
                        _f_pred,
                        (
                            _a_diagonal_i,
                            terms.vf,
                            terms.prod,
                            _ti,
                            _yi_partial,
                            args,
                            control,
                        ),
                        _jac_f,
                    )
                    _fi = _nonlinear_sol.root
                    if _return_ki:
                        _ki = terms.prod(_fi, control)
                    else:
                        _ki = None
                else:
                    if _return_ki:
                        if y0 is not None:
                            assert _jac_k is not None
                        _nonlinear_sol = self.nonlinear_solver(
                            _implicit_relation_k,
                            _k_pred,
                            (
                                _a_diagonal_i,
                                terms.vf_prod,
                                _ti,
                                _yi_partial,
                                args,
                                control,
                            ),
                            _jac_k,
                        )
                        _fi = None
                        _ki = _nonlinear_sol.root
                    else:
                        assert False
                _result = update_result(_result, _nonlinear_sol.result)
                del _nonlinear_sol
            else:
                # Explicit stage
                if _return_fi:
                    _fi = terms.vf(_ti, _yi_partial, args)
                    if _return_ki:
                        _ki = terms.prod(_fi, control)
                    else:
                        _ki = None
                else:
                    _fi = None
                    if _return_ki:
                        _ki = terms.vf_prod(_ti, _yi_partial, args, control)
                    else:
                        assert False

            #
            # Store output
            #

            if use_fs:
                _fs = ω(_fs).at[_i].set(ω(_fi)).ω
            else:
                _ks = ω(_ks).at[_i].set(ω(_ki)).ω
            if ssal:
                _yi_partial_out = _yi_partial
            else:
                _yi_partial_out = None
            if fsal:
                _fi_out = _fi
            else:
                _fi_out = None
            return (_yi_partial_out, _fi_out, _fs, _ks, _result), None

        if self.scan_stages:
            if scan_first_stage:
                tableau_a_lower = np.zeros((num_stages, num_stages))
                for i, a_lower_i in enumerate(self.tableau.a_lower):
                    tableau_a_lower[i + 1, : i + 1] = a_lower_i
                tableau_a_diagonal = self.tableau.a_diagonal
                tableau_a_predictor = self.tableau.a_predictor
                tableau_c = np.zeros(num_stages)
                tableau_c[1:] = self.tableau.c
                i_init = 0
                assert tableau_a_diagonal is None
                assert tableau_a_predictor is None
            else:
                tableau_a_lower = np.zeros((num_stages - 1, num_stages))
                for i, a_lower_i in enumerate(self.tableau.a_lower):
                    tableau_a_lower[i, : i + 1] = a_lower_i
                if self.tableau.a_diagonal is None:
                    tableau_a_diagonal = None
                else:
                    tableau_a_diagonal = self.tableau.a_diagonal[1:]
                if self.tableau.a_predictor is None:
                    tableau_a_predictor = None
                else:
                    tableau_a_predictor = np.zeros((num_stages - 1, num_stages))
                    for i, a_predictor_i in enumerate(self.tableau.a_predictor):
                        tableau_a_predictor[i, : i + 1] = a_predictor_i
                tableau_c = self.tableau.c
                i_init = 1
            if ssal:
                y_dummy = y0
            else:
                y_dummy = None
            if fsal:
                f_dummy = jax.tree_util.tree_map(
                    lambda x: jnp.zeros(x.shape, dtype=x.dtype), f0_struct
                )
            else:
                f_dummy = None
            (y1_partial, f1, fs, ks, result), _ = jax.lax.scan(
                eval_stage,
                (y_dummy, f_dummy, fs, ks, result),
                (
                    np.arange(i_init, num_stages),
                    tableau_a_lower,
                    tableau_a_diagonal,
                    tableau_a_predictor,
                    tableau_c,
                ),
            )
            del y_dummy, f_dummy
        else:
            assert not scan_first_stage
            if self.tableau.a_diagonal is None:
                a_diagonal = None
            else:
                a_diagonal = self.tableau.a_diagonal[1:]
            for i, a_lower_i, a_diagonal_i, a_predictor_i, c_i in _scan(
                range(1, num_stages),
                self.tableau.a_lower,
                a_diagonal,
                self.tableau.a_predictor,
                self.tableau.c,
            ):
                (yi_partial, fi, fs, ks, result), _ = eval_stage(
                    (None, None, fs, ks, result),
                    (i, a_lower_i, a_diagonal_i, a_predictor_i, c_i),
                )
            y1_partial = yi_partial
            f1 = fi
            del a_diagonal, yi_partial, fi
        del scan_first_stage, _vector_tree_dot

        #
        # Compute step output
        #

        if ssal:
            y1 = y1_partial
        else:
            if use_fs:
                increment = vector_tree_dot(self.tableau.b_sol, fs)
                increment = terms.prod(increment, control)
            else:
                increment = vector_tree_dot(self.tableau.b_sol, ks)
            y1 = (y0**ω + increment**ω).ω

        #
        # Compute error estimate
        #

        if use_fs:
            y_error = vector_tree_dot(self.tableau.b_error, fs)
            y_error = terms.prod(y_error, control)
        else:
            y_error = vector_tree_dot(self.tableau.b_error, ks)
        y_error = jax.tree_util.tree_map(
            lambda _y_error: jnp.where(is_okay(result), _y_error, jnp.inf),
            y_error,
        )  # i.e. an implicit step failed to converge

        #
        # Compute dense info
        #

        if use_fs:
            if fs is None:
                # Edge case for diffeqsolve(y0=None)
                ks = None
            else:
                ks = jax.vmap(lambda f: terms.prod(f, control))(fs)
        dense_info = dict(y0=y0, y1=y1, k=ks)

        #
        # Compute next solver state
        #

        if fsal:
            solver_state = f1
        else:
            solver_state = None

        return y1, y_error, dense_info, solver_state, result
