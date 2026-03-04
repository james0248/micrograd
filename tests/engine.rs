use micrograd::engine::{Op, Value, clear_graph, no_grad, reset_state, stats, with_grad};

fn assert_close(actual: f64, expected: f64, eps: f64) {
    assert!(
        (actual - expected).abs() <= eps,
        "expected {expected:.12}, got {actual:.12} (eps={eps})"
    );
}

fn eval_scalar_expr(x: f64) -> f64 {
    no_grad(|| {
        let x_node = Value::new(x);
        let xx = x_node.mul(&x_node);
        let s = xx.add(&x_node);
        s.tanh().data()
    })
}

#[test]
fn value_new_works_on_default_tape() {
    reset_state();
    let v = Value::new(1.0);
    assert_close(v.data(), 1.0, 1e-12);

    let snapshot = stats();
    assert_eq!(snapshot.generation, 0);
    assert_eq!(snapshot.context_depth, 0);
    assert!(snapshot.with_grad_active);
}

#[test]
fn value_basics_work_inside_with_grad() {
    reset_state();
    with_grad(|| {
        let v = Value::new(3.5);
        assert_close(v.data(), 3.5, 1e-12);
        assert_close(v.grad(), 0.0, 1e-12);
        assert!(v.is_leaf());
        assert_eq!(v.op(), None);
        assert_eq!(v.parents().len(), 0);

        v.set_data(-1.25);
        assert_close(v.data(), -1.25, 1e-12);
        v.set_grad(2.0);
        v.add_grad(0.5);
        assert_close(v.grad(), 2.5, 1e-12);
        v.zero_grad();
        assert_close(v.grad(), 0.0, 1e-12);
    });
}

#[test]
fn forward_ops_are_correct() {
    reset_state();
    with_grad(|| {
        let a = Value::new(4.0);
        let b = Value::new(2.0);

        let sum = a.add(&b);
        assert_close(sum.data(), 6.0, 1e-12);
        assert_eq!(sum.op(), Some(Op::Add));

        let product = a.mul(&b);
        assert_close(product.data(), 8.0, 1e-12);
        assert_eq!(product.op(), Some(Op::Mul));

        let diff = a.sub(&b);
        assert_close(diff.data(), 2.0, 1e-12);
        assert_eq!(diff.op(), Some(Op::Sub));

        let quotient = a.div(&b);
        assert_close(quotient.data(), 2.0, 1e-12);
        assert_eq!(quotient.op(), Some(Op::Div));

        let negated = -a;
        assert_close(negated.data(), -4.0, 1e-12);
        assert_eq!(negated.op(), Some(Op::Neg));

        let exponent = Value::new(2.0);
        let power = a.pow(&exponent);
        assert_close(power.data(), 16.0, 1e-12);
        assert_eq!(power.op(), Some(Op::Pow));
    });
}

#[test]
fn unary_ops_are_correct() {
    reset_state();
    with_grad(|| {
        let x = Value::new(1.0);
        assert_close(x.exp().data(), std::f64::consts::E, 1e-12);

        let e = Value::new(std::f64::consts::E);
        assert_close(e.log().data(), 1.0, 1e-12);

        let t = Value::new(0.5);
        assert_close(t.tanh().data(), 0.5f64.tanh(), 1e-12);

        let neg = Value::new(-3.0).relu();
        assert_close(neg.data(), 0.0, 1e-12);

        let pos = Value::new(2.5).relu();
        assert_close(pos.data(), 2.5, 1e-12);
    });
}

#[test]
fn backward_chain_rule_shared_subgraph() {
    reset_state();
    with_grad(|| {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let shared = a.mul(&b);
        let out = shared.add(&shared);

        out.backward();

        assert_close(out.grad(), 1.0, 1e-12);
        assert_close(shared.grad(), 2.0, 1e-12);
        assert_close(a.grad(), 6.0, 1e-12);
        assert_close(b.grad(), 4.0, 1e-12);
    });
}

#[test]
fn operator_and_method_paths_match() {
    reset_state();
    with_grad(|| {
        let mx = Value::new(2.0);
        let my = Value::new(-3.0);
        let mz = Value::new(10.0);
        let m_out = mx.mul(&my).add(&mx).add(&mz).tanh();
        m_out.backward();

        let m_val = m_out.data();
        let mx_grad = mx.grad();
        let my_grad = my.grad();
        let mz_grad = mz.grad();

        let ox = Value::new(2.0);
        let oy = Value::new(-3.0);
        let oz = Value::new(10.0);
        let o_out = (&(&ox * &oy) + &ox + oz).tanh();
        o_out.backward();

        assert_close(m_val, o_out.data(), 1e-12);
        assert_close(mx_grad, ox.grad(), 1e-12);
        assert_close(my_grad, oy.grad(), 1e-12);
        assert_close(mz_grad, oz.grad(), 1e-12);
    });
}

#[test]
fn finite_difference_sanity() {
    reset_state();
    let x0 = 1.5;

    let x = Value::new(x0);
    let xx = x.mul(&x);
    let s = xx.add(&x);
    let out = s.tanh();
    out.backward();
    let analytical = x.grad();

    let h = 1e-6;
    let numerical = (eval_scalar_expr(x0 + h) - eval_scalar_expr(x0 - h)) / (2.0 * h);

    assert_close(analytical, numerical, 1e-6);
}

#[test]
fn no_grad_does_not_leak_temps() {
    reset_state();
    let baseline = stats().temp_count;

    let data = no_grad(|| {
        let a = Value::new(1.5);
        let b = a.tanh();
        let c = b.exp();
        c.data()
    });

    assert!(data.is_finite());
    assert_eq!(stats().temp_count, baseline);
}

#[test]
fn no_grad_inside_with_grad_restores_outer_graph() {
    reset_state();
    with_grad(|| {
        let x = Value::new(2.0);
        let before = stats().temp_count;

        let probe = no_grad(|| {
            let t = x.tanh().exp();
            t.data()
        });
        assert!(probe.is_finite());
        assert_eq!(stats().temp_count, before);

        let y = x.mul(&x);
        y.backward();
        assert_close(x.grad(), 4.0, 1e-12);
    });
}

#[test]
fn nested_with_grad_is_allowed() {
    reset_state();
    with_grad(|| {
        let outer = Value::new(2.0);

        with_grad(|| {
            let inner = outer.mul(&Value::new(3.0));
            assert_close(inner.data(), 6.0, 1e-12);
        });

        let z = outer.mul(&outer);
        z.backward();
        assert_close(outer.grad(), 4.0, 1e-12);
    });
}

#[test]
fn inner_scope_values_are_stale_after_exit() {
    reset_state();

    with_grad(|| {
        let inner = with_grad(|| Value::new(5.0));
        let stale = std::panic::catch_unwind(|| inner.data());
        assert!(
            stale.is_err(),
            "inner-scope value should be stale after exit"
        );
    });
}

#[test]
fn backward_on_default_tape_works() {
    reset_state();

    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let out = a.mul(&b);
    out.backward();

    assert_close(a.grad(), 3.0, 1e-12);
    assert_close(b.grad(), 2.0, 1e-12);
}

#[test]
fn backward_on_no_grad_value_panics() {
    reset_state();

    let result = std::panic::catch_unwind(|| {
        no_grad(|| {
            let a = Value::new(2.0);
            let b = Value::new(3.0);
            let out = a.mul(&b);
            out.backward();
        });
    });
    assert!(result.is_err(), "backward on no_grad value should panic");
}

#[test]
fn ancestor_root_backward_allowed_while_nested() {
    reset_state();

    with_grad(|| {
        let x = Value::new(2.0);
        let root = x.mul(&x);

        with_grad(|| {
            let _inner = x.mul(&Value::new(3.0));
            root.backward();
        });

        assert_close(x.grad(), 4.0, 1e-12);
    });
}

#[test]
fn backward_with_options_is_compatible() {
    reset_state();

    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let out = a.mul(&b);

    out.backward_with_options(true);
    assert_close(a.grad(), 3.0, 1e-12);
    out.backward();
    assert_close(a.grad(), 6.0, 1e-12);
}

#[test]
fn two_losses_in_same_tape_work() {
    reset_state();
    let w = Value::parameter(1.5);

    let x1 = Value::new(2.0);
    let l1 = w.mul(&x1);
    l1.backward();
    assert_close(w.grad(), 2.0, 1e-12);

    let x2 = Value::new(-3.0);
    let l2 = w.mul(&x2);
    l2.backward();
    assert_close(w.grad(), -1.0, 1e-12);
}

#[test]
fn parameters_survive_context_cleanup_and_clear_graph() {
    reset_state();
    let p = Value::parameter(1.0);

    with_grad(|| {
        let x = Value::new(2.0);
        let y = x.mul(&p);
        y.backward();
        assert_close(p.grad(), 2.0, 1e-12);
    });

    assert_close(p.data(), 1.0, 1e-12);
    p.set_data(3.0);
    assert_close(p.data(), 3.0, 1e-12);

    clear_graph();
    assert_close(p.data(), 3.0, 1e-12);
}

#[test]
fn clear_graph_panics_with_active_scope() {
    reset_state();

    let result = std::panic::catch_unwind(|| {
        with_grad(|| {
            clear_graph();
        });
    });

    assert!(
        result.is_err(),
        "clear_graph should panic inside active scope"
    );
}

#[test]
fn stats_temp_count_includes_all_active_tapes() {
    reset_state();

    with_grad(|| {
        let _outer_node = Value::new(1.0);
        let outer = stats();
        let outer_temp_count = outer.temp_count;

        assert_eq!(outer.context_depth, 1);
        assert!(outer_temp_count >= 1);

        with_grad(|| {
            let _inner_node = Value::new(2.0);
            let nested = stats();
            assert_eq!(nested.context_depth, 2);
            assert_eq!(nested.temp_count, outer_temp_count + 1);
        });

        let after_nested = stats();
        assert_eq!(after_nested.context_depth, 1);
        assert_eq!(after_nested.temp_count, outer_temp_count);
    });
}

#[test]
fn deep_chain_backward_stays_correct() {
    reset_state();

    let x0 = Value::new(1.25);
    let mut out = x0;

    for _ in 0..8000 {
        out = out.mul(&Value::new(1.0001));
    }

    out.backward();
    assert!(x0.grad().is_finite(), "deep-chain grad should be finite");
    assert!(x0.grad() > 0.0, "deep-chain grad should stay positive");
}
