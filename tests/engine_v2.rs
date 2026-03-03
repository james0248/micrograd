use micrograd::engine_v2::{Op, Value, clear_graph, no_grad, reset_state, stats, with_grad};

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
fn value_new_requires_active_context() {
    reset_state();
    let result = std::panic::catch_unwind(|| Value::new(1.0));
    assert!(result.is_err(), "Value::new should panic outside context");
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

    let analytical = with_grad(|| {
        let x = Value::new(x0);
        let xx = x.mul(&x);
        let s = xx.add(&x);
        let out = s.tanh();
        out.backward();
        x.grad()
    });

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
fn with_grad_cannot_be_nested() {
    reset_state();
    let result = std::panic::catch_unwind(|| {
        with_grad(|| {
            with_grad(|| {
                let _ = Value::new(1.0);
            });
        });
    });
    assert!(result.is_err(), "nested with_grad should panic");
}

#[test]
fn temps_are_stale_after_context_exit() {
    reset_state();
    let y = with_grad(|| {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        a.mul(&b)
    });

    let stale = std::panic::catch_unwind(|| y.data());
    assert!(stale.is_err(), "temp from closed context should be stale");
}

#[test]
fn backward_requires_with_grad_context() {
    reset_state();
    let out = with_grad(|| {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        a.mul(&b)
    });

    let result = std::panic::catch_unwind(|| out.backward());
    assert!(result.is_err(), "backward should panic outside with_grad");
}

#[test]
fn backward_with_options_is_compatible() {
    reset_state();
    with_grad(|| {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let out = a.mul(&b);

        out.backward_with_options(true);
        assert_close(a.grad(), 3.0, 1e-12);
        out.backward();
        assert_close(a.grad(), 6.0, 1e-12);
    });
}

#[test]
fn two_losses_in_same_context_work() {
    reset_state();
    let w = Value::parameter(1.5);

    with_grad(|| {
        let x1 = Value::new(2.0);
        let l1 = w.mul(&x1);
        l1.backward();
        assert_close(w.grad(), 2.0, 1e-12);

        let x2 = Value::new(-3.0);
        let l2 = w.mul(&x2);
        l2.backward();
        assert_close(w.grad(), -1.0, 1e-12);
    });
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
