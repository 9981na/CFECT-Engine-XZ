"""Quick integration test for all 6 new DeepSeek RedTeam modules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []

# Test 1: cfect_core imports
try:
    from cfect_core import (
        RollingSolver, GrayBoxPINN, PathDecoder,
        compute_forward_backward_ratio, compute_entropy_production,
        verify_thermodynamic_consistency, honest_reparameterization,
        effective_info, compute_phi_e, renorm,
        spatial_variance, morans_i, spatial_skewness_kurtosis, verify_spatial_ews
    )
    print("[PASS] cfect_core: all 14 symbols imported")
except Exception as e:
    errors.append(f"cfect_core import: {e}")
    print(f"[FAIL] cfect_core: {e}")

# Test 2: statistics imports
try:
    from statistics import (
        LMMEvaluator, OLSTrendFlow, FDRChronoBin,
        verify_csd, mann_kendall_trend, compute_confidence_interval,
        augment_with_dickey_fuller,
        adf_stationarity_test, check_epoch_stationarity, adaptive_max_scale
    )
    print("[PASS] statistics: all 10 symbols imported")
except Exception as e:
    errors.append(f"statistics import: {e}")
    print(f"[FAIL] statistics: {e}")

# Test 3: critical_slowing_test functional
import numpy as np
try:
    from statistics.critical_slowing_test import verify_csd
    ac = np.linspace(0.1, 0.5, 50)
    var = np.linspace(0.5, 1.2, 50)
    result = verify_csd(ac, var, ar1_threshold=0.1, var_increase_pct=30)
    assert result["pass"] == True, f"CSD should pass with monotonic data, got {result}"
    print(f"[PASS] verify_csd: pass={result['pass']}, ar1={result['ar1_increase']:.4f}, var={result['var_increase_pct']:.1f}%")
except Exception as e:
    errors.append(f"critical_slowing_test: {e}")
    print(f"[FAIL] critical_slowing_test: {e}")

# Test 4: fluctuation_theorem
try:
    from cfect_core.fluctuation_theorem import honest_reparameterization, compute_entropy_production
    v = np.random.randn(50)
    ac = np.random.randn(50) * 0.3 + 0.5
    hr = honest_reparameterization(v, ac)
    assert len(hr["complex_state"]) == 50
    ep = compute_entropy_production(hr["complex_state"])
    print(f"[PASS] fluctuation_theorem: entropy slope={ep['slope']:.4f}")
except Exception as e:
    errors.append(f"fluctuation_theorem: {e}")
    print(f"[FAIL] fluctuation_theorem: {e}")

# Test 5: integrated_info
try:
    np.random.seed(42)
    ts = np.cumsum(np.random.randn(200))
    phi = compute_phi_e(ts, tau=5)
    print(f"[PASS] compute_phi_e: phi_full={phi['phi_e_full']:.4f}, phi_coarse={phi['phi_e_coarse']:.4f}, ratio={phi['ratio']:.4f}")
except Exception as e:
    errors.append(f"integrated_info: {e}")
    print(f"[FAIL] integrated_info: {e}")

# Test 6: spatial_ews
try:
    data = np.random.randn(100, 8)
    sv = spatial_variance(data)
    sw = verify_spatial_ews(data)
    print(f"[PASS] spatial_ews: variance shape={sv.shape}, verdict={sw['verdict']['pass']}")
except Exception as e:
    errors.append(f"spatial_ews: {e}")
    print(f"[FAIL] spatial_ews: {e}")

# Test 7: stationarity_prescreen
try:
    from statistics.stationarity_prescreen import adf_stationarity_test
    adf = adf_stationarity_test(np.random.randn(500))
    print(f"[PASS] adf_stationarity_test: stationary={adf['is_stationary']}, p={adf['p_value']:.4f}")
except Exception as e:
    errors.append(f"stationarity_prescreen: {e}")
    print(f"[FAIL] stationarity_prescreen: {e}")

# Test 8: n1_baseline_comparison
try:
    from pipelines.n1_baseline_comparison import compute_spectral_baseline, estimate_n1_expected_f1, detect_synthetic_data_n1_pattern
    theta = np.random.randn(100) * 0.1 + 0.3
    delta = np.random.randn(100) * 0.1 + 0.5
    spec = compute_spectral_baseline(theta, delta)
    exp = estimate_n1_expected_f1(spec)
    synthetic = detect_synthetic_data_n1_pattern(np.array([0.5, 0.10, 0.25, 0.35, 0.15, 0.15]))
    print(f"[PASS] n1_baseline: expected F1={exp['expected_f1_range']}, synthetic_detect={synthetic}")
except Exception as e:
    errors.append(f"n1_baseline: {e}")
    print(f"[FAIL] n1_baseline: {e}")

# Summary
print()
if errors:
    print(f"RESULT: {len(errors)} FAILURES")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL 8 TESTS PASSED")
    print("All 6 new DeepSeek RedTeam modules integrate successfully.")
