from utils.benchmark_constituents import get_supported_benchmark_keys


def test_supported_benchmarks_include_sp_midcap_400():
    keys = set(get_supported_benchmark_keys())
    assert "SP500" in keys
    assert "SPMIDCAP400" in keys
