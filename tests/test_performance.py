import time
from spatialclaw.core.registry import SpatialSkillRegistry

def test_lightweight_faster_than_full():
    # Test lightweight loading
    registry1 = SpatialSkillRegistry()
    start = time.time()
    registry1.load_lightweight()
    lightweight_time = time.time() - start

    # Test full loading
    registry2 = SpatialSkillRegistry()
    start = time.time()
    registry2.load_all()
    full_time = time.time() - start

    print(f"Lightweight: {lightweight_time:.3f}s")
    print(f"Full: {full_time:.3f}s")
    print(f"Speedup: {full_time/lightweight_time:.1f}x")

    # Lightweight should be faster
    assert lightweight_time < full_time
