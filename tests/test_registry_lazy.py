from spatialclaw.core.registry import SpatialSkillRegistry


def test_registry_load_lightweight():
    registry = SpatialSkillRegistry()
    registry.load_lightweight()

    assert len(registry.lazy_skills) > 0
    assert "spatial-preprocessing" in registry.lazy_skills

    # Should have basic info
    preprocess = registry.lazy_skills["spatial-preprocessing"]
    assert preprocess.name == "spatial-preprocessing"
    assert len(preprocess.description) > 0


def test_registry_load_lightweight_includes_spatial_downstream_api_skills():
    registry = SpatialSkillRegistry()
    registry.load_lightweight()

    expected_downstream_skills = {
        "spatial-histology",
        "spatial-morphology",
        "spatial-niches",
        "spatial-oncology",
        "spatial-regulation",
        "spatial-sc2spatial",
        "spatial-targets",
        "spatial-tls",
        "spatial-translation",
        "spatial-wsi",
    }

    assert expected_downstream_skills <= set(registry.lazy_skills)
    for skill_name in expected_downstream_skills:
        metadata = registry.lazy_skills[skill_name]
        assert metadata.name == skill_name
        assert metadata.domain == "spatial"
        assert metadata.description
        assert metadata.get_full()["metadata"]["SPATIALCLAW"]["interface"] == "python-api"
