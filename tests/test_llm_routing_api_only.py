from spatialclaw.core.registry import API_ONLY_SPATIAL_SKILLS
from spatialclaw.routing.llm_router import filter_cli_routing_skills


def test_llm_cli_routing_excludes_api_only_skills():
    skills = {
        "spatial-preprocessing": "CLI skill",
        **{name: "Python API only" for name in API_ONLY_SPATIAL_SKILLS},
    }

    filtered = filter_cli_routing_skills(skills)

    assert filtered == {"spatial-preprocessing": "CLI skill"}
