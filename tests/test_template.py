# Test that the Cookiecutter project generates successfully
# This requires 'pytest-cookies' to be installed in the environment


def test_bake_project(cookies):
    """Test that the Axiom template generates a valid project directory."""
    result = cookies.bake(extra_context={"project_name": "Axiom Fraud Engine"})

    assert result.exit_code == 0
    assert result.exception is None
    assert result.project_path.name == "axiom_fraud_engine"
    assert result.project_path.is_dir()
