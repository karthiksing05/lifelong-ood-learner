"""Script-style tests (each a standalone `python tests/x.py` with a module-level exit) are run by
tests/test_scripts.py in subprocesses, so pytest must not import them itself."""
SCRIPT_TESTS = ["test_keyed_lora.py", "test_layout.py", "test_field_contract.py", "test_sample_concept_path.py",
                "test_shapes3d_rows.py", "test_factored.py", "test_feature_space.py", "test_fid.py",
                "test_cobweb_correspondence.py"]
collect_ignore = list(SCRIPT_TESTS)
