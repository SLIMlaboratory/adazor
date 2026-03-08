def test_public_api():
    import adazor

    assert hasattr(adazor, "GaussianMixture")
    assert hasattr(adazor, "DriftingMixtureStream")
    assert hasattr(adazor, "NoDriftDetector")
    assert hasattr(adazor, "ThresholdDriftDetector")
    assert hasattr(adazor, "ZTestDriftDetector")
    assert hasattr(adazor, "HoeffdingTreeClassifier")
