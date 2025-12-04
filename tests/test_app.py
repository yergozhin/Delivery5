from predict import predict


def test_predict_returns_valid_class_id():
    features = [5.1, 3.5, 1.4, 0.2]
    result = predict(features)
    assert result in [0, 1, 2]
