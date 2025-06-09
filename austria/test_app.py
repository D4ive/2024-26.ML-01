import pytest
from austria.appdave import app as flask_app



@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_infer(client):
    response = client.post('/infer', json={'hyper_param_a': 5})
    assert response.status_code == 200
    data = response.get_json()
    print(f"Response data: {data}")
    assert 'result' in data
    assert isinstance(data["result"],float)
    assert "latency_seconds" in data