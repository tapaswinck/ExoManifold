from exomanifold.astronomy.mast import MASTClient

def test_client_creation():
    client = MASTClient()

    assert client.mission == "Kepler"

def test_default_cache():
    client = MASTClient()
    assert client.cache_dir is None

def test_custom_mission():
    client = MASTClient(
        mission = "TESS"
    )

    assert client.mission == "TESS"

def test_has_search():
    client = MASTClient()
    
    assert callable(client.search)

def test_has_download():
    client = MASTClient()

    assert callable(client.download_lightcurve)



    