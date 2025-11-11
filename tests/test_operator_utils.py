from c3.operator_utils import explode_connection_string


def test_explode_connection_string():
    (ac, sc, ep, p) = explode_connection_string('cos://DF)S)DFU8:!#$%^*(){}[]"><@s3.us-east.cloud-object-storage.appdomain.cloud/claimed-test/ds=335/dl=50254/dt=20220101/tm=000000/lvl=0/gh=0/S1A_IW_GRDH_1SDV_20220101T090715_20220101T090740_041265_04E78F_73F0_VH.cog')
    assert ac=='DF)S)DFU8'
    assert sc=='!#$%^*(){}[]"><'
    assert ep=='https://s3.us-east.cloud-object-storage.appdomain.cloud'
    assert p=='claimed-test/ds=335/dl=50254/dt=20220101/tm=000000/lvl=0/gh=0/S1A_IW_GRDH_1SDV_20220101T090715_20220101T090740_041265_04E78F_73F0_VH.cog'