import pytest
from datetime import datetime
from quantify.data.types import TUID


def test_TUID():
    tuid = TUID('20200409-123015-123-abcdef')

    dt = tuid.datetime()
    assert isinstance(dt, datetime)
    assert isinstance(tuid, str)

    assert dt.year == 2020
    assert dt.month == 4
    assert dt.day == 9

    assert dt.hour == 12
    assert dt.minute == 30
    assert dt.second == 15

    with pytest.raises(ValueError):
        tuid = TUID('200409-123015-123-abcdef')

    with pytest.raises(ValueError):
        tuid = TUID('200409-123015-123-abcdefasf')


def test_TUID_validation():
    TUID.is_valid('20200409-123015-123-abcdef')

    # too short uid
    with pytest.raises(ValueError):
        TUID.is_valid('20200409-123015-123-a')

    # too long uid
    with pytest.raises(ValueError):
        TUID.is_valid('20200409-123015-123-a135bcdefasf')

    # missing milliseconds
    with pytest.raises(ValueError):
        TUID.is_valid('20200409-123015-abcdef')

    # 2 digit year
    with pytest.raises(ValueError):
        TUID.is_valid('200409-123015-123-abcdef')

    # missing dash
    with pytest.raises(ValueError):
        TUID.is_valid('20200409123015-123-abcdef')
