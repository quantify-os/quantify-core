import pytest
from datetime import datetime
from quantify.data.types import TUID


def test_TUID():
    tuid = TUID('20200409-123015-123-abcdef')

    dt = tuid.datetime()
    assert isinstance(dt, datetime)
    assert isinstance(tuid, str)

    dt.year == 2020
    dt.month == 4
    dt.day == 9

    dt.hour == 12
    dt.min == 30
    dt.second == 15

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
